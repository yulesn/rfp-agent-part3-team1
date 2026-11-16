"""Synthetic cache fixtures verify integrity/source replay, never model quality."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from rfp_rag.generation.evidence import EvidenceRegistry
from rfp_rag.generation.llm import LLMConfig, OpenAIGroundedLLM
from rfp_rag.summary_fields.fields import FIELD_QUESTION, FIELD_SCHEMA

import rfp_rag.summary_full.legacy as legacy


def serial(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


class Index:
    def __init__(self, chunks): self.chunks = chunks
    def get_document_chunks(self, did): return [c for c in self.chunks if c['doc_id'] == did]


class LegacyCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.blocks, self.chunks = [], []
        for n in range(3):
            text = f'사업예산: {100+n}원(부가세 포함).'
            block = {'block_id': f'b{n}', 'text': text, 'locator': f'PDF p.{n+1}', 'page': n+1}
            self.blocks.append(block)
            self.chunks.append({'chunk_id': f'c{n}', 'doc_id': 'd1', 'canonical_doc_id': 'd1',
                'sha256': 'a'*64, 'file_name': '공개.pdf', 'text': text,
                'text_bytes': len(text.encode()), 'length_unit': 'utf8_bytes', 'kind': 'prose',
                'locator': block['locator'], 'source_method': 'raw_parser', 'page': n+1,
                'source_spans': [{'collection': 'blocks', 'block_id': block['block_id'],
                    'locator': block['locator'], 'page': n+1, 'char_start': 0, 'char_end': len(text),
                    'chunk_start': 0, 'chunk_end': len(text)}]})
        doc = {'doc_id': 'd1', 'canonical_doc_id': 'd1', 'sha256': 'a'*64,
               'file_name': '공개.pdf', 'blocks': self.blocks, 'supplemental_blocks': []}
        self.env = {'root': self.root, 'index': Index(self.chunks),
                    'registry': EvidenceRegistry([doc], self.chunks)}
        self.path = self.root/'phase4_outputs/generation_cache/grounded_responses.sqlite3'
        self.llm = OpenAIGroundedLLM(self.path.parent,
            config=LLMConfig(enabled=True, model='gpt-5.4-mini', max_output_tokens=6000),
            allowed_sources={'d1': 'a'*64})
        prepared = self.llm._prepare(legacy.LEGACY_TASK, FIELD_QUESTION, self.chunks[:2], schema=FIELD_SCHEMA)
        self.exact = json.loads(prepared['exact'])
        self.payload = {'status': 'answered', 'claims': [{'doc_id': 'd1', 'category': 'budget',
            'text': '[budget_vat|fact] 사업예산은 100원이며 부가세 포함입니다.',
            'citations': [{'chunk_id': 'c0', 'quote': self.chunks[0]['text']}]}], 'missing_information': []}

    def install(self, exact=None, payload=None, *, mode='live_openai', live=True,
                fingerprint_override=None, checksum_override=None):
        exact = copy.deepcopy(self.exact if exact is None else exact)
        request_text = serial(exact)
        fp = hashlib.sha256(request_text.encode()).hexdigest()
        response = {'payload': copy.deepcopy(self.payload if payload is None else payload),
            'execution_mode': mode, 'live_api': live, 'request_fingerprint': fp,
            'usage': {'input_tokens': 12, 'output_tokens': 8, 'total_tokens': 20}}
        response_text = serial(response)
        checksum = hashlib.sha256((request_text+'\n'+response_text).encode()).hexdigest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute('DROP TABLE IF EXISTS responses')
            connection.execute('CREATE TABLE responses(fingerprint TEXT PRIMARY KEY,request_json TEXT,response_json TEXT,checksum TEXT)')
            connection.execute('INSERT INTO responses VALUES(?,?,?,?)',
                (fingerprint_override or fp, request_text, response_text, checksum_override or checksum))
        return fp

    def scan(self, model='gpt-5.4-mini'):
        with patch('socket.socket', side_effect=AssertionError('network forbidden')), patch.dict('os.environ', {}, clear=True):
            return legacy.scan_legacy_batches(self.env, model)

    def mutate_user(self, exact, fn):
        user = json.loads(exact['request']['input'][1]['content'])
        fn(user)
        exact['request']['input'][1]['content'] = serial(user)

    def test_valid_cache_is_read_only_and_preserves_original_request_identity(self):
        fp = self.install(); before = self.path.read_bytes()
        report = self.scan(); self.assertEqual(before, self.path.read_bytes())
        self.assertEqual(report['candidate_count'], 1)
        candidate = report['candidates'][0]
        self.assertEqual(candidate['legacy_request_fingerprint'], fp)
        self.assertEqual(candidate['source_start'], 0)
        self.assertEqual(candidate['source_end'], 2)
        self.assertEqual(candidate['source_chunk_ids'], ['c0', 'c1'])
        self.assertEqual(candidate['origin'], 'legacy_literal_cache')
        self.assertNotIn('request_fingerprint', candidate)
        self.assertTrue(candidate['validated_claims'][0]['citations'][0]['provenance_verified'])
        self.assertFalse(candidate['semantic_quality_verified'])
        self.assertEqual(report['additional_api_calls'], 0)
        self.assertEqual(candidate['candidate_fingerprint'], self.scan()['candidates'][0]['candidate_fingerprint'])

    def test_absent_cache_is_normal_zero_call_state(self):
        report = self.scan()
        self.assertEqual(report['status'], 'cache_absent')
        self.assertEqual(report['candidate_count'], 0)
        self.assertEqual(report['additional_api_calls'], 0)

    def test_request_hash_and_checksum_mismatch_are_rejected(self):
        for kwargs, expected in [({'fingerprint_override': 'b'*64}, 'request_fingerprint_mismatch'),
                                 ({'checksum_override': 'c'*64}, 'cache_checksum_mismatch')]:
            with self.subTest(expected=expected):
                self.install(**kwargs); r = self.scan()
                self.assertEqual(r['candidate_count'], 0)
                self.assertEqual(r['skipped'][0]['reason'], expected)

    def test_request_evidence_text_sha_and_metadata_must_match_current_source(self):
        for key, value in [('text', '변경된 원문'), ('sha256', 'b'*64), ('file_name', '다른.pdf')]:
            with self.subTest(key=key):
                exact = copy.deepcopy(self.exact)
                self.mutate_user(exact, lambda u: u['evidence'][0].update({key: value}))
                self.install(exact=exact); r = self.scan()
                self.assertEqual(r['candidate_count'], 0)
                self.assertEqual(r['skipped'][0]['reason'], 'legacy_source_content_or_version_mismatch')

    def test_original_source_tampering_cannot_be_hidden_by_valid_cache(self):
        self.install()
        self.chunks[0]['text'] = '사후 조작된 원문'
        self.chunks[0]['text_bytes'] = len(self.chunks[0]['text'].encode())
        self.assertEqual(self.scan()['candidate_count'], 0)

    def test_mock_or_nonlive_response_never_accepted_as_actual_model(self):
        for mode, live in [('injected_test_client', False), ('live_openai', False), ('injected_test_client', True)]:
            with self.subTest(mode=mode, live=live):
                self.install(mode=mode, live=live)
                r = self.scan(); self.assertEqual(r['candidate_count'], 0)
                self.assertEqual(r['skipped'][0]['reason'], 'not_actual_model_origin')

    def test_wrong_task_question_model_schema_and_system_are_rejected(self):
        for kind in ['task', 'question', 'model', 'schema', 'system']:
            with self.subTest(kind=kind):
                exact = copy.deepcopy(self.exact)
                if kind == 'task': self.mutate_user(exact, lambda u: u.update(task='qa'))
                if kind == 'question': self.mutate_user(exact, lambda u: u.update(question='다른 질문'))
                if kind == 'model': exact['request']['model'] = 'other-model'
                if kind == 'schema': exact['request']['text']['format']['schema']['properties']['claims']['maxItems'] = 2
                if kind == 'system': exact['request']['input'][0]['content'] += ' 변경'
                self.install(exact=exact)
                self.assertEqual(self.scan()['candidate_count'], 0)

    def test_invented_quote_or_invalid_field_marker_rejects_whole_candidate(self):
        for kind in ['quote', 'field']:
            with self.subTest(kind=kind):
                payload = copy.deepcopy(self.payload)
                if kind == 'quote': payload['claims'][0]['citations'][0]['quote'] = '원문에 없는 999조원'
                if kind == 'field': payload['claims'][0]['text'] = '표지가 없는 모델 주장'
                self.install(payload=payload)
                self.assertEqual(self.scan()['candidate_count'], 0)

    def test_noncontiguous_reordered_and_duplicate_evidence_rejected(self):
        for kind in ['gap', 'reorder', 'duplicate']:
            with self.subTest(kind=kind):
                exact = copy.deepcopy(self.exact)
                def change(user):
                    if kind == 'gap': user['evidence'][1] = {k:self.chunks[2].get(k) for k in legacy._EVIDENCE_KEYS}
                    if kind == 'reorder': user['evidence'].reverse()
                    if kind == 'duplicate': user['evidence'][1] = copy.deepcopy(user['evidence'][0])
                self.mutate_user(exact, change); self.install(exact=exact)
                self.assertEqual(self.scan()['candidate_count'], 0)

    def test_unreadable_database_reports_safe_failure_without_raw_error(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text('not a sqlite database; sk-private-do-not-echo')
        report = self.scan()
        self.assertEqual(report['status'], 'cache_unreadable')
        self.assertNotIn('sk-private', json.dumps(report))

    def test_bad_fingerprint_is_not_echoed_in_diagnostics(self):
        self.install(fingerprint_override='sk-private-do-not-echo')
        report = self.scan()
        self.assertNotIn('sk-private', json.dumps(report))
        self.assertIsNone(report['skipped'][0]['legacy_request_fingerprint'])


if __name__ == '__main__':
    unittest.main()
