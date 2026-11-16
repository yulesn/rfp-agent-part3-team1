"""Short-ID, source-unit and cache invariants. All responses are offline doubles."""
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rfp_rag.generation.llm import LLMConfig, LLMError
from rfp_rag.summary_fields.fields import FIELD_QUESTION, FIELD_SCHEMA

import rfp_rag.summary_full.transport as t


def evidence(*texts, did='canonical-document-00000001', sha='a'*64):
    return [{'chunk_id': f'rfp:{sha[:16]}:{i:024x}', 'doc_id': did, 'canonical_doc_id': did,
        'sha256': sha, 'text': text, 'file_name': '공개_RFP.pdf', 'locator': f'PDF p.{i+1}',
        'source_method': 'raw_parser', 'quality_flags': [], 'source_spans': []}
        for i, text in enumerate(texts)]


def response_payload(alias='E001', doc='D001', quote='SOURCE'):
    return {'status': 'answered', 'claims': [{'doc_id': doc, 'category': 'budget',
        'text': '[budget_vat|fact] 예산의 원문 연결을 검사하는 합성 테스트 주장',
        'citations': [{'chunk_id': alias, 'quote': quote}]}], 'missing_information': []}


class FakeClient:
    def __init__(self):
        self.requests = []
        self.responses = self
        self.payload_override = None

    def create(self, **body):
        self.requests.append(copy.deepcopy(body))
        context = json.loads(body['input'][1]['content'])
        key = next(e['chunk_id'] for e in context['evidence'] if e['citable'])
        payload = self.payload_override or response_payload(alias=key)
        return SimpleNamespace(status='completed', output=[], output_text=json.dumps(payload),
            usage=SimpleNamespace(input_tokens=12, output_tokens=8, total_tokens=20,
                input_tokens_details=SimpleNamespace(cached_tokens=0)))


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.client = FakeClient()
        self.chunks = evidence('사업예산: 100원(부가세 포함).\r\n기간: 5개월.\n', '제출기한은 공고문을 참조합니다.')
        self.llm = t.ShortCitationLLM(self.root/'cache', config=LLMConfig(enabled=True, max_output_tokens=6000),
            allowed_sources={self.chunks[0]['doc_id']: 'a'*64}, client=self.client)

    def descriptor(self, chunks=None):
        return self.llm.canonical_request_descriptor('summary', FIELD_QUESTION,
            self.chunks if chunks is None else chunks, schema=FIELD_SCHEMA, include_wire_request=True)

    def test_model_receives_only_short_identity_fields_and_fixed_quote(self):
        d = self.descriptor(); body = d['wire_request']
        request = json.loads(body['input'][1]['content'])
        self.assertEqual(request['document'], {'doc_id': 'D001', 'file_name': '공개_RFP.pdf'})
        for row in request['evidence']:
            self.assertEqual(set(row), {'chunk_id', 'doc_id', 'text', 'citable'})
            self.assertRegex(row['chunk_id'], r'^E\d{3,}$')
            self.assertEqual(row['doc_id'], 'D001')
            self.assertNotIn('sha256', row)
        serialized = json.dumps(body)
        for c in self.chunks:
            self.assertNotIn(c['chunk_id'], serialized)
            self.assertNotIn(c['doc_id'], serialized)
        props = body['text']['format']['schema']['properties']['claims']['items']['properties']
        self.assertEqual(props['doc_id']['enum'], ['D001'])
        self.assertEqual(props['citations']['items']['properties']['quote']['enum'], ['SOURCE'])

    def test_exact_source_partition_handles_korean_emoji_crlf_and_whitespace(self):
        texts = ['한글🙂'*300+'\r\n둘째 줄\n셋째 줄\r\n넷째 줄\n다섯째 줄', '\n'*17+'끝', '한 줄', '  \t\r\n']
        for text in texts:
            with self.subTest(text=text[:10]):
                units = t.partition_source(text)
                self.assertEqual(''.join(u['quote'] for u in units), text)
                previous = 0
                for u in units:
                    self.assertEqual(u['char_start'], previous)
                    self.assertEqual(text[u['char_start']:u['char_end']], u['quote'])
                    self.assertLessEqual(len(u['quote'].encode()), 600)
                    self.assertLessEqual(len(u['quote'].splitlines()), 4)
                    previous = u['char_end']
                self.assertEqual(previous, len(text))

    def test_whitespace_only_units_not_citable_but_coverage_is_complete(self):
        chunks = evidence('\n'*17+'사업예산은 100원입니다.')
        d = self.descriptor(chunks)
        self.assertTrue(d['unit_coverage']['passed'])
        self.assertTrue(any(not u['citable'] for u in d['mapping']['units']))
        allowed = d['wire_schema']['properties']['claims']['items']['properties']['citations']['items']['properties']['chunk_id']['enum']
        for unit in d['mapping']['units']:
            self.assertEqual(unit['alias'] in allowed, unit['citable'])

    def test_decode_uses_exact_substring_and_does_not_mutate_payload_or_descriptor(self):
        d = self.descriptor(); p = response_payload()
        original = copy.deepcopy((d, p))
        decoded, bindings = t.decode_payload(p, d)
        self.assertEqual((d, p), original)
        self.assertEqual(decoded['claims'][0]['doc_id'], self.chunks[0]['doc_id'])
        self.assertEqual(decoded['claims'][0]['citations'][0]['chunk_id'], self.chunks[0]['chunk_id'])
        unit = d['mapping']['units'][0]
        self.assertEqual(decoded['claims'][0]['citations'][0]['quote'], unit['quote'])
        self.assertEqual(bindings[0]['char_start'], unit['char_start'])
        self.assertEqual(bindings[0]['char_end'], unit['char_end'])

    def test_repeated_unit_text_preserves_selected_offset(self):
        chunks = evidence(('반복된 문장입니다.\n'*4)*3)
        d = self.descriptor(chunks)
        decoded, bindings = t.decode_payload(response_payload(alias='E003'), d)
        self.assertEqual(decoded['claims'][0]['citations'][0]['quote'], d['mapping']['units'][2]['quote'])
        self.assertGreater(bindings[0]['char_start'], 0)

    def test_unknown_alias_wrong_doc_canonical_id_and_quote_copy_rejected(self):
        cases = [response_payload(alias='E9999'), response_payload(doc='D002'),
                 response_payload(doc=self.chunks[0]['doc_id']),
                 response_payload(alias=self.chunks[0]['chunk_id']),
                 response_payload(quote='사업예산: 100원(부가세 포함).')]
        for p in cases:
            with self.subTest(payload=p), self.assertRaises(t.TransportError):
                t.decode_payload(p, self.descriptor())

    def test_tampered_mapping_or_binding_rejected(self):
        for mode in ['quote', 'offset', 'hash', 'foreign_request', 'top_chunk_order', 'wire_schema', 'coverage']:
            d = self.descriptor()
            if mode == 'quote': d['mapping']['units'][0]['quote'] = '위조된 원문'
            if mode == 'offset': d['mapping']['units'][0]['char_start'] = 1
            if mode == 'hash': d['mapping_fingerprint'] = 'f'*64
            if mode == 'top_chunk_order': d['canonical_chunk_ids'].reverse()
            if mode == 'wire_schema': d['wire_schema']['description'] = 'modified schema'
            if mode == 'coverage': d['unit_coverage']['passed'] = False
            with self.subTest(mode=mode), self.assertRaises(t.TransportError):
                t.decode_payload(response_payload(), d,
                    expected_mapping_fingerprint='e'*64 if mode == 'foreign_request' else None)

    def test_duplicate_chunks_scope_sha_and_multiple_docs_rejected(self):
        cases = [self.chunks+self.chunks[:1], evidence('본문', did='outside'), evidence('본문', sha='b'*64)]
        for chunks in cases:
            with self.subTest(chunks=chunks), self.assertRaises(LLMError): self.descriptor(chunks)
        llm = t.ShortCitationLLM(self.root/'multi', config=self.llm.config,
            allowed_sources={self.chunks[0]['doc_id']:'a'*64,'other-document':'b'*64})
        with self.assertRaises(t.TransportError):
            llm.canonical_request_descriptor('summary', FIELD_QUESTION,
                self.chunks+evidence('다른 문서',did='other-document',sha='b'*64))

    def test_mapping_binding_prevents_cross_document_and_cross_batch_alias_reuse(self):
        first = self.descriptor()
        second = self.descriptor(evidence('다른 본문'))
        self.assertNotEqual(first['mapping_fingerprint'], second['mapping_fingerprint'])
        self.assertNotEqual(first['request_fingerprint'], second['request_fingerprint'])
        other = evidence(self.chunks[0]['text'], did='other-document')
        llm = t.ShortCitationLLM(self.root/'other', config=self.llm.config, allowed_sources={'other-document':'a'*64})
        third = llm.canonical_request_descriptor('summary', FIELD_QUESTION, other)
        self.assertNotEqual(first['request_fingerprint'], third['request_fingerprint'])

    def test_measure_preflight_and_generate_use_identical_request_and_schema(self):
        before = copy.deepcopy((self.chunks, FIELD_SCHEMA))
        d = self.descriptor()
        size = self.llm.request_input_bytes('summary', FIELD_QUESTION, self.chunks, schema=FIELD_SCHEMA)
        self.assertEqual(size, d['request_input_bytes'])
        with patch.dict(os.environ, {'OPENAI_API_KEY':'synthetic-test-key'}), patch('socket.socket.connect', side_effect=AssertionError('network prohibited')):
            preflight = self.llm.preflight_requests([{'task':'summary','question':FIELD_QUESTION,'evidence':self.chunks,'schema':FIELD_SCHEMA}])
            out = self.llm.generate('summary',FIELD_QUESTION,self.chunks,schema=FIELD_SCHEMA)
        self.assertEqual(preflight['input_bytes_to_reserve'], size)
        self.assertEqual(self.client.requests, [d['wire_request']])
        self.assertEqual(out['request_fingerprint'], d['request_fingerprint'])
        self.assertEqual((self.chunks, FIELD_SCHEMA), before)
        self.assertEqual(out['execution_mode'], 'injected_test_client')
        self.assertFalse(out['live_api'])
        self.assertFalse(out['citation_transport']['semantic_quality_verified'])

    def test_cache_stores_alias_payload_and_offline_replay_needs_no_key_or_sdk(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY':'synthetic-test-key'}):
            first = self.llm.generate('summary',FIELD_QUESTION,self.chunks)
        cache_bytes = self.llm._cache_path().read_bytes()
        before = self.llm.usage_report()
        with (patch.dict(os.environ, {}, clear=True),
              patch.object(self.llm,'_sdk',side_effect=AssertionError('SDK prohibited')),
              patch('socket.socket.connect', side_effect=AssertionError('network prohibited'))):
            replay = self.llm.offline_replay('summary',FIELD_QUESTION,self.chunks)
        self.assertEqual(self.llm.usage_report(), before)
        self.assertEqual(cache_bytes, self.llm._cache_path().read_bytes())
        self.assertEqual(replay['usage']['api_calls'],0)
        self.assertEqual(replay['raw_alias_payload']['claims'][0]['doc_id'],'D001')
        self.assertEqual(replay['raw_alias_payload']['claims'][0]['citations'][0]['quote'],'SOURCE')
        self.assertEqual(replay['payload'], first['payload'])
        self.assertFalse(replay['live_api'])

    def test_schema_copied_per_request_and_invalid_vendor_alias_is_rejected_locally(self):
        original = copy.deepcopy(FIELD_SCHEMA)
        d1 = self.descriptor(self.chunks[:1]);d2 = self.descriptor(self.chunks)
        self.assertNotEqual(d1['wire_schema_fingerprint'],d2['wire_schema_fingerprint'])
        self.assertEqual(FIELD_SCHEMA,original)
        self.client.payload_override = response_payload(alias='E9999')
        with patch.dict(os.environ, {'OPENAI_API_KEY':'synthetic-test-key'}), self.assertRaises(t.TransportError):
            self.llm.generate('summary',FIELD_QUESTION,self.chunks)
        self.assertEqual(self.llm.usage_report()['transport_decode_failures'],1)

    def test_disabled_generation_still_plans_and_replay_does_not_generate(self):
        llm = t.ShortCitationLLM(self.root/'disabled',config=replace(self.llm.config,enabled=False),
            allowed_sources=self.llm.allowed_sources)
        with patch.dict(os.environ,{},clear=True), patch.object(llm,'_sdk',side_effect=AssertionError('SDK prohibited')):
            self.assertGreater(llm.request_input_bytes('summary',FIELD_QUESTION,self.chunks),0)
            self.assertIsNone(llm.offline_replay('summary',FIELD_QUESTION,self.chunks))
            with self.assertRaises(LLMError):llm.generate('summary',FIELD_QUESTION,self.chunks)

    def test_field_category_alignment_is_left_to_claim_wise_engine_validation(self):
        payload = response_payload()
        invalid = copy.deepcopy(payload['claims'][0])
        invalid['text'] = '[duration|fact] 잘못된 category 대응을 개별 거절할 테스트 주장'
        payload['claims'].append(invalid)
        decoded, bindings = t.decode_payload(payload, self.descriptor())
        self.assertEqual(len(decoded['claims']), 2)
        self.assertEqual(len(bindings), 2)
        self.assertEqual(decoded['claims'][1]['category'], 'budget')


if __name__ == '__main__': unittest.main()
