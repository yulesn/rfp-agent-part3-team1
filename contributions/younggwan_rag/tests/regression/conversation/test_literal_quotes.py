"""Closed quotation choices, strict failure and aggregate budget invariants."""
import copy
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rfp_rag.generation.evidence import EvidenceRegistry
from rfp_rag.generation.llm import LLMConfig, LLMError, OpenAIGroundedLLM, COMPARISON_DOCUMENT_TASK
from rfp_rag.conversation.literal_quotes import (QA_TASK, LiteralQuoteClient, accepts_citations,
                                     literal_options, literal_schema, quote_references)
from rfp_rag.conversation.runtime import make_conversation
from regression.conversation.test_focus import GroundedClient, synthetic_environment


def citations(schema):
    return schema['properties']['claims']['items']['properties']['citations']['items']['properties']


class LiteralQuoteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = synthetic_environment(Path(self.tmp.name))
        self.addCleanup(self.env['index'].close)
        key = patch.dict(os.environ, {'OPENAI_API_KEY': 'synthetic-not-a-real-key'})
        key.start()
        self.addCleanup(key.stop)

    def llm(self, *, max_calls=8, cache=False, client=None):
        client = client or GroundedClient()
        delegate = OpenAIGroundedLLM(self.env['root'] / 'cache',
            LLMConfig(enabled=True, max_calls=max_calls, cache_enabled=cache),
            allowed_sources=self.env['registry'].source_hashes, client=client)
        return LiteralQuoteClient(delegate), delegate, client

    def test_each_choice_is_a_contiguous_original_substring_and_preserves_characters(self):
        text = '대상 업무\nFUN-공통-101 사용 승인\n' + '반출 수량은 137개, 부가세 포함. ' * 80
        options = literal_options(text)
        self.assertTrue(options)
        self.assertTrue(all(q in text for q in options))
        covered = set()
        for q in options:
            start = text.find(q)
            while start >= 0:
                covered.update(range(start, start + len(q)))
                start = text.find(q, start + 1)
        self.assertTrue(all(c.isspace() or i in covered for i, c in enumerate(text)))
        self.assertTrue(any('137개' in q for q in options))

    def test_ellipsis_joined_outline_is_not_a_permitted_quote(self):
        chunks = self.env['index'].get_document_chunks('A')[-4:]
        schema = literal_schema(chunks)
        cid = chunks[0]['chunk_id']
        payload = {'claims': [{'doc_id': 'A', 'citations': [{'chunk_id': cid,
            'quote': '사업 범위 ... 반출 신청 ... 승인 처리 ... 반환 이력'}]}]}
        self.assertFalse(accepts_citations(payload, schema, chunks))

    def test_source_ellipsis_is_allowed_only_when_it_is_actually_in_the_source(self):
        chunks = copy.deepcopy(self.env['index'].get_document_chunks('A')[-1:])
        chunks[0]['text'] = '원문 자체의 표기 ... 를 그대로 보존한다.'
        schema = literal_schema(chunks)
        self.assertEqual(citations(schema)['quote']['enum'], ['q0001'])
        self.assertEqual(quote_references(chunks)['q0001']['quote'], chunks[0]['text'])

    def test_flat_schema_and_local_binding_preserve_actual_chunk_ownership(self):
        chunks = self.env['index'].get_document_chunks('A')[-4:]
        schema = literal_schema(chunks)
        self.assertEqual(citations(schema)['chunk_id']['enum'], [c['chunk_id'] for c in chunks])
        self.assertFalse(schema['properties']['claims']['items']['properties']['citations']['items']['additionalProperties'])
        for ref, choice in quote_references(chunks).items():
            self.assertIn(ref, citations(schema)['quote']['enum'])
            payload = {'status': 'answered', 'claims': [{
                'doc_id': 'A', 'category': 'requirements', 'text': choice['quote'],
                'citations': [{'chunk_id': choice['chunk_id'], 'quote': ref}]}],
                'missing_information': []}
            self.assertTrue(accepts_citations(payload, schema, chunks))
        self.assertEqual(ANSWER_BASELINE, self._schema_hash())

    def test_flat_field_enums_never_approve_a_cross_paired_quote(self):
        chunks = self.env['index'].get_document_chunks('A')[-4:]
        schema = literal_schema(chunks)
        first, second = chunks[0], chunks[-1]
        ref = next(r for r, c in quote_references(chunks).items() if c['chunk_id'] == first['chunk_id'])
        self.assertIn(ref, citations(schema)['quote']['enum'])
        self.assertIn(second['chunk_id'], citations(schema)['chunk_id']['enum'])
        payload = {'claims': [{'doc_id': 'A', 'citations': [
            {'chunk_id': second['chunk_id'], 'quote': ref}]}]}
        self.assertFalse(accepts_citations(payload, schema, chunks))

    def test_flat_field_enums_never_approve_a_cross_document_quote(self):
        chunks = self.env['index'].get_document_chunks('A')[-1:] + self.env['index'].get_document_chunks('B')[-1:]
        schema = literal_schema(chunks)
        ref = next(iter(quote_references(chunks)))
        payload = {'claims': [{'doc_id': 'B', 'citations': [
            {'chunk_id': chunks[0]['chunk_id'], 'quote': ref}]}]}
        self.assertFalse(accepts_citations(payload, schema, chunks))

    def test_wire_schema_contains_only_existing_primitives_and_flat_enums(self):
        schema = literal_schema(self.env['index'].get_document_chunks('A')[-4:])
        def walk(node):
            self.assertFalse({'anyOf', 'oneOf', 'allOf', '$ref', 'if', 'then'} & set(node))
            if node['type'] == 'object':
                self.assertFalse(node['additionalProperties'])
                self.assertEqual(set(node['required']), set(node['properties']))
                for child in node['properties'].values():
                    walk(child)
            elif node['type'] == 'array':
                walk(node['items'])
            else:
                self.assertEqual(node['type'], 'string')
                if 'enum' in node:
                    self.assertEqual(len(node['enum']), len(set(node['enum'])))
        walk(schema)

    def test_wire_uses_ascii_references_and_preserves_source_text_and_menu_quotes(self):
        wrapper, delegate, client = self.llm()
        chunks = self.env['index'].get_document_chunks('A')[-4:]
        before = copy.deepcopy(chunks)
        result = wrapper.generate('qa', '구축 대상 업무는?', chunks)
        body = client.bodies[0]
        self.assertIn('citations[].quote에는 원문 문자열 대신', body['input'][0]['content'])
        refs = citations(body['text']['format']['schema'])['quote']['enum']
        self.assertTrue(all(re.fullmatch(r'q[0-9]{4}', ref) for ref in refs))
        wire_chunks = json.loads(body['input'][1]['content'])['evidence']
        self.assertEqual([c['text'] for c in wire_chunks], [c['text'] for c in chunks])
        self.assertEqual([c['sha256'] for c in wire_chunks], [c['sha256'] for c in chunks])
        menu = [(c, option) for c in wire_chunks for option in c['citation_options']]
        self.assertEqual({o['ref'] for _, o in menu}, set(refs))
        self.assertTrue(all(o['quote'] in c['text'] for c, o in menu))
        self.assertTrue(any('\n' in o['quote'] for _, o in menu))
        self.assertEqual(chunks, before)
        self.assertTrue(result['literal_citation_contract']['reference_ids_resolved_locally'])
        self.assertEqual(delegate.usage_report()['mock_calls'], 1)

    def test_every_reference_is_expanded_before_registry_checks_and_after_cache_hits(self):
        class AllOptionsClient(GroundedClient):
            def create(inner, **body):
                inner.bodies.append(body)
                chunks = json.loads(body['input'][1]['content'])['evidence']
                inner.raw_payload = {'status': 'answered', 'claims': [
                    {'doc_id': c['doc_id'], 'category': 'requirements', 'text': option['quote'],
                     'citations': [{'chunk_id': c['chunk_id'], 'quote': option['ref']}]}
                    for c in chunks for option in c['citation_options']], 'missing_information': []}
                return {'status': 'completed', 'output_text': json.dumps(inner.raw_payload), 'usage': None}
        wrapper, delegate, client = self.llm(cache=True, client=AllOptionsClient())
        chunks = self.env['index'].get_document_chunks('A')[-4:]
        first = wrapper.generate('qa', '범위는?', chunks)
        second = wrapper.generate('qa', '범위는?', chunks)
        self.assertEqual(first['payload'], second['payload'])
        expected = quote_references(chunks)
        self.assertEqual(len(first['payload']['claims']), len(expected))
        for raw, claim in zip(client.raw_payload['claims'], first['payload']['claims']):
            ref = raw['citations'][0]['quote']
            self.assertRegex(ref, r'^q[0-9]{4}$')
            self.assertEqual(claim['citations'][0]['quote'], expected[ref]['quote'])
        checked = self.env['registry'].validate_answer(first['payload'], chunks, allowed_doc_ids=['A'])
        self.assertTrue(checked['valid'], checked['errors'])
        self.assertTrue(second['cache_hit'])
        self.assertEqual(len(client.bodies), 1)
        self.assertEqual(delegate.usage_report()['cache_hits'], 1)

    def test_valid_source_literal_and_unknown_reference_are_rejected_without_guessing(self):
        class BadSelectionClient(GroundedClient):
            def create(inner, **body):
                result = super().create(**body)
                payload = json.loads(result['output_text'])
                chunks = json.loads(body['input'][1]['content'])['evidence']
                payload['claims'][0]['citations'][0]['quote'] = (
                    chunks[0]['text'] if inner.mode == 'literal' else 'q9999')
                result['output_text'] = json.dumps(payload)
                return result
        chunks = self.env['index'].get_document_chunks('A')[-4:]
        for mode in ('literal', 'unknown'):
            with self.subTest(mode=mode):
                wrapper, delegate, client = self.llm(client=BadSelectionClient(mode))
                with self.assertRaises(LLMError):
                    wrapper.generate('qa', '범위는?', chunks)
                self.assertEqual(len(client.bodies), 1)
                self.assertEqual(delegate.usage_report()['mock_calls'], 1)

    def test_identical_source_strings_keep_distinct_reference_owners(self):
        chunks = copy.deepcopy(self.env['index'].get_document_chunks('A')[-1:]
                               + self.env['index'].get_document_chunks('B')[-1:])
        for chunk in chunks:
            chunk['text'] = '같은 원문 구절\n원문 소유자는 구별한다.'
        refs = quote_references(chunks)
        self.assertEqual(len(refs), 2)
        self.assertEqual(refs['q0001']['quote'], refs['q0002']['quote'])
        payload = {'claims': [{'doc_id': 'B', 'citations': [
            {'chunk_id': chunks[1]['chunk_id'], 'quote': 'q0001'}]}]}
        self.assertFalse(accepts_citations(payload, literal_schema(chunks), chunks))

    def test_request_local_references_do_not_reuse_other_document_cache_entries(self):
        wrapper, delegate, client = self.llm(cache=True)
        chunks_a = self.env['index'].get_document_chunks('A')[-1:]
        chunks_b = self.env['index'].get_document_chunks('B')[-1:]
        a = wrapper.generate('qa', '조건은?', chunks_a)
        b = wrapper.generate('qa', '조건은?', chunks_b)
        again = wrapper.generate('qa', '조건은?', chunks_a)
        self.assertNotEqual(a['request_fingerprint'], b['request_fingerprint'])
        self.assertEqual(a['payload'], again['payload'])
        self.assertNotEqual(a['payload'], b['payload'])
        self.assertEqual(len(client.bodies), 2)
        self.assertEqual(delegate.usage_report()['cache_hits'], 1)

    def test_string_literal_rejection_is_reported_and_selectors_keep_single_attempts(self):
        class SchemaError(RuntimeError):
            status_code = 400
            code = 'invalid_json_schema'
            body = {'message': 'Invalid schema: newline is not allowed in string literals '
                    'for structured outputs (strict=true).'}
        class LiteralRestrictedClient(GroundedClient):
            attempts = 0
            def create(inner, **body):
                inner.attempts += 1
                schema = body['text']['format']['schema']
                if any(any(ord(c) < 32 for c in q) for q in citations(schema)['quote']['enum']):
                    raise SchemaError('PRIVATE_PROVIDER_BODY')
                return super().create(**body)
        client = LiteralRestrictedClient()
        wrapper, delegate, _ = self.llm(client=client)
        chunks = self.env['index'].get_document_chunks('A')[-4:]
        old_schema = literal_schema(chunks)
        citations(old_schema)['quote']['enum'] = [c['quote'] for c in quote_references(chunks).values()]
        with self.assertRaises(LLMError) as caught:
            delegate.generate('qa', '사업 범위는?', chunks, schema=old_schema)
        self.assertEqual(caught.exception.diagnostic['http_status'], 400)
        self.assertEqual(caught.exception.diagnostic['api_error_code'], 'invalid_json_schema')
        self.assertIn('string literals for structured outputs', caught.exception.diagnostic['schema_error_message'])
        self.assertEqual(client.attempts, 1)
        result = wrapper.generate('qa', '사업 범위는?', chunks)
        self.assertTrue(result['literal_citation_contract']['selection_valid'])
        self.assertEqual(client.attempts, 2)
        self.assertEqual(delegate.usage_report()['failed_attempts'], 1)

    @staticmethod
    def _schema_hash():
        from rfp_rag.generation.evidence import ANSWER_SCHEMA
        return hashlib.sha256(json.dumps(ANSWER_SCHEMA, sort_keys=True).encode()).hexdigest()

    def test_empty_duplicate_and_excessive_choices_fail_before_dispatch(self):
        for evidence in [[], self.env['index'].get_document_chunks('A')[-1:] * 2]:
            with self.assertRaises(ValueError):
                literal_schema(evidence)
        with self.assertRaises(ValueError):
            literal_options(' \n ')
        huge = copy.deepcopy(self.env['index'].get_document_chunks('A')[-1:])
        huge[0]['text'] = '\n'.join(f'{i:04d} 별도 원문 구간 ' + '가' * 150 for i in range(300))
        with self.assertRaises(ValueError):
            literal_schema(huge)

    def test_preflight_and_generation_use_identical_transformed_requests_and_one_client(self):
        wrapper, delegate, client = self.llm()
        evidence = self.env['index'].get_document_chunks('A')[-4:]
        request = {'task': 'qa', 'question': '대상 업무는?', 'evidence': evidence}
        with patch.object(delegate, 'preflight_requests', wraps=delegate.preflight_requests) as planned:
            wrapper.preflight_requests([request])
            result = wrapper.generate(**request)
        expected = planned.call_args.args[0][0]
        self.assertEqual(expected['task'], QA_TASK)
        self.assertEqual(client.bodies[0]['text']['format']['schema'], expected['schema'])
        body = json.loads(client.bodies[0]['input'][1]['content'])
        self.assertEqual(body['question'], expected['question'])
        self.assertTrue(result['literal_citation_contract']['selection_valid'])
        self.assertEqual(delegate.usage_report()['mock_calls'], 1)

    def test_invalid_model_selection_stays_blocked_without_retry_or_claim_dropping(self):
        wrapper, delegate, client = self.llm(client=GroundedClient('wrong_quote'))
        evidence = self.env['index'].get_document_chunks('A')[-4:]
        with self.assertRaises(LLMError):
            wrapper.generate('qa', '대상 업무는?', evidence)
        self.assertEqual(len(client.bodies), 1)
        self.assertEqual(delegate.usage_report()['mock_calls'], 1)

    def test_existing_free_quote_cache_and_literal_quote_cache_are_separate(self):
        wrapper, delegate, client = self.llm(cache=True)
        evidence = self.env['index'].get_document_chunks('A')[-4:]
        delegate.generate('qa', '대상 업무는?', evidence)
        wrapper.generate('qa', '대상 업무는?', evidence)
        cached = wrapper.generate('qa', '대상 업무는?', evidence)
        self.assertEqual(len(client.bodies), 2)
        self.assertTrue(cached['cache_hit'])
        self.assertEqual(delegate.usage_report()['cache_hits'], 1)

    def test_comparison_checks_two_requests_against_remaining_budget_before_first_call(self):
        wrapper, delegate, client = self.llm(max_calls=1)
        requests = [{'task': COMPARISON_DOCUMENT_TASK, 'question': '구축 범위 비교',
            'evidence': self.env['index'].get_document_chunks(did)[-4:]} for did in ['A', 'B']]
        with self.assertRaises(LLMError):
            wrapper.preflight_requests(requests)
        self.assertEqual(client.bodies, [])

    def test_context_and_schema_stay_in_the_existing_request_byte_budget(self):
        client = GroundedClient()
        chat = make_conversation(self.env, enabled=True, approved_doc_ids=['A'], client=client,
            retrieval={'method': 'rrf', 'top_k': 8, 'fetch_k': 40, 'max_context_bytes': 42000})
        chat.select_documents(['A'])
        result = chat.ask('구체적인 공사의 내용은?')
        self.assertEqual(result['execution_status'], 'completed')
        self.assertTrue(result['validation']['valid'])
        self.assertLessEqual(result['generation']['request_input_bytes'], chat.llm.config.max_request_bytes)
        self.assertEqual(chat.llm.usage_report()['max_calls'], 30)

    def test_scope_comparison_keeps_per_document_validation_and_shared_usage(self):
        client = GroundedClient()
        chat = make_conversation(self.env, enabled=True, approved_doc_ids=['A', 'B'], client=client,
            retrieval={'method': 'rrf', 'top_k': 8, 'per_doc_top_k': 6, 'fetch_k': 40})
        chat.select_documents(['A', 'B'])
        result = chat.ask('두 사업의 구체적인 구축 대상 업무 범위를 비교해줘.')
        self.assertEqual(result['execution_status'], 'completed')
        self.assertTrue(result['validation']['valid'])
        self.assertEqual(len(client.bodies), 2)
        self.assertEqual(chat.llm.usage_report()['mock_calls'], 2)
        self.assertEqual({c['doc_id'] for c in result['claims']}, {'A', 'B'})

    def test_budget_request_keeps_existing_schema_and_task(self):
        from rfp_rag.generation.evidence import ANSWER_SCHEMA
        client = GroundedClient()
        chat = make_conversation(self.env, enabled=True, approved_doc_ids=['A'], client=client)
        chat.select_documents(['A'])
        chat.ask('사업예산은 얼마야?')
        self.assertEqual(client.bodies[0]['text']['format']['schema'], ANSWER_SCHEMA)
        self.assertEqual(json.loads(client.bodies[0]['input'][1]['content'])['task'], 'qa')


ANSWER_BASELINE = LiteralQuoteTests._schema_hash()


if __name__ == '__main__':
    unittest.main()
