"""Grounded answer integration with original source chunks and no network."""
import copy
from types import SimpleNamespace
import unittest

from rfp_rag.retrieval.chunking import build_chunks
from rfp_rag.generation.evidence import EvidenceRegistry
from rfp_rag.generation.rag import AnswerSession


def fixture():
    docs = []
    for did, text in [('A', '사업예산 100백만원, 부가세 포함.'), ('B', '사업예산 200백만원, 부가세 별도.')]:
        docs.append({'doc_id': did, 'canonical_doc_id': did, 'sha256': did.lower() * 64,
                     'file_name': did + '_시스템구축사업.hwp', 'agency': '테스트기관' + did,
                     'document_year': 2026, 'detected_format': 'hwp', 'blocks': [
                         {'block_id': 'p1', 'text': text, 'locator': 'Section0 / paragraph 1',
                          'section': 'Section0', 'kind': 'paragraph'}],
                     'supplemental_blocks': [], 'reference_blocks': [], 'tables': []})
    chunks, _ = build_chunks(docs)
    return docs, chunks


class Index:
    def __init__(self, docs, chunks):
        self.docs, self.chunks, self.calls = docs, chunks, []
        self.no_hits_for = set()

    def resolve_doc_ids(self, ids):
        values = list(ids)
        if any(v not in {'A', 'B', 'alias_A'} for v in values):
            raise ValueError('unknown ID')
        return list(dict.fromkeys('A' if v == 'alias_A' else v for v in values))

    def get_catalog(self):
        return [{**{k: d[k] for k in ['doc_id','sha256','file_name','agency','document_year']},
                 'title': '독립적인 ' + d['doc_id'] + ' 시스템 개발 사업'} for d in self.docs]

    def search(self, query, top_k=5, allowed_doc_ids=None, **kwargs):
        self.calls.append({'query': query, 'scope': allowed_doc_ids, **kwargs})
        ids = set(self.resolve_doc_ids(allowed_doc_ids)) if allowed_doc_ids is not None else {'A','B'}
        return [dict(c, score=1.0, score_is_confidence=False) for c in self.chunks
                if c['doc_id'] in ids and c['doc_id'] not in self.no_hits_for][:top_k]


class LLM:
    def __init__(self, enabled=True, mode='valid'):
        self.config = SimpleNamespace(enabled=enabled)
        self.mode, self.requests = mode, []

    def generate(self, task, question, evidence):
        self.requests.append((task, question, copy.deepcopy(evidence)))
        if self.mode == 'error':
            raise RuntimeError('PRIVATE_UPSTREAM_BODY_AND_KEY')
        chosen = [c for c in evidence if c['doc_id']=='A'] if self.mode == 'only_first' else evidence
        claims = [{'doc_id': c['doc_id'], 'category': 'budget', 'text': c['text'],
                   'citations': [{'chunk_id': c['chunk_id'],
                                  'quote': 'FABRICATED_QUOTE' if self.mode == 'bad_quote' else c['text']}]}
                  for c in chosen]
        return {'payload': {'status': 'answered' if claims else 'insufficient_evidence',
                'claims': claims, 'missing_information': [] if claims else ['현재 문서의 예산']},
                'usage': {'input_tokens': 10, 'output_tokens': 10},
                'live_api': False, 'execution_mode': 'mock', 'cache_hit': False}


class AnswerTests(unittest.TestCase):
    def setUp(self):
        self.docs, self.chunks = fixture()
        self.index = Index(self.docs, self.chunks)
        self.registry = EvidenceRegistry(self.docs, self.chunks)

    def session(self, llm=None, ids=('A',), **config):
        return AnswerSession(self.index, llm or LLM(), self.registry,
                             config=config or None, active_doc_ids=list(ids))

    def test_disabled_mode_returns_source_candidates_without_fake_answer(self):
        llm = LLM(enabled=False)
        result = self.session(llm).ask('예산을 알려줘')
        self.assertEqual(result['execution_status'], 'disabled')
        self.assertFalse(result['answer_generated'])
        self.assertTrue(result['source_chunks'])
        self.assertEqual(llm.requests, [])

    def test_valid_answer_has_resolved_original_source(self):
        result = self.session().ask('예산을 알려줘')
        self.assertEqual(result['status'], 'answered')
        self.assertTrue(result['validation']['valid'])
        self.assertEqual(result['claims'][0]['doc_id'], 'A')
        self.assertTrue(result['claims'][0]['citations'][0]['source_ranges'])
        self.assertFalse(result['semantic_entailment_verified'])
        self.assertNotIn('payload', result['generation'])

    def test_fabricated_quote_is_not_rendered_or_returned_as_claim(self):
        result = self.session(LLM(mode='bad_quote')).ask('예산은?')
        self.assertEqual(result['execution_status'], 'validation_failed')
        self.assertEqual(result['claims'], [])
        self.assertNotIn('FABRICATED_QUOTE', result['markdown'])
        self.assertNotIn('payload', result['generation'])

    def test_comparison_reserves_each_document_before_merged_context(self):
        result = self.session(ids=('A','B'), top_k=1).ask('두 사업 예산을 비교해줘', feature='comparison')
        self.assertEqual({c['doc_id'] for c in result['source_chunks']}, {'A','B'})
        self.assertTrue(result['retrieval']['document_quotas_applied_before_packing'])
        self.assertTrue(any(c['scope'] == ['B'] for c in self.index.calls))
        self.assertIn('| 항목 |', result['markdown'])
        self.assertIn('[1]', result['markdown'])
        self.assertIn('[2]', result['markdown'])

    def test_one_sided_comparison_is_partial(self):
        result = self.session(LLM(mode='only_first'), ids=('A','B')).ask('예산 비교', feature='comparison')
        self.assertEqual(result['status'], 'partial')
        self.assertTrue(result['missing_information'])

    def test_missing_document_candidates_cannot_look_like_full_comparison(self):
        self.index.no_hits_for.add('B')
        result = self.session(ids=('A','B')).ask('예산 비교', feature='comparison')
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['retrieval']['missing_selected_doc_ids'], ['B'])

    def test_comparison_without_two_selected_documents_does_not_generate(self):
        llm = LLM()
        result = self.session(llm).ask('예산 비교', feature='comparison')
        self.assertEqual(result['status'], 'needs_clarification')
        self.assertEqual(llm.requests, [])

    def test_followup_only_uses_user_questions(self):
        session = self.session()
        session.ask('예산은?')
        session.ask('그건 부가세 포함이야?')
        self.assertEqual(session.query_history, ['예산은?', '그건 부가세 포함이야?'])
        self.assertIn('직전 사용자 질문: 예산은?', session.llm.requests[-1][1])
        self.assertNotIn('100백만원', session.llm.requests[-1][1])
        session.select_documents(['B'])
        session.ask('그 문서의 예산은?')
        self.assertNotIn('직전 사용자 질문:', session.llm.requests[-1][1])

    def test_api_errors_do_not_expose_sensitive_upstream_messages(self):
        result = self.session(LLM(mode='error')).ask('예산은?')
        self.assertEqual(result['execution_status'], 'error')
        self.assertNotIn('PRIVATE_UPSTREAM', str(result))

    def test_unknown_selection_preserves_scope(self):
        session = self.session()
        with self.assertRaises(ValueError):
            session.select_documents(['unknown'])
        self.assertEqual(session.active_doc_ids, ['A'])

    def test_context_size_limit_never_cuts_original_chunk(self):
        session = self.session(max_context_bytes=1)
        result = session.ask('예산은?')
        self.assertEqual(result['status'], 'insufficient_evidence')
        self.assertEqual(session.llm.requests, [])

    def test_invalid_config_and_oversized_questions_are_rejected(self):
        with self.assertRaises(ValueError):
            self.session(top_k=0)
        session = self.session(max_question_chars=2)
        with self.assertRaises(ValueError):
            session.ask('너무 긴 질문')


if __name__ == '__main__':
    unittest.main()
