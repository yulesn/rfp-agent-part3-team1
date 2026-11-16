"""Adversarial intent/retrieval cases with independently invented source text."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rfp_rag.retrieval.chunking import build_chunks
from rfp_rag.retrieval.retrieval import LocalSearchIndex
from rfp_rag.generation.evidence import EvidenceRegistry
from rfp_rag.conversation.focus import focused_question, response_focus_audit
from rfp_rag.conversation.query import plan_content_queries, prioritize_content_candidates
from rfp_rag.conversation.runtime import make_conversation
from rfp_rag.conversation.session import ConversationSession


def independent_sources():
    definitions = [
        ('A', '분산 장비 반출입 통합관리시스템 구축', 'pdf',
         '장비 반출입 예약, 출입 승인, 반환 이력 관리 기능을 구축한다.',
         'FUN-장비-101 반출 신청\nFUN-장비-102 승인 처리\nFUN-장비-103 반환 이력 조회',
         '사업예산 137,000,000원, 부가가치세 포함. 과업기간 계약 후 11개월(시범운영 3개월 포함).'),
        ('B', '해양 소음지도 분석도구 고도화', 'hwp',
         '음향 데이터 편집, 지도 타일 생성, 알림 기능을 구축한다.',
         'SFR-201 음향 데이터 편집\nSFR-202 지도 타일 생성\nSFR-203 알림 처리',
         '사업예산 54백만원, VAT 별도. 과업기간 착수 후 6개월.'),
        ('C', '산책로 개선', 'hwpx',
         '보행로 재포장 420m, 난간 교체 82m, 안내판 설치 8개를 수행한다.',
         '시공 대상 업무\n보행로 포장과 난간 및 안내판 교체',
         '사업예산 92백만원, 부가세 포함. 과업기간 5개월.')]
    docs = []
    for did, title, fmt, scope, inventory, numbers in definitions:
        texts = [f'본 사업의 구체적인 내용 및 요구사항. 제안사는 추진목표와 추진전략을 파악하여 '
                 f'최적의 개발 방향과 개발방법론을 제시하여야 한다. 제안서 작성지침 {i}.' for i in range(16)]
        texts += [f'가격 제안서\n사업 내용\n사업 예산:     원\n준공 및 완료 조건:     {i}' for i in range(10)]
        texts += ['3. 사업 범위\n' + scope, '요구사항 목록표\n대상 업무\n' + inventory,
            '준공 검수 조건\n인수시험 결과보고서를 제출하고 발주기관 승인을 받아야 한다.\n'
            '완료 검수 후 17일 이내 최종보고서를 제출한다.\n검수 후 9개월간 지원한다.', numbers]
        blocks = []
        for i, text in enumerate(texts):
            location = {'page': i + 1, 'locator': f'PDF p.{i + 1}'} if fmt == 'pdf' else {
                'section': f'BodyText/Section{i}', 'locator': f'BodyText/Section{i} / paragraph 1'}
            blocks.append({'block_id': f'b{i}', 'text': text, 'kind': 'paragraph', **location})
        docs.append({'doc_id': did, 'canonical_doc_id': did, 'title': title,
            'file_name': f'가상검증원_2036_{title}.{fmt}', 'agency': '가상검증원', 'document_year': 2036,
            'detected_format': fmt, 'sha256': hashlib.sha256('\n'.join(texts).encode()).hexdigest(),
            'blocks': blocks, 'tables': [], 'supplemental_blocks': []})
    return docs


class GroundedClient:
    def __init__(self, mode='grounded'):
        self.responses = self
        self.bodies = []
        self.mode = mode

    def create(self, **body):
        self.bodies.append(body)
        evidence = json.loads(body['input'][1]['content'])['evidence']
        source = evidence[0]
        source = next((c for c in evidence if '3. 사업 범위' in c['text']), source)
        did = 'ANOTHER_DOCUMENT' if self.mode == 'wrong_doc' else source['doc_id']
        quote = '원문에 존재하지 않는 구절' if self.mode == 'wrong_quote' else source['text']
        schema = body.get('text', {}).get('format', {}).get('schema', {})
        choices = (schema.get('properties', {}).get('claims', {}).get('items', {}).get('properties', {})
                   .get('citations', {}).get('items', {}).get('properties', {}).get('quote', {}).get('enum'))
        if choices and self.mode != 'wrong_quote':
            quote = next(option['ref'] for option in source['citation_options'] if option['ref'] in choices)
        payload = {'status': 'answered', 'claims': [{'doc_id': did, 'category': 'requirements',
            'text': source['text'], 'citations': [{'chunk_id': source['chunk_id'], 'quote': quote}]}],
            'missing_information': []}
        if self.mode == 'absence':
            payload = {'status': 'insufficient_evidence', 'claims': [], 'missing_information': ['추가 근거 확인']}
        return {'status': 'completed', 'output_text': json.dumps(payload, ensure_ascii=False), 'output': [],
            'usage': {'input_tokens': 113, 'output_tokens': 29, 'total_tokens': 142}}


def synthetic_environment(root):
    docs = independent_sources()
    chunks, _ = build_chunks(docs)
    index = LocalSearchIndex(root / 'index')
    index.build(chunks, {'min_df': 1, 'max_features': 20000})
    config6 = {'conversation': {}, 'retrieval': {'method': 'rrf', 'top_k': 8, 'per_doc_top_k': 6,
        'fetch_k': 40, 'mmr_lambda': .75, 'max_context_bytes': 42000, 'max_question_chars': 4000}}
    config8 = {'generation': {'max_calls': 8, 'max_total_input_bytes': 600000},
        'pricing': None, 'summary_run': 'missing'}
    (root / 'configs').mkdir()
    (root / 'configs/phase6.json').write_text(json.dumps(config6))
    (root / 'configs/phase8.json').write_text(json.dumps(config8))
    return {'root': root, 'documents': docs, 'chunks': chunks, 'index': index,
        'registry': EvidenceRegistry(docs, chunks), 'phase5_config': {'generation': {'cache_enabled': False}}}


class FocusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.env = synthetic_environment(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.env['index'].close()
        cls.tmp.cleanup()

    def chat(self, client=None):
        settings = json.loads((self.env['root'] / 'configs/phase6.json').read_text())
        return make_conversation(self.env, enabled=client is not None, approved_doc_ids=['A', 'B', 'C'],
            client=client, config=settings['conversation'], retrieval=settings['retrieval'])

    def test_scope_reformulation_drops_previous_completion_question(self):
        chat = self.chat()
        chat.select_documents(['A'])
        chat.ask('이 사업의 공사 완료 조건은?')
        result = chat.ask('그러니까, 구체적인 공사의 내용이 무엇이냐는거지.')
        plan = result['conversation']['retrieval_plan']
        self.assertEqual(plan['topics'], ['work_scope'])
        self.assertNotIn('완료 조건', plan['content_question'])
        self.assertNotIn('직전 사용자 질문', result['conversation']['standalone_query'])
        self.assertIn('반출 신청', '\n'.join(c['text'] for c in result['source_chunks']))

    def test_work_scope_paraphrases_recover_actual_inventory_for_each_format(self):
        for did, target in [('A', '반환 이력'), ('B', '지도 타일'), ('C', '난간')]:
            for question in ['구체적인 공사 내용은?', '어떤 업무와 기능을 구축하는 사업이야?',
                             '이 사업에서는 뭘 만드는 거야?', '구축 대상 업무 범위는?']:
                with self.subTest(did=did, question=question):
                    chat = self.chat()
                    chat.select_documents([did])
                    result = chat.ask(question)
                    self.assertIn(target, '\n'.join(c['text'] for c in result['source_chunks']))
                    self.assertTrue(all(c['doc_id'] == did for c in result['source_chunks']))

    def test_mixed_budget_scope_question_keeps_both_source_types(self):
        chat = self.chat()
        chat.select_documents(['A'])
        result = chat.ask('사업예산과 부가세 조건, 구체적인 구축 대상 업무는?')
        text = '\n'.join(c['text'] for c in result['source_chunks'])
        self.assertIn('137,000,000원', text)
        self.assertIn('반출 신청', text)

    def test_scope_to_duration_does_not_reuse_old_work_items(self):
        chat = self.chat()
        chat.select_documents(['A'])
        chat.ask('구체적인 공사 내용은?')
        result = chat.ask('그건 몇 개월짜리야?')
        self.assertEqual(result['conversation']['retrieval_plan']['topics'], ['duration'])
        self.assertIn('11개월', '\n'.join(c['text'] for c in result['source_chunks']))

    def test_selection_change_and_comparison_do_not_mix_sources(self):
        chat = self.chat()
        chat.select_documents(['A'])
        chat.ask('구체적인 사업 내용은?')
        chat.select_documents(['B'])
        result = chat.ask('구체적인 사업 내용은?')
        self.assertTrue(all(c['doc_id'] == 'B' for c in result['source_chunks']))
        self.assertNotIn('반출 신청', '\n'.join(c['text'] for c in result['source_chunks']))
        chat.select_documents(['A', 'B'])
        result = chat.ask('두 사업의 구축 대상 업무 범위를 비교해줘.')
        text = '\n'.join(c['text'] for c in result['source_chunks'])
        self.assertIn('반출 신청', text)
        self.assertIn('지도 타일', text)

    def test_source_form_is_available_when_explicitly_requested(self):
        chat = self.chat()
        chat.select_documents(['A'])
        result = chat.ask('사업 내용 기입용 가격제안서 양식 원문을 보여줘.')
        self.assertIn('가격 제안서', '\n'.join(c['text'] for c in result['source_chunks']))

    def test_very_broad_request_stays_inside_query_count_limit(self):
        question = '사업예산 기간 참가자격 제출 평가 구축 대상 업무 범위 준공 조건과 GPU 필수 요구사항을 알려줘.'
        plan = plan_content_queries(question, [])
        self.assertLessEqual(len(plan['queries']), 7)
        self.assertIn(question, plan['queries'])
        self.assertFalse(plan['facts_added_by_rewrite'])

    def test_focus_instructions_add_no_business_values_or_contract_facts(self):
        text = focused_question('사업 범위를 알려줘.', ['work_scope', 'acceptance'])
        self.assertNotIn('137', text)
        self.assertNotIn('11개월', text)
        self.assertNotIn('반출', text)
        self.assertIn('일반 원칙', text)
        self.assertIn('선행 조건', text)

    def test_submission_validity_clause_is_not_demoted_by_unrequested_scope_rules(self):
        hits = [{'chunk_id': 'VALIDITY', 'text': '제안서 평가\n입찰보증보험 유효기간은 제출일부터 72일 이상이어야 한다.'},
                {'chunk_id': 'BACKGROUND', 'text': '일반적인 사업 일정은 관련 자료를 확인한다.'}]
        result = prioritize_content_candidates(hits, topics=['duration'],
            content_question='입찰보증보험의 유효기간은 어떻게 되나요?')
        self.assertEqual(result[0]['chunk_id'], 'VALIDITY')

    def test_generation_receives_current_intent_and_actual_scope_source(self):
        client = GroundedClient()
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'synthetic-not-a-real-key'}):
            chat = self.chat(client)
            chat.select_documents(['A'])
            result = chat.ask('구체적인 공사의 내용은?')
        request = json.loads(client.bodies[0]['input'][1]['content'])
        self.assertIn('반출 신청', '\n'.join(c['text'] for c in request['evidence']))
        self.assertIn('일반 원칙', request['question'])
        self.assertTrue(result['validation']['valid'])
        self.assertFalse(result['answer_focus_audit']['semantic_correctness_verified'])

    def test_foreign_document_and_fabricated_quotes_are_withheld(self):
        for mode in ['wrong_doc', 'wrong_quote']:
            with self.subTest(mode=mode), patch.dict(os.environ, {'OPENAI_API_KEY': 'synthetic-not-a-real-key'}):
                chat = self.chat(GroundedClient(mode))
                chat.select_documents(['A'])
                result = chat.ask('구체적인 사업 내용은?')
                self.assertEqual(result['execution_status'], 'error')
                self.assertEqual(result['error_diagnostic']['code'], 'invalid_answer_structure')
                self.assertEqual(result['claims'], [])
                self.assertFalse(result['answer_generated'])

    def test_generic_process_warning_never_asserts_semantic_pass(self):
        result = {'claims': [{'text': '제안사는 추진목표를 파악하고 개발 방향을 제시해야 한다.'}]}
        audit = response_focus_audit(result, ['work_scope'])
        self.assertTrue(audit['needs_scope_review'])
        self.assertFalse(audit['semantic_correctness_verified'])
        self.assertEqual(result['claims'][0]['text'], '제안사는 추진목표를 파악하고 개발 방향을 제시해야 한다.')


if __name__ == '__main__':
    unittest.main()
