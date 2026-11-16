"""Exercise actual lexical ranking against invented forms and source clauses.

Unlike a scope-only fake index, this fixture can reproduce a long project name
outranking the paragraph containing a duration. No corpus answers are embedded.
"""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rfp_rag.retrieval.chunking import build_chunks
from rfp_rag.retrieval.retrieval import LocalSearchIndex
from rfp_rag.generation.evidence import EvidenceRegistry
from rfp_rag.conversation.query import plan_content_queries
from rfp_rag.conversation.runtime import GroundedBackend
from rfp_rag.conversation.session import ConversationSession
from regression.conversation.test_session import MockLLM


def _documents():
    docs = []
    titles = ['해안 관측 데이터 통합서비스 운영 지원 플랫폼 구축 및 고도화',
              '도시 자원 순환 통합정보 플랫폼 고도화',
              '생활 환경 관측자료 통합서비스 구축']
    formats = ['hwp', 'hwpx', 'pdf']
    periods = ['과업기간 : 계약일로부터 13개월(안정화 2개월 포함)',
               '사업기간 : 착수일부터 7개월(시범운영 1개월 포함)',
               '수행기간 : 계약 체결 후 16개월']
    for i, (title, fmt, period) in enumerate(zip(titles, formats, periods)):
        texts = [f'{title}\n제안요청서\n사업제안서 양식 {j}' for j in range(24)]
        texts += [f'수행실적 증명서 {j}\n용역기간\n발주처\n계약금액\n직원 경력 기간'
                  for j in range(12)]
        texts += ['2. 사업기간', '소프트웨어 개발사업 적정 사업기간 산정 기준을 참조한다.',
                  '계약수행기간\n년 월 일 ~ 년 월 일\n수행사업명'] * 4
        texts += [f'가상기관 가격제안서 양식 {j}\n예산이 얼마인지 세금 포함 여부를 기재\n'
                  '사업예산 :      원\n입찰금액 :      원\n부가세 포함 여부 : 예/아니오'
                  for j in range(18)]
        texts += ['사업예산 : 243,000,000원(부가가치세 포함)', period]
        blocks = []
        for j, text in enumerate(texts):
            location = ({'page': j + 1, 'locator': f'page {j + 1}'} if fmt == 'pdf'
                        else {'section': f'BodyText/Section{j}',
                              'locator': f'BodyText/Section{j} / paragraph 1'})
            blocks.append({'block_id': f'b{j}', 'text': text, 'kind': 'paragraph', **location})
        did = f'TEST_{i}'
        docs.append({'doc_id': did, 'canonical_doc_id': did,
                     'file_name': f'가상기관_2032_{title}_제안요청서.{fmt}',
                     'title': title, 'agency': '가상기관', 'document_year': 2032,
                     'detected_format': fmt,
                     'sha256': hashlib.sha256('\n'.join(texts).encode()).hexdigest(),
                     'blocks': blocks, 'tables': [], 'supplemental_blocks': []})
    return docs, periods


class QueryPlanningTests(unittest.TestCase):
    def test_deictic_period_uses_terms_without_inserting_a_duration(self):
        p = plan_content_queries('그건 몇 개월짜리야?', ['가상 자료 플랫폼 고도화 사업'])
        self.assertIn('과업기간', ' '.join(p['queries']))
        self.assertEqual(p['topics'], ['duration'])
        self.assertFalse(any(c.isdigit() for c in ' '.join(p['queries'])))

    def test_named_project_is_removed_from_content_but_question_is_kept(self):
        p = plan_content_queries('2032년 해안 관측 데이터 통합서비스 구축의 과업기간은?',
                                 ['2032년 해안 관측 데이터 통합서비스 구축'])
        text = ' '.join(p['queries'])
        self.assertNotIn('해안 관측', text)
        self.assertIn('과업기간', text)
        self.assertNotIn('2032', text)

    def test_multiple_topics_receive_independent_searches(self):
        p = plan_content_queries('사업비와 사업기간을 비교해 줘.', [])
        self.assertEqual(set(p['topics']), {'budget', 'duration'})
        self.assertTrue(any('과업기간' in q and '사업예산' not in q for q in p['queries']))
        self.assertTrue(any('사업예산' in q and '과업기간' not in q for q in p['queries']))

    def test_specific_nonstandard_question_is_not_replaced_with_generic_terms(self):
        p = plan_content_queries('L40S GPU 메모리는 몇 GB인가요?', [])
        self.assertIn('L40S GPU 메모리는 몇 GB인가요?', ' '.join(p['queries']))

    def test_selected_agency_and_project_are_scope_not_budget_search_terms(self):
        p=plan_content_queries('가상기관의 해안 자료 플랫폼 구축 예산은 얼마야? 세금 포함 여부도.',
            ['해안 자료 플랫폼 구축'],agencies=['가상기관'])
        text=' '.join(p['queries'])
        self.assertNotIn('가상기관',text)
        self.assertNotIn('해안 자료 플랫폼',text)
        self.assertIn('사업예산',text)
        self.assertFalse(any(c.isdigit() for c in text))


class LexicalConversationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.docs, cls.periods = _documents()
        cls.chunks, _ = build_chunks(cls.docs)
        cls.registry = EvidenceRegistry(cls.docs, cls.chunks)
        cls.index = LocalSearchIndex(Path(cls.temporary.name) / 'index')
        cls.index.build(cls.chunks, config={'min_df': 1, 'max_features': 20000})
        cls.env = {'index': cls.index, 'registry': cls.registry, 'documents': cls.docs}

    @classmethod
    def tearDownClass(cls):
        cls.index.close()
        cls.temporary.cleanup()

    def setUp(self):
        block = patch('socket.socket.connect', side_effect=AssertionError('network prohibited'))
        block.start()
        self.addCleanup(block.stop)

    def session(self, *, method='rrf', enabled=False):
        llm = MockLLM(enabled=enabled)
        s = ConversationSession(self.docs, backend=GroundedBackend(self.env, llm,
                                retrieval={'method': method, 'top_k': 3}))
        return s, llm

    def test_period_clause_survives_repeated_cover_and_form_titles(self):
        for method in ['bm25', 'char_tfidf', 'rrf', 'mmr']:
            for i in range(3):
                for question in ['그건 몇 개월짜리야?', '사업기간은?']:
                    with self.subTest(method=method, format=self.docs[i]['detected_format'], question=question):
                        s, _ = self.session(method=method)
                        r = s.ask(question, doc_ids=[f'TEST_{i}'])
                        self.assertIn(self.periods[i], '\n'.join(c['text'] for c in r['source_chunks']))
                        self.assertTrue(all(c['doc_id'] == f'TEST_{i}' for c in r['source_chunks']))

    def test_actual_generation_input_contains_new_period_evidence(self):
        s, llm = self.session(enabled=True)
        s.ask('사업예산과 부가세 포함 여부를 알려줘.', doc_ids=['TEST_0'])
        r = s.ask('그건 몇 개월짜리야?')
        _, question, evidence = llm.requests[-1]
        self.assertTrue(r['validation']['valid'])
        self.assertIn(self.periods[0], '\n'.join(c['text'] for c in evidence))
        self.assertIn(self.docs[0]['title'], question)
        self.assertIn('그건 몇 개월짜리야?', question)
        queries = ' '.join(r['retrieval']['content_query_plan']['queries'])
        self.assertNotIn('부가세', queries)
        self.assertNotIn(self.docs[0]['title'], queries)

    def test_stabilization_followup_preserves_period_source_and_conditions(self):
        s, _ = self.session()
        s.ask('사업기간은?', doc_ids=['TEST_0'])
        r = s.ask('그 안에 안정화도 포함돼?')
        self.assertIn(self.periods[0], '\n'.join(c['text'] for c in r['source_chunks']))
        self.assertIn('안정화', ' '.join(r['retrieval']['content_query_plan']['queries']))

    def test_named_title_with_support_word_does_not_pollute_duration_topic(self):
        s, _ = self.session()
        r = s.ask(self.docs[0]['title'] + '는 몇 개월짜리야?')
        self.assertEqual(r['retrieval']['content_query_plan']['topics'], ['duration'])
        self.assertIn(self.periods[0], '\n'.join(c['text'] for c in r['source_chunks']))

    def test_switch_retrieves_new_document_period_and_never_reuses_old_clause(self):
        s, _ = self.session()
        s.ask('사업기간은?', doc_ids=['TEST_0'])
        s.ask('사업비는?', doc_ids=['TEST_1'])
        r = s.ask('그건 몇 개월짜리야?')
        text = '\n'.join(c['text'] for c in r['source_chunks'])
        self.assertIn(self.periods[1], text)
        self.assertNotIn(self.periods[0], text)
        self.assertEqual(r['effective_doc_ids'], ['TEST_1'])

    def test_each_compared_document_keeps_both_budget_and_period_input(self):
        s, _ = self.session()
        r = s.ask('두 문서의 사업예산과 사업기간을 비교해 줘.', doc_ids=['TEST_0', 'TEST_1'])
        for i in range(2):
            text = '\n'.join(c['text'] for c in r['source_chunks'] if c['doc_id'] == f'TEST_{i}')
            self.assertIn(self.periods[i], text)
            self.assertIn('243,000,000원', text)

    def test_duration_in_days_does_not_inherit_previous_budget(self):
        s, _ = self.session()
        s.ask('예산은?', doc_ids=['TEST_0'])
        r = s.ask('그건 몇 일이야?')
        self.assertEqual(r['conversation']['facet'], 'duration')
        self.assertNotIn('사업예산', ' '.join(r['retrieval']['content_query_plan']['queries']))

    def test_populated_price_example_does_not_outrank_the_project_budget(self):
        from regression.summary_content.test_integration import source_fixture
        doc,chunks,registry=source_fixture(['Ⅰ. 사업개요','사업명 : 해양 자료 통합플랫폼 운영',
            '사업예산 : 319,000,000원(부가세 제외)','Ⅱ. 입찰 서식',
            '<서식 제1호> 가격제안서','사업예산 : 811,000,000원(부가세 포함)'])
        with tempfile.TemporaryDirectory() as tmp:
            with LocalSearchIndex(Path(tmp)/'index') as index:
                index.build(chunks,{'min_df':1})
                env={'index':index,'registry':registry,'documents':[doc]}
                s=ConversationSession([doc],backend=GroundedBackend(env,MockLLM(enabled=False),retrieval={'top_k':1}))
                r=s.ask('사업예산과 부가세 포함 여부는?',doc_ids=[doc['doc_id']])
                self.assertIn('319,000,000원',r['source_chunks'][0]['text'])
                self.assertNotIn('811,000,000원',r['source_chunks'][0]['text'])

    def test_explicit_price_form_question_can_still_retrieve_the_form(self):
        from regression.summary_content.test_integration import source_fixture
        doc,chunks,registry=source_fixture(['Ⅰ. 사업개요','사업예산 : 319,000,000원(부가세 제외)',
            'Ⅱ. 입찰 서식','<서식 제1호> 가격제안서','사업예산 : 811,000,000원(부가세 포함)'])
        with tempfile.TemporaryDirectory() as tmp:
            with LocalSearchIndex(Path(tmp)/'index') as index:
                index.build(chunks,{'min_df':1})
                env={'index':index,'registry':registry,'documents':[doc]}
                s=ConversationSession([doc],backend=GroundedBackend(env,MockLLM(enabled=False),retrieval={'top_k':5}))
                r=s.ask('가격제안서 서식의 사업예산란은?',doc_ids=[doc['doc_id']])
                self.assertTrue(any('811,000,000원' in c['text'] for c in r['source_chunks']))

    def test_pdf_header_inherits_old_form_but_budget_belongs_to_new_section(self):
        from regression.summary_content.test_integration import source_fixture
        doc,chunks,registry=source_fixture([
            '<별지 제3호 서식> 가격제안서',
            '- 3 -\nⅠ 사업개요\n가. 사업명 : 해양 자료 통합플랫폼 운영\n'
            '나. 사업예산 : 319,000천원(부가세 제외)',
            'Ⅱ. 입찰 서식\n<서식 제1호> 가격제안서\n사업예산 : 811,000,000원(부가세 포함)'])
        doc['detected_format']='pdf'
        with tempfile.TemporaryDirectory() as tmp:
            with LocalSearchIndex(Path(tmp)/'index') as index:
                index.build(chunks,{'min_df':1})
                env={'index':index,'registry':registry,'documents':[doc]}
                s=ConversationSession([doc],backend=GroundedBackend(env,MockLLM(enabled=False),retrieval={'top_k':1}))
                r=s.ask('사업예산과 부가세 포함 여부는?',doc_ids=[doc['doc_id']])
                self.assertIn('319,000천원',r['source_chunks'][0]['text'])
                self.assertNotIn('811,000,000원',r['source_chunks'][0]['text'])

    def test_budget_body_beats_repeated_price_templates_for_all_lexical_methods(self):
        for method in ['bm25','char_tfidf','rrf','mmr']:
            for i in range(3):
                with self.subTest(method=method,format=self.docs[i]['detected_format']):
                    s,_=self.session(method=method)
                    r=s.ask(f"가상기관의 {self.docs[i]['title']} 예산이 얼마야? 세금 포함 여부도 알려줘.",
                            doc_ids=[f'TEST_{i}'])
                    self.assertIn('사업예산 : 243,000,000원(부가가치세 포함)',r['source_chunks'][0]['text'])
                    self.assertTrue(all(c['doc_id']==f'TEST_{i}' for c in r['source_chunks']))

    def test_model_input_includes_populated_budget_without_rewriting_the_amount(self):
        s,llm=self.session(enabled=True)
        r=s.ask(f"가상기관의 {self.docs[0]['title']} 예산이 얼마야? 세금 포함 여부도 알려줘.",doc_ids=['TEST_0'])
        _,question,evidence=llm.requests[-1]
        self.assertTrue(r['validation']['valid'])
        self.assertIn('243,000,000원(부가가치세 포함)','\n'.join(c['text'] for c in evidence))
        self.assertNotIn('243,000,000',question)
        self.assertNotIn('243,000,000',' '.join(r['retrieval']['content_query_plan']['queries']))


if __name__ == '__main__':
    unittest.main()
