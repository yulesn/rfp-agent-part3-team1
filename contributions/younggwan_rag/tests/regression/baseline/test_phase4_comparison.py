"""Independent comparison identities, local omissions and source section roles."""
import copy
import unittest

from rfp_rag.retrieval.chunking import build_chunks
from rfp_rag.generation.comparison import document_question,local_missing_information,DurationSources
from rfp_rag.generation.evidence import EvidenceRegistry
from rfp_rag.generation.rag import AnswerSession
from regression.baseline.test_phase4_rag import fixture, Index, LLM


CATALOG=[{'doc_id':'DOC_ALPHA','title':'해안 관측정보 플랫폼 구축','agency':'시험기관'},
         {'doc_id':'DOC_BETA','title':'도시 자원관리 서비스 개선','agency':'가상재단'}]


class LocalRequestTests(unittest.TestCase):
    def test_prompt_keeps_requested_conditions_and_only_current_project(self):
        q='「시험기관: 해안 관측정보 플랫폼 구축」 / 「가상재단: 도시 자원관리 서비스 개선」. 비교 항목은 예산·부가세와 사업기간(안정화·조정 조건 포함)이야. 문서별 열을 두는 표로 정리해.'
        p=document_question(q,'DOC_ALPHA',CATALOG)
        self.assertIn('해안 관측정보 플랫폼 구축',p)
        for other in ['도시 자원관리 서비스 개선','DOC_BETA','가상재단']:self.assertNotIn(other,p)
        for condition in ['예산','부가세','사업기간','안정화','조정 조건']:self.assertIn(condition,p)

    def test_other_document_missing_is_audited_but_local_missing_is_preserved(self):
        missing=['도시 자원관리 서비스 개선의 사업기간','해안 관측정보 플랫폼 구축의 안정화 조건','제출기한']
        kept,audit=local_missing_information(missing,'DOC_ALPHA',CATALOG)
        self.assertEqual(kept,missing[1:]);self.assertEqual(len(audit['excluded_other_document_items']),1)

    def test_shared_title_and_cross_document_relation_stay_ambiguous(self):
        cat=copy.deepcopy(CATALOG);cat[1]['title']=cat[0]['title']
        q='해안 관측정보 플랫폼 구축의 사업기간'
        self.assertEqual(local_missing_information([q],'DOC_ALPHA',cat)[0],[q])
        q='해안 관측정보 플랫폼 구축과 도시 자원관리 서비스 개선의 조건 차이'
        self.assertEqual(local_missing_information([q],'DOC_ALPHA',CATALOG)[0],[q])

    def test_nested_title_uses_longest_occurrence(self):
        cat=copy.deepcopy(CATALOG);cat[1]['title']='중장기 '+cat[0]['title']
        kept,_=local_missing_information(['중장기 해안 관측정보 플랫폼 구축의 예산'],'DOC_ALPHA',cat)
        self.assertEqual(kept,[])


class DurationRoleTests(unittest.TestCase):
    def setUp(self):
        self.text=('1. 사업 개요\n사업기간: 계약일부터 241일\n'
            '<별첨 8> 소프트웨어 개발사업 적정 사업기간 종합 산정서\n'
            '구축 4개월\n서비스 제공 7개월\n사업기간 11개월\n'
            '<별첨 9> 기술적용계획표\n사업기간: 별도 안내\n')
        self.d={'blocks':[{'block_id':'page','text':self.text}]}
        self.sources=DurationSources(self.d)

    def claim(self,*quotes):
        return {'citations':[{'source_ranges':[{'block_id':'page','char_start':self.text.index(q),
                                                'char_end':self.text.index(q)+len(q)}]} for q in quotes]}

    def test_exact_ranges_on_same_pdf_page_keep_main_and_estimate_distinct(self):
        self.assertEqual(self.sources.claim_role(self.claim('사업기간: 계약일부터 241일'))['role'],'main_or_other_source')
        a=self.sources.claim_role(self.claim('사업기간 11개월'))
        self.assertEqual(a['role'],'duration_estimate');self.assertTrue(a['source_headings'])

    def test_next_attachment_ends_estimate_and_mixed_claim_is_flagged(self):
        self.assertEqual(self.sources.claim_role(self.claim('사업기간: 별도 안내'))['role'],'main_or_other_source')
        self.assertEqual(self.sources.claim_role(self.claim('241일','11개월'))['role'],'duration_mixed')


class CompositionTests(unittest.TestCase):
    def setup_session(self,mode='complete'):
        docs,_=fixture()
        for d in docs:
            d['title']=CATALOG[0 if d['doc_id']=='A' else 1]['title']
            d['blocks'][0]['text']=('사업예산: 137백만원\n사업기간: 계약일부터 241일\n'
                '<별첨 8> 소프트웨어 개발사업 적정 사업기간 종합 산정서\n'
                '구축 4개월\n서비스 제공 7개월\n사업기간 11개월\n')
        chunks,_=build_chunks(docs);registry=EvidenceRegistry(docs,chunks)
        class LocalIndex(Index):
            def get_catalog(inner):return [{'doc_id':d['doc_id'],'title':d['title']} for d in docs]
        class Model(LLM):
            def generate(inner,task,question,evidence):
                inner.requests.append((task,question,evidence));did=evidence[0]['doc_id'];cid=evidence[0]['chunk_id']
                def c(text,quote):return {'doc_id':did,'category':'duration','text':text,
                    'citations':[{'chunk_id':cid,'quote':quote}]}
                claims=[c('사업기간: 계약일부터 241일','사업기간: 계약일부터 241일')]
                if mode=='only_estimate':claims=[]
                claims.append(c('산정서의 사업기간 11개월','사업기간 11개월'))
                if mode=='bad_number':claims.append(c('산정에는 4개월과 7개월이 반영됨','서비스 제공 7개월'))
                missing=[docs[1 if did=='A' else 0]['title']+'의 예산']
                if mode=='local_missing' and did=='B':missing.append('안정화 조건')
                return {'payload':{'status':'partial','claims':claims,'missing_information':missing},
                        'execution_mode':'mock','usage':{},'cache_hit':False}
        llm=Model();return AnswerSession(LocalIndex(docs,chunks),llm,registry,active_doc_ids=['A','B']),llm

    def test_complete_pair_does_not_require_other_document_and_estimates_have_own_row(self):
        s,llm=self.setup_session();r=s.ask('두 문서의 사업기간과 산정자료 비교',feature='comparison')
        self.assertEqual(r['status'],'answered');self.assertEqual(r['missing_information'],[])
        self.assertIn('기간 산정자료(별첨)',r['markdown'])
        row=next(x for x in r['markdown'].splitlines() if x.startswith('| 사업기간 |'))
        self.assertIn('241일',row);self.assertNotIn('11개월',row)
        self.assertTrue(r['validation']['valid'])
        self.assertEqual(sum(len(d['missing_scope_audit']['excluded_other_document_items']) for d in r['document_validation']),2)

    def test_genuine_local_missing_stays_partial(self):
        s,_=self.setup_session('local_missing');r=s.ask('사업기간 비교',feature='comparison')
        self.assertEqual(r['status'],'partial');self.assertEqual(len(r['missing_information']),1)
        self.assertIn('안정화 조건',r['missing_information'][0])

    def test_unquoted_duration_value_is_not_rendered_even_if_present_in_chunk(self):
        s,_=self.setup_session('bad_number');r=s.ask('기간과 산정자료 비교',feature='comparison')
        self.assertEqual(r['status'],'partial')
        self.assertNotIn('산정에는 4개월',r['markdown'])
        self.assertTrue(any(a.get('withheld') for a in r['comparison_audit']))
        self.assertTrue(r['validation']['valid'])

    def test_only_estimate_cannot_complete_a_contract_period_request(self):
        s,_=self.setup_session('only_estimate');r=s.ask('사업기간 비교',feature='comparison')
        self.assertEqual(r['status'],'partial')
        self.assertTrue(any('계약·과업기간' in m for m in r['missing_information']))
        self.assertNotIn('| 사업기간 |',r['markdown'])

    def test_unsolicited_unquoted_auxiliary_detail_does_not_hide_supported_main_period(self):
        s,_=self.setup_session('bad_number');r=s.ask('사업기간 비교',feature='comparison')
        self.assertEqual(r['status'],'answered')
        self.assertTrue(any(a.get('withheld') for a in r['comparison_audit']))
        self.assertNotIn('산정에는 4개월',r['markdown'])


if __name__=='__main__':unittest.main()
