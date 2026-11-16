"""Requested facts, comparison boundaries and budgets use invented evidence."""
import copy
import unittest

from regression.baseline.test_phase4_rag import fixture, Index, LLM
from rfp_rag.generation.evidence import EvidenceRegistry
from rfp_rag.generation.rag import AnswerSession
from rfp_rag.generation.request_policy import award_evidence_gate


class RequestedFactsTests(unittest.TestCase):
    def gate(self,question,text='사업예산 : 137백만원. 낙찰자는 보증서를 제출한다.'):
        return award_evidence_gate(question,[{'text':text}])

    def test_rfp_budget_is_not_an_actual_contract_amount(self):
        r=self.gate('최종 낙찰업체와 실제 계약금액을 알려줘.')
        self.assertEqual(r['requested_fields'],['최종 낙찰업체','실제 계약금액'])
        self.assertEqual(r['additional_materials'],['낙찰 결과 공고','체결 계약서'])

    def test_tax_qualification_does_not_turn_award_question_into_budget_question(self):
        self.assertIsNotNone(self.gate('실제 계약금액과 부가세 포함 여부는?'))

    def test_generic_rfp_questions_and_mixed_questions_are_not_withheld(self):
        for q in ['낙찰자 선정방식은?','최종 낙찰자 선정기준은?',
                  '사업예산과 실제 계약금액을 각각 알려줘.','사업기간은?',
                  '계약금액의 보증 비율은?']:
            with self.subTest(question=q):self.assertIsNone(self.gate(q))

    def test_concrete_award_and_amount_candidates_reach_strict_generation(self):
        for t in ['최종 낙찰업체: 가상상사','최종 낙찰업체는 가상상사',
                  '실제 계약금액: 321,000,000원','체결 계약금액은 321백만원']:
            self.assertIsNone(self.gate('최종 낙찰업체와 실제 계약금액은?',t))

    def test_blank_negative_and_template_result_fields_are_not_positive_evidence(self):
        for t in ['최종 낙찰업체: 미정','낙찰업체: 공란','낙찰업체: 없음',
                  '낙찰업체: (인)','계약금액:     원','사업예산: 321백만원']:
            self.assertIsNotNone(self.gate('최종 낙찰업체와 실제 계약금액은?',t))


class ComparisonRequestTests(unittest.TestCase):
    def setUp(self):
        docs,chunks=fixture();self.index=Index(docs,chunks)
        self.registry=EvidenceRegistry(docs,chunks)

    def session(self,llm=None,ids=('A','B')):
        return AnswerSession(self.index,llm or LLM(),self.registry,active_doc_ids=list(ids))

    def test_comparison_requests_and_claims_stay_within_each_document(self):
        llm=LLM();r=self.session(llm).ask('사업예산 비교',feature='comparison')
        self.assertEqual(r['status'],'answered');self.assertTrue(r['validation']['valid'])
        self.assertEqual([set(c['doc_id'] for c in e) for _,_,e in llm.requests],[{'A'},{'B'}])
        self.assertEqual(len(r['document_validation']),2)
        self.assertEqual(r['generation']['request_count'],2)
        self.assertEqual(r['usage']['input_tokens'],20)
        self.assertNotIn('payload',str(r['generation']))
        self.assertIn('| 항목 |',r['markdown'])

    def test_preflight_all_comparison_requests_before_any_generation(self):
        class Limited(LLM):
            def preflight_requests(inner,requests):
                self.assertEqual(len(requests),2);raise RuntimeError('limit')
        llm=Limited();r=self.session(llm).ask('예산 비교',feature='comparison')
        self.assertEqual(r['execution_status'],'error');self.assertEqual(llm.requests,[])

    def test_source_validation_failure_is_explained_and_never_retried(self):
        llm=LLM(mode='bad_quote');r=self.session(llm).ask('예산 비교',feature='comparison')
        self.assertEqual(r['execution_status'],'validation_failed')
        self.assertEqual(len(llm.requests),1)
        self.assertFalse(r['document_validation'][0]['valid'])
        self.assertTrue(r['document_validation'][0]['errors'])
        self.assertEqual(r['claims'],[]);self.assertNotIn('FABRICATED_QUOTE',r['markdown'])

    def test_wrong_document_claim_remains_rejected_with_exact_other_document_quote(self):
        class Crossed(LLM):
            def generate(inner,*args,**kwargs):
                result=super().generate(*args,**kwargs)
                result['payload']['claims'][0]['doc_id']='B'
                return result
        r=self.session(Crossed()).ask('예산 비교',feature='comparison')
        self.assertEqual(r['execution_status'],'validation_failed');self.assertFalse(r['claims'])

    def test_provider_failure_does_not_continue_to_second_document(self):
        llm=LLM(mode='error');r=self.session(llm).ask('예산 비교',feature='comparison')
        self.assertEqual(r['execution_status'],'error');self.assertEqual(len(llm.requests),1)

    def test_award_only_question_is_withheld_without_model_call(self):
        llm=LLM();r=self.session(llm,ids=('A',)).ask('최종 낙찰업체와 실제 계약금액은?')
        self.assertEqual(r['status'],'insufficient_evidence');self.assertEqual(llm.requests,[])
        self.assertFalse(r['claims']);self.assertTrue(r['source_chunks'])
        self.assertIn('낙찰 결과 공고',r['markdown'])
        self.assertNotIn('100백만원',r['markdown'])

    def test_mixed_question_reaches_model_with_requested_fact_instruction(self):
        llm=LLM();self.session(llm,ids=('A',)).ask('사업예산과 실제 계약금액을 구분해줘.')
        self.assertEqual(len(llm.requests),1)
        self.assertIn('과업예산과 실제 계약금액을 구분',llm.requests[0][1])


if __name__=='__main__':unittest.main()
