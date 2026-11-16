"""Evaluation-integrity tests on independently invented documents and labels."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rfp_rag.retrieval.retrieval import LocalSearchIndex
from regression.conversation.test_session import fixture
from rfp_rag.evaluation.evidence import load_dev_cases,latency,score_retrieval,user_steps
from rfp_rag.evaluation.evaluate import run_evaluation,save_report,summarize
from rfp_rag.evaluation.review import assess_review,review_template,JUDGEMENTS


class Client:
    def __init__(self,fail=False):self.responses=self;self.bodies=[];self.fail=fail
    def create(self,**body):
        self.bodies.append(body)
        if self.fail:
            error=RuntimeError('PRIVATE_PROVIDER_BODY')
            error.status_code=429
            raise error
        payload={'status':'insufficient_evidence','claims':[],'missing_information':['추가 근거 확인']}
        return {'status':'completed','output_text':json.dumps(payload,ensure_ascii=False),'output':[],
            'usage':{'input_tokens':100,'output_tokens':20,'total_tokens':120,
                     'input_tokens_details':{'cached_tokens':50}}}


class EvaluationTests(unittest.TestCase):
    def test_scoped_answered_with_missing_is_partial_and_later_case_still_executes(self):
        comparison=copy.deepcopy(self.case)
        comparison.update(feature='comparison',question='선택한 두 사업의 예산과 안정화 조건 비교',target_doc_ids=['A','B'])
        comparison['context']['selected_doc_ids']=['A','B']
        uncertainty=copy.deepcopy(self.case)
        uncertainty.update(case_id='INDEPENDENT_AWARD',question='최종 낙찰업체와 실제 계약금액은?',
            expected_status='insufficient_evidence',required_facts=[])
        class Inconsistent(Client):
            def create(inner,**body):
                result=super().create(**body)
                c=json.loads(body['input'][1]['content'])['evidence'][0]
                result['output_text']=json.dumps({'status':'answered','claims':[{
                    'doc_id':c['doc_id'],'category':'budget','text':c['text'],
                    'citations':[{'chunk_id':c['chunk_id'],'quote':c['text']}]}],
                    'missing_information':['현재 문서의 추가 조건'] if c['doc_id']=='B' else []})
                return result
        client=Inconsistent()
        r=run_evaluation(self.env,[comparison,uncertainty],profiles=['phase6_rrf'],enabled=True,
            case_ids=['DEV1','INDEPENDENT_AWARD'],approved_doc_ids=['A','B'],client=client)
        self.assertNotIn('halted_at_row',r)
        self.assertEqual(len(client.bodies),2)
        self.assertEqual(r['rows'][0]['result']['status'],'partial')
        self.assertEqual(r['rows'][1]['result']['status'],'insufficient_evidence')
        self.assertEqual(r['usage']['failed_attempts'],0)

    def test_validation_failure_is_reported_even_when_no_citations_are_rendered(self):
        class BadQuote(Client):
            def create(inner,**body):
                result=super().create(**body)
                c=json.loads(body['input'][1]['content'])['evidence'][0]
                payload={'status':'answered','claims':[{'doc_id':c['doc_id'],
                    'category':'budget','text':'예산 정보',
                    'citations':[{'chunk_id':c['chunk_id'],'quote':'INVENTED_SOURCE'}]}],
                    'missing_information':[]}
                result['output_text']=json.dumps(payload);return result
        r=self.live(BadQuote())
        self.assertEqual(r['metrics']['phase6_rrf']['validation_failed_turns'],1)
        self.assertIsNone(r['metrics']['phase6_rrf']['citation_source_validity'])
        self.assertEqual(r['rows'][0]['result']['claims'],[])

    def test_comparison_checks_all_request_reservations_before_first_call(self):
        self.case.update(feature='comparison',question='선택한 두 사업 예산 비교',target_doc_ids=['A','B'])
        self.case['context']['selected_doc_ids']=['A','B'];client=Client()
        r=run_evaluation(self.env,[self.case],profiles=['phase6_rrf'],enabled=True,
            case_ids=['DEV1'],approved_doc_ids=['A','B'],max_calls=1,client=client)
        self.assertEqual(client.bodies,[])
        self.assertEqual(r['metrics']['phase6_rrf']['generation_error_turns'],1)
        self.assertEqual(r['usage']['mock_calls'],0)

    def test_requested_award_results_do_not_spend_on_unrelated_rfp_answers(self):
        self.case['question']='최종 낙찰업체와 실제 계약금액을 확정해서 알려줘.'
        self.case['expected_status']='insufficient_evidence';client=Client()
        r=self.live(client)
        self.assertEqual(client.bodies,[])
        self.assertEqual(r['rows'][0]['result']['status'],'insufficient_evidence')
        self.assertEqual(r['usage']['mock_calls'],0)

    def test_mixed_cache_and_live_comparison_is_not_counted_as_pure_cache_latency(self):
        r=self.run_sample()
        r['rows'][0]['usage_delta'].update(live_api_calls=1,cache_hits=1)
        r=summarize(r);m=r['metrics']['phase6_rrf']
        self.assertEqual(m['mixed_live_cache_latency']['count'],1)
        self.assertEqual(m['fresh_live_latency']['count'],0)
        self.assertEqual(m['local_cache_latency']['count'],0)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.docs,self.chunks,self.registry=fixture()
        self.index=LocalSearchIndex(self.root/'index');self.index.build(self.chunks,{'min_df':1})
        self.addCleanup(self.index.close)
        (self.root/'configs').mkdir();(self.root/'data').mkdir()
        cfg={'conversation':{},'retrieval':{'method':'rrf','top_k':8,'per_doc_top_k':6,'fetch_k':40,
            'mmr_lambda':.75,'max_context_bytes':42000,'max_question_chars':4000}}
        (self.root/'configs/phase6.json').write_text(json.dumps(cfg))
        self.env={'root':self.root,'index':self.index,'documents':self.docs,'chunks':self.chunks,
            'registry':self.registry,'phase5_config':{'generation':{'model':'gpt-5.4-mini','cache_enabled':False}}}
        text=self.docs[0]['blocks'][0]['text']
        self.case={'case_id':'DEV1','split':'dev','feature':'fact','question':'사업기간은?',
            'context':{'selected_doc_ids':['A']},'expected_status':'answered','target_doc_ids':['B'],
            'required_facts':[{'fact_id':'f1','doc_id':'A','field':'duration',
                'expected_value':'GOLD_ONLY_SECRET_987654',
                'evidence':[{'doc_id':'A','sha256':self.docs[0]['sha256'],'block_id':'p1','quote':text}]}]}
        record={'case_id':'DEV1','fact_id':'f1','evidence_index':1,'evidence_kind':'fact','doc_id':'A',
            'gold_sha256':self.docs[0]['sha256'],'source_sha256':self.docs[0]['sha256'],
            'quote':text,'match_mode':'exact','location_resolved':True,
            'candidates':[{'block_id':'p1','char_start':0,'char_end':len(text)}]}
        self.metadata={'records':[record]};self.write_metadata()
        self.net=patch('socket.socket.connect',side_effect=AssertionError('network prohibited'))
        self.net.start();self.addCleanup(self.net.stop)
        key=patch.dict(os.environ,{'OPENAI_API_KEY':'test-only-not-a-real-provider-key'})
        key.start();self.addCleanup(key.stop)

    def write_metadata(self):
        (self.root/'data/source_evidence_validation.json').write_text(json.dumps(self.metadata))

    def run_sample(self,**kwargs):
        return run_evaluation(self.env,[self.case],profiles=['phase6_rrf'],**kwargs)

    def live(self,client,**kwargs):
        return self.run_sample(enabled=True,case_ids=['DEV1'],approved_doc_ids=['A'],client=client,**kwargs)

    def test_default_executes_retrieval_with_zero_api_calls(self):
        r=self.run_sample()
        self.assertEqual(r['usage']['api_calls'],0)
        self.assertEqual(r['status'],'offline_measured')
        self.assertEqual(r['metrics']['phase6_rrf']['source_recovery']['micro_source_span_recall'],1)
        self.assertIsNone(r['semantic_accuracy'])
        self.assertFalse(r['rows'][0]['reviewable'])

    def test_lightweight_case_isolation_preserves_full_catalog_and_aliases(self):
        from rfp_rag.conversation.session import ConversationSession
        self.case['context']['selected_doc_ids']=['COPY_A']
        r=self.run_sample()
        self.assertEqual(r['routing_catalog_fingerprint'],ConversationSession(self.docs).catalog_fingerprint)
        self.assertEqual(r['rows'][0]['result']['effective_doc_ids'],['A'])
        self.assertEqual(r['rows'][0]['result']['source_chunks'][0]['text'],self.docs[0]['blocks'][0]['text'])

    def test_runtime_inputs_exclude_targets_expected_values_and_statuses(self):
        inputs=user_steps(self.case)
        self.assertEqual(inputs,{'selected_doc_ids':['A'],'filters':{},'steps':[{'turn_id':'t1','user':'사업기간은?'}]})
        client=Client();r=self.live(client)
        sent=json.dumps(client.bodies,ensure_ascii=False)
        self.assertNotIn('GOLD_ONLY_SECRET',sent)
        self.assertEqual(r['rows'][0]['result']['effective_doc_ids'],['A'])
        self.assertFalse(r['gold_used_as_generation_input'])

    def test_client_budget_and_usage_are_shared_across_isolated_cases(self):
        second=copy.deepcopy(self.case);second['case_id']='DEV2'
        client=Client()
        r=run_evaluation(self.env,[self.case,second],profiles=['phase6_rrf'],enabled=True,
            case_ids=['DEV1','DEV2'],approved_doc_ids=['A'],client=client,max_calls=2)
        self.assertEqual(len(client.bodies),2)
        self.assertEqual(r['usage']['mock_calls'],2)
        self.assertEqual(sum(x['usage_delta']['api_calls'] for x in r['rows']),2)
        self.assertEqual(r['usage']['reported_input_tokens'],200)
        self.assertEqual(r['usage']['reported_cached_input_tokens'],100)

    def test_turn_budget_rejected_before_request(self):
        client=Client()
        case=copy.deepcopy(self.case);case['feature']='followup'
        case['turns']=[{'turn_id':'t1','user':'예산은?'},{'turn_id':'t2','user':'몇 개월이야?'}]
        with self.assertRaises(ValueError):run_evaluation(self.env,[case],profiles=['phase6_rrf'],enabled=True,
            case_ids=['DEV1'],approved_doc_ids=['A'],max_calls=1,client=client)
        self.assertEqual(client.bodies,[])

    def test_no_live_sweep_without_explicit_case_and_profile_selection(self):
        with self.assertRaises(ValueError):self.run_sample(enabled=True,approved_doc_ids=['A'])
        with self.assertRaises(ValueError):run_evaluation(self.env,[self.case],enabled=True,
            case_ids=['DEV1'],approved_doc_ids=['A'])

    def test_failure_stops_remaining_live_cases_without_retry(self):
        second=copy.deepcopy(self.case);second['case_id']='DEV2'
        client=Client(fail=True)
        r=run_evaluation(self.env,[self.case,second],profiles=['phase6_rrf'],enabled=True,
            case_ids=['DEV1','DEV2'],approved_doc_ids=['A'],client=client)
        self.assertEqual(len(client.bodies),1)
        self.assertIn('halted_at_row',r)
        self.assertEqual(r['usage']['usage_unknown_attempts'],1)
        self.assertNotIn('PRIVATE_PROVIDER_BODY',json.dumps(r))

    def test_missing_alignment_is_not_reported_as_perfect_recall(self):
        self.metadata={'records':[]};self.write_metadata()
        r=self.run_sample()
        score=r['metrics']['phase6_rrf']['source_recovery']
        self.assertEqual(score['unmapped_links'],1)
        self.assertIsNone(score['micro_source_span_recall'])

    def test_quotation_requires_all_characters_not_only_a_number(self):
        case=copy.deepcopy(self.case)
        label={'link_id':'DEV1|fact|f1|1','mapped':True,'alternatives':[{
            'required_non_whitespace_characters':10,'chunk_coverage':{'C1':[[0,5]],'C2':[[5,10]]}}]}
        a=score_retrieval(case,'t1',[{'chunk_id':'C1'}],{'records':[label]})
        b=score_retrieval(case,'t1',[{'chunk_id':'C1'},{'chunk_id':'C2'}],{'records':[label]})
        self.assertEqual(a['packed_context']['source_span_recall'],0)
        self.assertEqual(a['reciprocal_rank_first_source_overlap'],1)
        self.assertEqual(b['packed_context']['source_span_recall'],1)

    def test_mock_provenance_is_recorded_separately_from_live_calls(self):
        r=self.live(Client())
        self.assertEqual(r['usage']['mock_calls'],1)
        self.assertEqual(r['usage']['live_api_calls'],0)
        self.assertEqual(r['metrics']['phase6_rrf']['fresh_live_latency']['count'],0)

    def test_configured_cost_uses_cached_rate_and_checks_model(self):
        prices={'model':'gpt-5.4-mini','input_per_million':2,'cached_input_per_million':1,'output_per_million':4}
        r=self.live(Client(),pricing=prices)
        self.assertAlmostEqual(r['cost_estimate']['estimated_usd_for_reported_tokens'],.00023)
        prices['model']='another-model'
        with self.assertRaises(ValueError):self.live(Client(),pricing=prices)

    def test_unconfigured_price_is_unknown_rather_than_zero(self):
        r=self.live(Client())
        self.assertIsNone(r['cost_estimate'])

    def test_synthetic_parser_failure_is_explicitly_skipped(self):
        self.case['context']['tool_results']=[{'status':'error'}]
        r=self.run_sample()
        self.assertEqual(r['coverage']['skipped_rows'],{'synthetic_parser_fixture_not_simulated':1})
        self.assertEqual(r['usage']['api_calls'],0)

    def test_candidate_list_is_not_injected_as_active_gold_scope(self):
        self.case['context']={'selected_doc_ids':[],'candidate_doc_ids':['A','B']}
        r=self.run_sample()
        self.assertEqual(r['rows'][0]['result']['status'],'needs_clarification')
        self.assertEqual(r['rows'][0]['result']['active_doc_ids'],[])

    def test_whole_summary_never_falls_back_to_top_k(self):
        self.case['feature']='summary'
        r=self.run_sample()
        self.assertEqual(r['coverage']['skipped_rows'],{'summary_folder_not_supplied':1})
        self.assertEqual(r['usage']['api_calls'],0)

    def test_missing_saved_summary_does_not_generate(self):
        self.case['feature']='summary'
        r=self.run_sample(summary_folder=self.root)
        self.assertEqual(r['coverage']['skipped_rows'],{'no_saved_content_revision':1})

    def test_saved_summary_is_imported_and_revalidated_without_generation(self):
        from regression.summary_content.test_integration import saved_result, make_claim
        from rfp_rag.summary_content.storage import save_content_result
        chunks=[c for c in self.chunks if c['doc_id']=='A']
        claim=make_claim(self.registry,chunks,'duration',0)
        raw=saved_result(self.docs[0],chunks,[claim])
        raw['missing_information']=['나머지 요약 항목 확인 필요']
        save_content_result(self.root,raw,copy.deepcopy(raw))
        self.case['feature']='summary'
        self.case['target_doc_ids']=['A']
        r=self.run_sample(summary_folder=self.root)
        self.assertEqual(r['coverage']['imported_summary_rows'],1)
        row=r['rows'][0]
        self.assertTrue(row['citation_audit']['valid'])
        self.assertEqual(len(row['summary_field_states']),11)
        self.assertTrue(row['historical_generation_usage_not_rebilled'])
        self.assertEqual(r['usage']['api_calls'],0)

    def test_summary_with_wrong_latest_fingerprint_is_not_accepted(self):
        folder=self.root/'content_documents/A'
        folder.mkdir(parents=True)
        (folder/'latest.json').write_text(json.dumps({'folder':'../../elsewhere'}))
        self.case['feature']='summary'
        r=self.run_sample(summary_folder=self.root)
        self.assertEqual(r['coverage']['skipped_rows'],{'saved_summary_validation_failed':1})

    def test_holdout_is_rejected_before_opening_its_file(self):
        with patch.object(Path,'read_text',side_effect=AssertionError('must not read')):
            with self.assertRaises(ValueError):load_dev_cases(self.root/'questions.holdout.jsonl')
        c=copy.deepcopy(self.case);c['split']='holdout'
        with self.assertRaises(ValueError):run_evaluation(self.env,[c])

    def test_dev_loader_rejects_duplicate_case_ids(self):
        p=self.root/'cases.jsonl';p.write_text('\n'.join(json.dumps(self.case) for _ in range(2)))
        with self.assertRaises(ValueError):load_dev_cases(p)

    def test_source_validity_without_claims_is_not_one_hundred_percent(self):
        r=self.live(Client())
        self.assertIsNone(r['metrics']['phase6_rrf']['citation_source_validity'])
        self.assertIsNone(r['semantic_accuracy'])

    def test_review_remains_pending_and_never_fills_itself(self):
        r=self.live(Client());review=review_template(r)
        scored=assess_review(r,review)
        self.assertEqual(scored['status'],'review_pending')
        self.assertIsNone(scored['whole_sample_accuracy'])
        self.assertTrue(all(v is None for row in review['rows'] for v in row['judgements'].values()))

    def test_review_is_bound_to_result_reference_and_user_scope(self):
        r=self.live(Client());review=review_template(r)
        r['rows'][0]['result']['missing_information']=['different result']
        with self.assertRaises(ValueError):assess_review(r,review)

    def test_review_does_not_accept_truthy_text_in_place_of_a_judgement(self):
        r=self.live(Client());review=review_template(r)
        review['rows'][0]['judgements']['answer_correct']='true'
        with self.assertRaises(ValueError):assess_review(r,review)

    def test_failed_request_cost_marks_usage_as_incomplete(self):
        prices={'model':'gpt-5.4-mini','input_per_million':2,'cached_input_per_million':1,'output_per_million':4}
        r=self.live(Client(fail=True),pricing=prices)
        self.assertFalse(r['cost_estimate']['complete_usage_available'])

    def test_positive_mock_review_cannot_claim_live_sample_accuracy(self):
        r=self.live(Client());review=review_template(r);review['reviewer']='test reviewer'
        for row in review['rows']:row['judgements']=dict.fromkeys(JUDGEMENTS,True)
        scored=assess_review(r,review)
        self.assertEqual(scored['status'],'review_complete')
        self.assertEqual(scored['accuracy_on_reviewed_rows'],1)
        self.assertIsNone(scored['whole_sample_accuracy'])

    def test_save_creates_distinct_runs_and_preserves_review(self):
        r=self.run_sample();first=save_report(self.root,r)
        p=first/'human_review.json';p.write_text('preserved original review')
        second=save_report(self.root,r)
        self.assertNotEqual(first,second)
        self.assertEqual(p.read_text(),'preserved original review')

    def test_p95_uses_small_sample_nearest_rank(self):
        self.assertEqual(latency([1,2,100])['p95_ms'],100)
        self.assertIsNone(latency([])['p50_ms'])


if __name__=='__main__':unittest.main()
