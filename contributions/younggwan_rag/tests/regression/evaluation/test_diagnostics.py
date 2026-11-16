"""Latest-run selection and explicit review use independent synthetic records."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from rfp_rag.evaluation.diagnostics import load_current_review,comparison_checklist,final_verification
from rfp_rag.evaluation.review import review_template


def sample(*,enabled=True,fingerprint='test-live'):
    result={'status':'partial','execution_status':'completed','answer_generated':True,
        'claims':[{'doc_id':'DOC_A','category':'duration','text':'계약일로부터 14개월',
            'citations':[{'quote':'사업기간: 계약일로부터 14개월','locator':'section one',
                'source_ranges':[{'block_id':'block-one','char_start':0,'char_end':19}]}]}],
        'missing_information':['DOC_A 문서의 안정화 조건'],
        'comparison_audit':{'durations':[{'role':'duration_estimate'}]},
        'document_validation':[{'doc_ids':['DOC_A'],'missing_scope_audit':{'excluded':['DOC_B의 기간']}}],
        'generation':{'execution_mode':'live_openai'}}
    row={'row_id':'phase6_rrf/INVENTED/t1','question':'두 사업 기간 비교',
        'expected_status':'answered','expected_scope':['DOC_A','DOC_B'],
        'expected_facts':[],'required_behaviors':[],'feature':'comparison',
        'scope_match':True,'reviewable':True,'result':result,'citation_audit':{'valid':True}}
    return {'run_fingerprint':fingerprint,'rows':[row],'enabled':enabled,
        'execution_origin':'live_openai' if enabled else 'offline',
        'planned_qa_turns_per_profile':1,'profiles':['phase6_rrf'],
        'status':'live_review_pending' if enabled else 'offline_measured',
        'coverage':{'skipped_rows':{}},'usage':{'api_calls':2 if enabled else 0}}


class DiagnosticsTests(unittest.TestCase):
    def test_status_adjustment_audit_does_not_become_an_automatic_correctness_pass(self):
        r=sample();adjustment={'original_status':'answered','effective_status':'partial',
            'claims_changed':False,'missing_information_changed':False,'additional_api_calls':0}
        r['rows'][0]['result']['generation']['document_requests']=[{}, {'response_status_adjustment':adjustment}]
        before=copy.deepcopy(r);check=comparison_checklist(r)[0]
        self.assertEqual(check['conservative_status_adjustments'],[adjustment])
        self.assertFalse(check['automatic_semantic_pass'])
        self.assertEqual(r,before)

    def write_run(self,folder,report):
        folder.mkdir();(folder/'report.json').write_text(json.dumps(report))
        (folder/'human_review.json').write_text(json.dumps(review_template(report)))

    def test_current_live_folder_wins_over_offline_or_a_newer_unrelated_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);offline=root/'offline';live=root/'live';newer=root/'99999999_other'
            self.write_run(offline,sample(enabled=False,fingerprint='offline'))
            self.write_run(live,sample());self.write_run(newer,sample(fingerprint='other'))
            loaded=load_current_review(live,offline)
            self.assertEqual(loaded['folder'],live)
            self.assertEqual(loaded['report']['run_fingerprint'],'test-live')

    def test_offline_folder_is_selected_only_when_this_namespace_has_no_live_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'offline';self.write_run(folder,sample(enabled=False))
            self.assertEqual(load_current_review(None,folder)['report']['execution_origin'],'offline')

    def test_wrong_review_binding_is_rejected_instead_of_reusing_stale_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'live';r=sample();self.write_run(folder,r)
            wrong=review_template(sample(fingerprint='another-run'))
            (folder/'human_review.json').write_text(json.dumps(wrong))
            with self.assertRaises(ValueError):load_current_review(folder,folder)

    def test_execution_success_is_separate_from_unreviewed_answer_correctness(self):
        r=sample();v=final_verification(r,review_template(r))
        self.assertEqual(v['execution_status'],'execution_complete')
        self.assertEqual(v['human_review_status'],'review_pending')
        self.assertIsNone(v['whole_sample_accuracy'])
        self.assertFalse(v['automatic_semantic_pass'])
        self.assertEqual(v['comparison_rows_with_missing_information'],1)

    def test_no_offline_or_incomplete_invocation_is_declared_live_complete(self):
        for change in [{'enabled':False},{'planned_qa_turns_per_profile':2},
                       {'halted_at_row':'phase6_rrf/INVENTED/t1'}]:
            r=sample();r.update(change);v=final_verification(r,review_template(r))
            self.assertNotEqual(v['execution_status'],'execution_complete')
        r=sample();r['execution_origin']='injected_test_client'
        self.assertEqual(final_verification(r,review_template(r))['execution_status'],'test_execution_only')

    def test_blocked_validation_and_missing_result_remain_visible(self):
        r=sample();r['rows'][0]['result']['execution_status']='validation_failed'
        v=final_verification(r,review_template(r))
        self.assertEqual(v['blocked_or_failed_turns'],1)
        self.assertEqual(v['execution_status'],'execution_needs_attention')
        r=sample();r['rows'][0]['result']={}
        self.assertEqual(final_verification(r,review_template(r))['execution_status'],'execution_needs_attention')

    def test_latest_human_file_edits_are_loaded_without_model_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'live';r=sample();self.write_run(folder,r)
            v=load_current_review(folder,folder);self.assertEqual(v['assessment']['reviewed_rows'],0)
            review=v['review'];review['reviewer']='independent test reviewer'
            review['rows'][0]['judgements']={k:True for k in review['rows'][0]['judgements']}
            (folder/'human_review.json').write_text(json.dumps(review))
            v=load_current_review(folder,folder)
            self.assertEqual(v['assessment']['status'],'review_complete')
            self.assertFalse(final_verification(v['report'],v['review'])['automatic_semantic_pass'])

    def test_checklist_preserves_duration_and_missing_scope_provenance_without_mutation(self):
        r=sample();before=copy.deepcopy(r);v=comparison_checklist(r)
        self.assertEqual(v[0]['duration_claims'][0]['citations'][0]['locator'],'section one')
        self.assertEqual(v[0]['comparison_audit'],r['rows'][0]['result']['comparison_audit'])
        self.assertEqual(v[0]['document_missing_scope_audits'][0]['missing_scope_audit']['excluded'],['DOC_B의 기간'])
        self.assertFalse(v[0]['automatic_semantic_pass'])
        v[0]['duration_claims'][0]['citations'][0]['source_ranges'][0]['block_id']='CHANGED'
        self.assertEqual(r,before)

    def test_noncomparison_and_skipped_rows_do_not_enter_the_comparison_checklist(self):
        r=sample();r['rows'][0]['feature']='fact';self.assertEqual(comparison_checklist(r),[])
        r=sample();r['rows'][0]['skipped_reason']='fixture';self.assertEqual(comparison_checklist(r),[])


if __name__=='__main__':unittest.main()
