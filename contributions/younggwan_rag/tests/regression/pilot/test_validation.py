"""Reject semantic auto-pass, stale submissions and misbound pilot evidence."""
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from regression.conversation.test_focus import GroundedClient, synthetic_environment
from rfp_rag.pilot.pilot import PilotSession, review_template, save_review
from rfp_rag.pilot.validation import run_validation, validation_plan, validation_status


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = synthetic_environment(Path(self.tmp.name))
        self.addCleanup(self.env['index'].close)
        self.key = patch.dict(os.environ, {'OPENAI_API_KEY': 'synthetic-not-a-real-key'})
        self.key.start()
        self.addCleanup(self.key.stop)

    def test_offline_exercise_covers_followup_switch_comparison_and_abstention_without_generation(self):
        with patch('rfp_rag.generation.llm.OpenAIGroundedLLM.generate', side_effect=AssertionError('generation prohibited')):
            folder, report = run_validation(self.env, ['A', 'B'])
        self.assertEqual(report['executed_cases'], 8)
        self.assertTrue(report['mechanical_checks_passed'])
        self.assertEqual(report['session']['usage']['api_calls'], 0)
        self.assertIsNone(report['semantic_accuracy'])
        self.assertFalse(report['independent_quality_verified'])

    def test_absent_or_incomplete_live_approval_prevents_first_call(self):
        client = GroundedClient()
        for approved in [[], ['A']]:
            with self.assertRaises(ValueError):
                run_validation(self.env, ['A', 'B'], enabled=True, approved_doc_ids=approved, client=client)
        self.assertEqual(client.bodies, [])

    def test_one_shared_budget_and_no_gold_or_review_in_generation_inputs(self):
        client = GroundedClient('absence')
        folder, report = run_validation(self.env, ['A', 'B'], enabled=True,
            approved_doc_ids=['A', 'B'], client=client)
        self.assertEqual(len(client.bodies), 7)
        self.assertEqual(report['session']['usage']['mock_calls'], 7)
        self.assertEqual(report['session']['usage']['live_api_calls'], 0)
        self.assertTrue(report['mechanical_checks_passed'])
        self.assertEqual(report['human_review_status'], 'review_pending')
        self.assertIsNone(report['semantic_accuracy'])
        self.assertIsNone(report['session']['accuracy_on_reviewed_live_turns'])
        for body in client.bodies:
            text = json.dumps(body)
            self.assertNotIn('review_hint', text)
            self.assertNotIn('expected_status', text)
            self.assertNotIn('forbidden_topic', text)

    def test_first_generation_validation_error_stops_remaining_exercise(self):
        client = GroundedClient('wrong_quote')
        folder, report = run_validation(self.env, ['A'], enabled=True, approved_doc_ids=['A'], client=client)
        self.assertEqual(len(client.bodies), 1)
        self.assertEqual(report['executed_cases'], 1)
        self.assertFalse(report['mechanical_checks_passed'])
        self.assertEqual(len(report['session']['failed_turns']), 1)

    def test_six_case_exercise_surfaces_claim_gaps_without_semantic_auto_pass(self):
        from regression.pilot.test_claim_audit import NamedItemClient
        client = NamedItemClient()
        folder, report = run_validation(self.env, ['A'], enabled=True, approved_doc_ids=['A'], client=client)
        self.assertEqual(report['executed_cases'], 6)
        self.assertTrue(report['mechanical_checks_passed'])
        self.assertGreater(report['claim_citation_review']['claims_with_literal_anchor_gaps'], 0)
        self.assertFalse(report['automatic_semantic_pass'])
        self.assertIsNone(report['semantic_accuracy'])
        self.assertEqual(report['human_review_status'], 'review_pending')
        self.assertEqual(report['session']['usage']['live_api_calls'], 0)
        self.assertLessEqual(len(client.bodies), 5)
        for body in client.bodies:
            self.assertNotIn('claim_citation_audit', json.dumps(body))

    def test_claim_diagnostics_are_recomputed_without_modifying_old_validation_reports(self):
        from rfp_rag.pilot.pilot import digest
        from regression.pilot.test_claim_audit import NamedItemClient
        folder, report = run_validation(self.env, ['A'], enabled=True, approved_doc_ids=['A'], client=NamedItemClient())
        path = folder / 'validation_report.json'
        old = json.loads(path.read_text())
        for row in old['rows']:
            row.pop('claim_citation_audit')
        old['report_fingerprint'] = digest({k: v for k, v in old.items() if k != 'report_fingerprint'})
        path.write_text(json.dumps(old))
        before = {p: p.read_bytes() for p in folder.rglob('*') if p.is_file()}
        with patch('rfp_rag.generation.llm.OpenAIGroundedLLM.generate', side_effect=AssertionError('no API')):
            updated = validation_status(folder, include_reviews=False)
        self.assertGreater(updated['claim_citation_review']['claims_with_literal_anchor_gaps'], 0)
        self.assertEqual(updated['diagnostics']['diagnostic_api_calls'], 0)
        self.assertEqual(before, {p: p.read_bytes() for p in folder.rglob('*') if p.is_file()})

    def test_saved_diagnostics_expose_rejection_reason_without_showing_rejected_model_quotes(self):
        folder, report = run_validation(self.env, ['A'], enabled=True,
            approved_doc_ids=['A'], client=GroundedClient('wrong_quote'))
        diagnostic = report['diagnostics']
        self.assertFalse(diagnostic['mechanical_checks_passed'])
        self.assertEqual(diagnostic['diagnostic_api_calls'], 0)
        self.assertTrue(diagnostic['read_only'])
        failed = diagnostic['turns'][0]
        self.assertEqual(failed['execution_status'], 'validation_failed')
        self.assertTrue(failed['validation_errors'])
        self.assertTrue(failed['citation_checks'])
        self.assertNotIn('model_quote', json.dumps(diagnostic))
        self.assertNotIn('원문에 존재하지 않는 구절', json.dumps(diagnostic, ensure_ascii=False))

    def test_diagnostic_read_does_not_generate_change_files_or_require_human_review(self):
        folder, report = run_validation(self.env, ['A'])
        files = {p: p.read_bytes() for p in folder.rglob('*') if p.is_file()}
        with patch('rfp_rag.generation.llm.OpenAIGroundedLLM.generate', side_effect=AssertionError('no generation')), \
                patch('rfp_rag.pilot.validation.session_report', side_effect=AssertionError('review not needed')):
            diagnostic = validation_status(folder, include_reviews=False)['diagnostics']
        self.assertEqual(diagnostic['executed_cases'], 6)
        self.assertEqual(files, {p: p.read_bytes() for p in folder.rglob('*') if p.is_file()})


    def test_review_of_one_exact_turn_does_not_approve_other_answers(self):
        folder, report = run_validation(self.env, ['A'], enabled=True,
            approved_doc_ids=['A'], client=GroundedClient('absence'))
        row = report['rows'][0]
        turn = folder / 'turns' / row['turn_id']
        record = json.loads((turn / 'record.json').read_text())
        review = review_template(record)
        review['reviewer'] = 'Synthetic reviewer'
        review['judgements'] = dict.fromkeys(review['judgements'], True)
        review['judgements']['required_facts_complete'] = False
        save_review(turn, review)
        updated = validation_status(folder)
        self.assertEqual(updated['session']['reviewed_turns'], 1)
        self.assertEqual(updated['session']['fully_correct_reviewed_turns'], 0)
        self.assertEqual(updated['human_review_status'], 'review_pending')
        self.assertFalse(updated['independent_quality_verified'])

    def test_record_or_plan_swaps_are_rejected(self):
        folder, report = run_validation(self.env, ['A'])
        path = folder / 'validation_plan.json'
        plan = json.loads(path.read_text())
        plan['cases'][0]['question'] = 'Changed question'
        path.write_text(json.dumps(plan))
        with self.assertRaises(ValueError):
            validation_status(folder)

    def test_changed_report_checks_are_not_trusted(self):
        folder, report = run_validation(self.env, ['A'])
        path = folder / 'validation_report.json'
        stored = json.loads(path.read_text())
        stored['rows'][0]['checks']['document_scope'] = False
        path.write_text(json.dumps(stored))
        with self.assertRaises(ValueError):
            validation_status(folder)

    def test_same_submission_id_cannot_reuse_answer_after_document_change(self):
        pilot = PilotSession(self.env)
        pilot.select(['A'])
        pilot.submit('구체적인 사업 내용은?', request_id='unchanged-id')
        pilot.select(['B'])
        with self.assertRaises(ValueError):
            pilot.submit('구체적인 사업 내용은?', request_id='unchanged-id')
        self.assertEqual(len(pilot.records), 1)

    def test_same_submission_id_cannot_reuse_answer_after_filter_or_approval_change(self):
        for change in ['filters', 'approval']:
            with self.subTest(change=change):
                pilot = PilotSession(self.env)
                pilot.select(['A'])
                pilot.submit('사업예산은?', request_id='same-id')
                if change == 'filters':
                    pilot.select(['A'], filters={'document_year': 2036})
                else:
                    pilot.configure_generation(enabled=False, approved_doc_ids=['A'])
                with self.assertRaises(ValueError):
                    pilot.submit('사업예산은?', request_id='same-id')

    def test_validation_plan_uses_only_explicit_context_and_bounded_document_count(self):
        plan = validation_plan(['CALLER_A', 'CALLER_B'])
        self.assertEqual(plan['maximum_generation_requests'], 8)
        self.assertNotIn('required_facts', json.dumps(plan))
        with self.assertRaises(ValueError):
            validation_plan(['A', 'B', 'C'])


if __name__ == '__main__':
    unittest.main()
