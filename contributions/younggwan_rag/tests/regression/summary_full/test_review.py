"""Review-contract tests use invented source-backed records, not model scores."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import rfp_rag.summary_full.review as review


def result(*, rejected=False, live=True):
    citation = {'chunk_id': 'c1', 'doc_id': 'd1', 'sha256': 'a'*64,
        'file_name': '합성테스트.pdf', 'locator': 'PDF p.1',
        'quote': '사업예산은 100원이며 부가세를 포함합니다.',
        'source_ranges': [{'char_start': 0, 'char_end': 25}],
        'provenance_verified': True}
    raw = {'doc_id': 'd1', 'category': 'budget',
           'text': '[budget_vat|fact] 사업예산은 100원이며 부가세 포함입니다.',
           'citations': [citation]}
    fields = [{'field_id': fid, 'label': label, 'state': 'not_found_in_processed_input',
               'claims': []} for fid, label in review.FIELDS]
    field = next(f for f in fields if f['field_id'] == 'budget_vat')
    field['state'] = 'evidence_found'
    field['claims'] = [{'text': '사업예산은 100원이며 부가세 포함입니다.',
                        'mode': 'fact', 'citations': [copy.deepcopy(citation)]}]
    value = {'doc_id': 'd1', 'source_sha256': 'a'*64, 'run_fingerprint': 'run-1',
        'execution_status': 'completed_with_rejections' if rejected else 'completed',
        'status': 'partial', 'claims': [raw], 'fields': fields,
        'source_chunk_ids': ['c1', 'c2'], 'input_received_chunk_ids': ['c1', 'c2'],
        'checked_chunk_ids': ['c1', 'c2'], 'input_chunk_coverage': 1.0,
        'checked_chunk_coverage': 1.0, 'rejected_claims': [],
        'model_origin': {'all_live_or_live_cache': live},
        'batch_outcomes': [{'batch_id': 'map-0001', 'chunk_ids': ['c1', 'c2'],
            'status': 'completed_with_rejections' if rejected else 'completed',
            'execution_mode': 'live_openai' if live else 'injected_test_client',
            'request_fingerprint': 'request-1'}]}
    if rejected:
        value['rejected_claims'] = [{'rejection_id': 'map-0001-claim-0002',
            'batch_id': 'map-0001', 'claim_index': 2, 'field_id': 'budget_vat',
            'reason': 'literal_quote_not_found',
            'raw_claim': {'text': '[budget_vat|fact] 검증되지 않은 999원 주장'}}]
    return value


def references():
    return {'samples': [{'sample_id': 'D01', 'doc_id': 'd1', 'source_sha256': 'a'*64}],
        'rows': [{'sample_id': 'D01', 'field_id': fid, 'label': label,
                  'expected': '합성 기준 ' + label, 'conditions': '원문 조건 대조',
                  'source_reference': 'PDF p.1'} for fid, label in review.FIELDS]}


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def save_csv(path, headers, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader(); writer.writerows(rows)


def fill(folder, *, disposition=None):
    path = Path(folder)/'comparison.csv'
    headers, rows = csv_rows(path)
    for row in rows:
        for key in review.JUDGEMENTS:
            row[key] = '통과'
    save_csv(path, headers, rows)
    if disposition is not None:
        path = Path(folder)/'rejection_review.csv'
        headers, rows = csv_rows(path)
        for row in rows:
            row['판정'] = disposition
        save_csv(path, headers, rows)


class ReviewContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def test_new_review_has_11_blank_rows_utf8bom_and_never_auto_accepts(self):
        value = result()
        paths = review.create_review_files(self.folder, value, references())
        self.assertEqual(set(paths), {'comparison_csv', 'rejection_review_csv', 'binding_json'})
        raw = Path(paths['comparison_csv']).read_bytes()
        self.assertTrue(raw.startswith(b'\xef\xbb\xbf'))
        _, rows = csv_rows(paths['comparison_csv'])
        self.assertEqual(len(rows), 11)
        self.assertTrue(all(row[key] == '' for row in rows for key in review.JUDGEMENTS))
        assessment = review.assess_review_files(self.folder, value, '검토자')
        self.assertEqual(assessment['status'], 'review_pending')
        self.assertEqual(len(assessment['pending_judgements']), 33)
        self.assertFalse(assessment['case_accepted'])

    def test_positive_human_review_accepts_one_full_live_case_only(self):
        value = result(); review.create_review_files(self.folder, value, references())
        fill(self.folder)
        assessment = review.assess_review_files(self.folder, value, '원문 검토자')
        self.assertTrue(assessment['case_accepted'], assessment)
        self.assertEqual(assessment['positive_judgements'], 33)
        self.assertFalse(assessment['semantic_quality_automatically_verified'])
        self.assertFalse(assessment['system_quality_verified'])

    def test_nonblank_reviewer_required(self):
        value = result(); review.create_review_files(self.folder, value); fill(self.folder)
        for reviewer in ['', '   ', None, 42]:
            with self.subTest(reviewer=reviewer):
                self.assertFalse(review.assess_review_files(self.folder, value, reviewer)['case_accepted'])

    def test_rejection_disposition_is_separate_and_not_prefilled(self):
        value = result(rejected=True)
        review.create_review_files(self.folder, value, references()); fill(self.folder)
        assessment = review.assess_review_files(self.folder, value, '검토자')
        self.assertEqual(assessment['status'], 'review_pending')
        self.assertFalse(assessment['case_accepted'])
        self.assertEqual(assessment['pending_judgements'][0]['rejection_id'], 'map-0001-claim-0002')
        fill(self.folder, disposition='제외 적절')
        self.assertTrue(review.assess_review_files(self.folder, value, '검토자')['case_accepted'])

    def test_required_summary_supplement_blocks_even_positive_11_field_review(self):
        value = result(rejected=True); review.create_review_files(self.folder, value)
        fill(self.folder, disposition='요약 보완 필요')
        assessment = review.assess_review_files(self.folder, value, '검토자')
        self.assertEqual(assessment['status'], 'review_failed')
        self.assertFalse(assessment['case_accepted'])

    def test_blank_and_arbitrary_rejection_dispositions_are_not_auto_accepted(self):
        value = result(rejected=True); review.create_review_files(self.folder, value)
        for text in ['', '통과', 'true', '제외적절']:
            fill(self.folder, disposition=text)
            self.assertFalse(review.assess_review_files(self.folder, value, '검토자')['case_accepted'])

    def test_rejected_claim_is_diagnostic_only_not_in_normal_summary_columns(self):
        value = result(rejected=True); paths = review.create_review_files(self.folder, value)
        _, rows = csv_rows(paths['comparison_csv'])
        self.assertNotIn('999원', '\n'.join(row['실제요약'] + row['검증된원문인용'] for row in rows))
        _, rejected = csv_rows(paths['rejection_review_csv'])
        self.assertIn('999원', rejected[0]['검증실패모델주장_진단용'])

    def test_formula_prefixes_are_escaped_in_normal_and_diagnostic_csv(self):
        value = result(rejected=True)
        value['fields'][4]['claims'][0]['text'] = '=HYPERLINK("bad")'
        value['rejected_claims'][0]['raw_claim']['text'] = '+UNVERIFIED()'
        refs = references(); refs['rows'][0]['expected'] = '@UNTRUSTED()'
        paths = review.create_review_files(self.folder, value, refs)
        _, rows = csv_rows(paths['comparison_csv'])
        self.assertTrue(rows[0]['기준내용초안'].startswith("'@"))
        self.assertTrue(rows[4]['실제요약'].startswith("'="))
        _, rejected = csv_rows(paths['rejection_review_csv'])
        self.assertTrue(rejected[0]['검증실패모델주장_진단용'].startswith("'+"))

    def test_no_reference_document_has_no_fabricated_expected_answer(self):
        value = result(); value['doc_id'] = 'unlisted'
        paths = review.create_review_files(self.folder, value, references())
        _, rows = csv_rows(paths['comparison_csv'])
        self.assertTrue(all(row['기준내용초안'] == '' for row in rows))
        self.assertTrue(all(row['필수조건'] == '' and row['기준표구분'] == '원문 직접 대조' for row in rows))

    def test_reviewed_csv_is_preserved_on_repeated_create(self):
        value = result(rejected=True); refs = references()
        paths = review.create_review_files(self.folder, value, refs)
        fill(self.folder, disposition='제외 적절')
        before = {key: Path(p).read_bytes() for key, p in paths.items()}
        review.create_review_files(self.folder, value, refs)
        self.assertEqual(before, {key: Path(p).read_bytes() for key, p in paths.items()})

    def test_stale_answer_source_execution_and_origin_invalidate_review(self):
        original = result(); review.create_review_files(self.folder, original); fill(self.folder)
        for change in [lambda r:r['claims'][0].update(text='변경된 답변'),
                       lambda r:r.update(source_sha256='b'*64),
                       lambda r:r.update(execution_status='incomplete'),
                       lambda r:r['model_origin'].update(all_live_or_live_cache=False),
                       lambda r:r.update(run_fingerprint='different-run')]:
            value = copy.deepcopy(original); change(value)
            with self.subTest(change=change):
                assessment = review.assess_review_files(self.folder, value, '검토자')
                self.assertEqual(assessment['status'], 'invalid_review')
                self.assertFalse(assessment['case_accepted'])

    def test_reference_change_does_not_overwrite_old_review(self):
        value = result(); refs = references(); paths = review.create_review_files(self.folder, value, refs)
        fill(self.folder); before = Path(paths['comparison_csv']).read_bytes()
        refs['rows'][0]['expected'] = '달라진 기준'
        with self.assertRaises(ValueError): review.create_review_files(self.folder, value, refs)
        self.assertEqual(before, Path(paths['comparison_csv']).read_bytes())

    def test_reference_sha_mismatch_is_rejected_before_creating_review(self):
        refs = references(); refs['samples'][0]['source_sha256'] = 'b'*64
        with self.assertRaises(ValueError): review.create_review_files(self.folder, result(), refs)
        self.assertFalse((self.folder/'comparison.csv').exists())

    def test_readonly_answer_and_reference_edits_invalidate_even_positive_flags(self):
        value = result(); paths = review.create_review_files(self.folder, value, references()); fill(self.folder)
        headers, rows = csv_rows(paths['comparison_csv'])
        rows[0]['기준내용초안'] = '사용자가 변경한 기준'; save_csv(paths['comparison_csv'], headers, rows)
        assessment = review.assess_review_files(self.folder, value, '검토자')
        self.assertEqual(assessment['status'], 'invalid_review')
        self.assertFalse(assessment['case_accepted'])

    def test_readonly_rejection_change_cannot_clear_a_rejection(self):
        value = result(rejected=True); paths = review.create_review_files(self.folder, value)
        fill(self.folder, disposition='제외 적절')
        headers, rows = csv_rows(paths['rejection_review_csv']); rows[0]['거절사유'] = '원문과 일치'
        save_csv(paths['rejection_review_csv'], headers, rows)
        self.assertEqual(review.assess_review_files(self.folder, value, '검토자')['status'], 'invalid_review')

    def test_sorting_review_rows_keeps_identity_but_duplicates_do_not(self):
        value = result(); paths = review.create_review_files(self.folder, value); fill(self.folder)
        headers, rows = csv_rows(paths['comparison_csv']); rows.reverse()
        save_csv(paths['comparison_csv'], headers, rows)
        self.assertTrue(review.assess_review_files(self.folder, value, '검토자')['case_accepted'])
        rows[-1] = copy.deepcopy(rows[0]); save_csv(paths['comparison_csv'], headers, rows)
        self.assertEqual(review.assess_review_files(self.folder, value, '검토자')['status'], 'invalid_review')

    def test_any_negative_field_judgement_blocks_acceptance(self):
        value = result(); paths = review.create_review_files(self.folder, value); fill(self.folder)
        headers, rows = csv_rows(paths['comparison_csv']); rows[0]['누락없음'] = '실패'
        save_csv(paths['comparison_csv'], headers, rows)
        assessment = review.assess_review_files(self.folder, value, '검토자')
        self.assertEqual(assessment['status'], 'review_failed')
        self.assertFalse(assessment['case_accepted'])

    def test_usage_time_and_latency_do_not_invalidate_same_content_review(self):
        value = result(); before = review.answer_fingerprint(value)
        value.update(usage={'input_tokens': 999}, elapsed_seconds=1000, created_at_utc='tomorrow')
        value['batch_outcomes'][0]['usage'] = {'input_tokens': 1000}
        self.assertEqual(before, review.answer_fingerprint(value))

    def test_mock_origin_never_accepts_even_if_human_flags_positive(self):
        value = result(live=False); review.create_review_files(self.folder, value); fill(self.folder)
        assessment = review.assess_review_files(self.folder, value, '검토자')
        self.assertFalse(assessment['case_accepted'])
        self.assertIn('live_model_origin_not_confirmed', assessment['technical_blockers'])

    def test_empty_or_all_rejected_output_never_accepts_and_hides_unverified_text(self):
        value = result(rejected=True); value['claims'] = []
        value['fields'][4]['claims'][0]['citations'][0]['provenance_verified'] = False
        paths = review.create_review_files(self.folder, value); fill(self.folder, disposition='제외 적절')
        _, rows = csv_rows(paths['comparison_csv'])
        self.assertNotIn('사업예산은 100원', rows[4]['실제요약'])
        assessment = review.assess_review_files(self.folder, value, '검토자')
        self.assertFalse(assessment['case_accepted'])
        self.assertIn('validated_answer_claims_missing', assessment['technical_blockers'])

    def test_incomplete_coverage_duplicate_ids_and_runtime_failure_block(self):
        mutations = [lambda r:r.update(input_received_chunk_ids=['c1']),
                     lambda r:r.update(checked_chunk_ids=['c1']),
                     lambda r:r.update(checked_chunk_coverage=True),
                     lambda r:r.update(source_chunk_ids=['c1','c1']),
                     lambda r:r['batch_outcomes'][0].update(status='api_failed'),
                     lambda r:r['fields'][0].update(state='not_processed')]
        for i, change in enumerate(mutations):
            value = result(); change(value); folder = self.folder/str(i)
            review.create_review_files(folder, value); fill(folder)
            with self.subTest(i=i): self.assertFalse(review.assess_review_files(folder, value, '검토자')['case_accepted'])

    def test_missing_or_tampered_binding_never_accepts(self):
        value = result(); paths = review.create_review_files(self.folder, value); fill(self.folder)
        binding = json.loads(Path(paths['binding_json']).read_text()); binding['reference_snapshot'] = [{'field_id':'budget_vat','expected':'변조'}]
        Path(paths['binding_json']).write_text(json.dumps(binding))
        self.assertEqual(review.assess_review_files(self.folder, value, '검토자')['status'], 'invalid_review')
        Path(paths['binding_json']).unlink()
        self.assertFalse(review.assess_review_files(self.folder, value, '검토자')['case_accepted'])

    def test_input_result_and_reference_are_not_mutated(self):
        value = result(rejected=True); refs = references(); before = copy.deepcopy((value, refs))
        review.create_review_files(self.folder, value, refs)
        review.assess_review_files(self.folder, value, '검토자')
        self.assertEqual(before, (value, refs))

    def test_claim_citation_must_belong_to_declared_document_chunks(self):
        value = result(); value['claims'][0]['citations'][0]['chunk_id'] = 'outside'
        review.create_review_files(self.folder, value); fill(self.folder)
        assessment = review.assess_review_files(self.folder, value, '검토자')
        self.assertFalse(assessment['case_accepted'])
        self.assertIn('citation_outside_document_chunks', assessment['technical_blockers'])

    def test_unknown_batch_status_cannot_pass_as_completed(self):
        value = result(); value['batch_outcomes'][0]['status'] = 'mystery'
        review.create_review_files(self.folder, value); fill(self.folder)
        self.assertFalse(review.assess_review_files(self.folder, value, '검토자')['case_accepted'])

    def test_field_uncertainty_is_visible_and_labelled_as_model_note(self):
        value = result()
        value['fields'][4]['notes'] = ['부분 입력 B0001에서 미확인: VAT 별도 적용 범위']
        paths = review.create_review_files(self.folder, value)
        _, rows = csv_rows(paths['comparison_csv'])
        self.assertIn('사업예산은 100원', rows[4]['실제요약'])
        self.assertIn('추가 확인(모델 표시):', rows[4]['실제요약'])
        self.assertIn('VAT 별도 적용 범위', rows[4]['실제요약'])
        self.assertNotIn('VAT 별도 적용 범위', rows[4]['검증된원문인용'])

    def test_uncertainty_stays_visible_without_accepted_field_claim(self):
        value = result()
        value['fields'][8]['notes'] = ['부분 입력 B0002에서 미확인: 제출기한의 구체적인 날짜']
        paths = review.create_review_files(self.folder, value)
        _, rows = csv_rows(paths['comparison_csv'])
        self.assertIn('검증된 주장을 표시할 수 없습니다.', rows[8]['실제요약'])
        self.assertIn('추가 확인(모델 표시):', rows[8]['실제요약'])
        self.assertIn('제출기한의 구체적인 날짜', rows[8]['실제요약'])
        self.assertEqual(rows[8]['검증된원문인용'], '')


if __name__ == '__main__': unittest.main()
