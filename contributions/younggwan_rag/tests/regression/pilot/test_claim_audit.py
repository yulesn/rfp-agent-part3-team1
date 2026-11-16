"""Independent requirement names expose citation gaps without semantic auto-pass."""
import copy
import json
import unittest

from regression.conversation.test_focus import GroundedClient
from rfp_rag.pilot.claim_audit import claim_citation_audit, claim_review_summary


def example():
    return {'execution_status': 'completed', 'validation': {'valid': True},
        'source_chunks': [{'doc_id': 'A', 'text':
            '요구사항 목록\nSFR-901 예약 취소\nSFR-902 승인 통보\nDAT-701 보관 및 폐기 정책'},
            {'doc_id': 'B', 'text': 'SFR-801 지도 편집'}],
        'claims': [{'doc_id': 'A', 'text': '예약 취소와 승인 통보 기능을 구축한다.',
                    'citations': [{'chunk_id': 'a1', 'quote': 'SFR-901 예약 취소'}]},
                   {'doc_id': 'A', 'text': '승인 통보 기능을 구축한다.',
                    'citations': [{'chunk_id': 'a2', 'quote': 'SFR-902 승인 통보'}]}]}


class NamedItemClient(GroundedClient):
    """Deliberately attach scope prose to a different inventory's full list."""
    def create(self, **body):
        response = super().create(**body)
        evidence = json.loads(body['input'][1]['content'])['evidence']
        inventory = next((c for c in evidence if 'FUN-장비-101' in c['text']), None)
        if inventory:
            payload = json.loads(response['output_text'])
            payload['claims'][0]['text'] = inventory['text']
            response['output_text'] = json.dumps(payload)
        return response


class ClaimAuditTests(unittest.TestCase):
    def test_another_claims_citation_never_fills_the_current_claims_gap(self):
        audit = claim_citation_audit(example())
        self.assertEqual(audit['flagged_claim_indices'], [0])
        self.assertEqual(audit['rows'][0]['missing_literal_anchors'],
            [{'requirement_id': 'SFR-902', 'label': '승인 통보'}])
        self.assertEqual(audit['rows'][1]['signal'], 'literal_anchors_found')

    def test_all_names_may_be_present_across_the_same_claims_own_citations(self):
        result = example()
        result['claims'][0]['citations'].append({'chunk_id': 'a2', 'quote': '승인 통보 기능'})
        audit = claim_citation_audit(result)
        self.assertEqual(audit['flagged_claim_indices'], [])
        self.assertEqual(audit['assessable_claim_count'], 2)
        self.assertFalse(audit['semantic_correctness_verified'])
        self.assertFalse(audit['automatic_semantic_pass'])

    def test_negation_is_not_mistaken_for_a_semantic_pass(self):
        result = example()
        result['claims'] = [{'doc_id': 'A', 'text': '예약 취소 기능을 제공하지 않는다.',
            'citations': [{'quote': '예약 취소 기능을 제공한다.'}]}]
        audit = claim_citation_audit(result)
        self.assertEqual(audit['flagged_claim_indices'], [])
        self.assertFalse(audit['semantic_correctness_verified'])
        self.assertFalse(audit['automatic_semantic_pass'])

    def test_omitted_heading_can_flag_a_supported_description_without_rejecting_it(self):
        result = example()
        result['source_chunks'] = [{'doc_id': 'A', 'text':
            'SFR-901 예약 취소\n사용자가 신청을 철회하도록 구현한다.'}]
        result['claims'] = [{'doc_id': 'A', 'text': '예약 취소는 신청 철회를 지원한다.',
            'citations': [{'quote': '사용자가 신청을 철회하도록 구현한다.'}]}]
        before = copy.deepcopy(result)
        audit = claim_citation_audit(result)
        self.assertEqual(audit['flagged_claim_indices'], [0])
        self.assertEqual(result, before)
        self.assertTrue(result['validation']['valid'])
        self.assertFalse(audit['automatic_semantic_pass'])

    def test_paraphrases_and_construction_scopes_remain_unassessed(self):
        for text in ('사용자가 신청을 철회하도록 구현한다.', '보행로 재포장 380m를 수행한다.'):
            result = example()
            result['claims'][0]['text'] = text
            audit = claim_citation_audit(result)
            self.assertEqual(audit['rows'][0]['signal'], 'not_assessable')
            self.assertFalse(audit['automatic_semantic_pass'])

    def test_catalog_names_are_not_borrowed_from_other_documents(self):
        result = example()
        result['claims'] = [{'doc_id': 'B', 'text': '예약 취소를 제공한다.', 'citations': [{'quote': '지도 편집'}]}]
        audit = claim_citation_audit(result)
        self.assertEqual(audit['assessable_claim_count'], 0)
        self.assertEqual(audit['rows'][0]['signal'], 'not_assessable')

    def test_heading_lines_and_spaced_ids_are_supported(self):
        result = example()
        result['source_chunks'] = [{'doc_id': 'A', 'text':
            'DAT - 보관 - 701\n데이터 필수\n보관 및 폐기 정책\n정의\n보관 기간을 설정한다.'}]
        result['claims'] = [{'doc_id': 'A', 'text': '보관·폐기 정책을 구현한다.',
            'citations': [{'quote': 'DAT- 보관- 701'}]}]
        audit = claim_citation_audit(result)
        self.assertEqual(audit['assessable_claim_count'], 1)
        self.assertEqual(audit['rows'][0]['named_requirements'][0]['requirement_id'], 'DAT-보관-701')
        self.assertEqual(audit['flagged_claim_indices'], [])

    def test_entities_and_conjunctions_are_only_literal_normalization(self):
        result = example()
        result['claims'] = [{'doc_id': 'A', 'text': '보관 및 폐기 정책을 구현한다.',
            'citations': [{'quote': '보관 &amp; 폐기 정책'}]}]
        audit = claim_citation_audit(result)
        self.assertEqual(audit['rows'][0]['signal'], 'literal_anchors_found')
        self.assertFalse(audit['semantic_correctness_verified'])

    def test_rejected_or_disabled_claims_never_become_reviewable_text(self):
        for status, valid in [('validation_failed', False), ('disabled', True), ('completed', False)]:
            result = example()
            result.update(execution_status=status, validation={'valid': valid})
            result['claims'][0]['text'] = 'PRIVATE_REJECTED_MODEL_TEXT'
            audit = claim_citation_audit(result)
            self.assertEqual(audit['rows'], [])
            self.assertNotIn('PRIVATE', json.dumps(audit))
            self.assertFalse(audit['automatic_semantic_pass'])

    def test_long_prose_and_numeric_rows_are_not_invented_requirement_names(self):
        result = example()
        result['source_chunks'] = [{'doc_id': 'A', 'text':
            'SFR-901 ' + '조건을 설명하는 장문의 본문입니다. ' * 15 + '\nSFR-902 123456'}]
        self.assertEqual(claim_citation_audit(result)['assessable_claim_count'], 0)

    def test_the_audit_never_modifies_claims_quotes_or_source_chunks(self):
        result = example()
        before = copy.deepcopy(result)
        claim_citation_audit(result)
        self.assertEqual(result, before)

    def test_empty_or_unmatched_sources_do_not_produce_an_approval(self):
        result = example()
        result['source_chunks'] = []
        audit = claim_citation_audit(result)
        self.assertEqual(audit['claim_count'], 2)
        self.assertEqual(audit['assessable_claim_count'], 0)
        self.assertFalse(audit['automatic_semantic_pass'])

    def test_review_summary_counts_signals_without_filling_human_judgements(self):
        audit = claim_citation_audit(example())
        summary = claim_review_summary([audit, claim_citation_audit({})])
        self.assertEqual(summary['turns_with_literal_anchor_gaps'], 1)
        self.assertEqual(summary['claims_with_literal_anchor_gaps'], 1)
        self.assertEqual(summary['assessable_claims'], 2)
        self.assertFalse(summary['automatic_semantic_pass'])
        self.assertNotIn('accuracy', summary)


if __name__ == '__main__':
    unittest.main()
