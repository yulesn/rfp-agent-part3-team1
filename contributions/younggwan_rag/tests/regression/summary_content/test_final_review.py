"""Regressions found by reading the final uploaded summaries, not just logs."""
import unittest

from rfp_rag.summary_content.conditions import recover_conditions
from rfp_rag.summary_content.pipeline import rebuild_document_summary
from regression.summary_content.test_integration import source_fixture, saved_result, make_claim, field_text


class FinalReviewTests(unittest.TestCase):
    def rebuild(self,lines,fields=()):
        doc,chunks,registry=source_fixture(lines)
        claims=[make_claim(registry,chunks,fid,i) for fid,i in fields]
        raw=saved_result(doc,chunks,claims)
        return rebuild_document_summary(raw,doc,chunks,registry)

    def test_system_narajangteo_function_is_not_bid_route(self):
        result=self.rebuild(['Ⅱ. 시스템 요구사항','입찰 기능 개선',
            '- 나라장터를 통한 입력 편의화 고려',
            'Ⅲ. 제안서 제출 방법','입찰서류 제출 방법은 입찰유의서 참조'])
        text=field_text(result,'submission_method')
        self.assertNotIn('입력 편의화',text)
        self.assertIn('입찰유의서',text)

    def test_unfilled_party_labels_are_not_organization(self):
        result=self.rebuild(['발주기관','계약상대자'])
        self.assertEqual(field_text(result,'issuing_agency'),'')

    def test_wrapped_sw_qualification_and_explicit_prohibition_are_recovered(self):
        result=self.rebuild(['Ⅱ. 입찰참가자격',
            '◦ 소프트웨어사업자[컴퓨터 관련\n서비스 사업] 신고를 필한 자',
            '◦ 대기업 및 중견기업은 본 입찰에 참여할 수 없음'])
        text=field_text(result,'eligibility')
        self.assertIn('신고를 필한 자',text)
        self.assertIn('참여할 수 없음',text)

    def test_mentions_of_effect_in_evaluation_do_not_open_purpose_section(self):
        result=self.rebuild(['가. 과업목적','ㅇ 서비스 접근성 개선 및 업무 효율 제고',
            '나. 평가항목','- 정성·정량적 기대효과 분석방안의 적정성',
            '- 사업관리자의 인력 배치계획을 평가'])
        text=field_text(result,'purpose')
        self.assertIn('서비스 접근성',text)
        self.assertNotIn('배치계획',text)
        self.assertNotIn('분석방안의 적정성',text)

    def test_negotiation_time_is_not_execution_period(self):
        result=self.rebuild(['사업기간 : 계약체결 후 9개월',
            '협상기간은 대상자 선정 후 12일 이내로 함(필요시 조정 가능)'])
        self.assertIn('9개월',field_text(result,'duration'))
        self.assertNotIn('12일',field_text(result,'duration'))

    def test_batch_absence_variant_does_not_survive_as_fact(self):
        result=self.rebuild(['제출기한은 이 제공 근거에는 명시되어 있지 않다.'],
            [('submission_deadline',0)])
        self.assertEqual(field_text(result,'submission_deadline'),'')

    def test_quantity_is_estimated_and_may_change(self):
        result=self.rebuild(['예정인원 : 3,700명 이내',
            '※ 계약시점에 예정인원 변동 가능'])
        text=field_text(result,'scope_requirements')
        self.assertIn('3,700',text)
        self.assertIn('변동 가능',text)

    def test_incomplete_negated_score_is_not_shown_beside_complete_threshold(self):
        result=self.rebuild(['기술능력 평가결과 71점 이상을 득점하지 못하는 경우는',
            '“협상적격자”에서 제외한다.'])
        rows=next(f for f in result['fields'] if f['field_id']=='evaluation')['claims']
        self.assertTrue(any('제외한다' in c['text'] for c in rows))
        self.assertFalse(any(c['text'].rstrip().endswith('경우는') for c in rows))

    def test_monitoring_and_pre_presentation_materials_are_preserved(self):
        result=self.rebuild(['사용량을 실시간으로 모니터링하고 기간별 통계를 제공한다.',
            'Ⅴ. 제안서 평가','PPT로 발표하며 발표자료는 사전 제출한다.'])
        self.assertIn('모니터링',field_text(result,'scope_requirements'))
        self.assertIn('사전 제출',field_text(result,'submission_documents'))

    def test_blank_destruction_certificate_does_not_assert_completed_work(self):
        result=self.rebuild(['개인정보 파기 확인서',
            '상기와 같이 개인정보가 안전하게 파기되었음을 확인하였으며 책임을 인정합니다.',
            '과업 완료 후 개인정보를 안전하게 파기하여야 한다.'],
            [('scope_requirements',0),('scope_requirements',1),('scope_requirements',2)])
        text=field_text(result,'scope_requirements')
        self.assertNotIn('확인하였',text)
        self.assertNotIn('파기 확인서',text)
        self.assertIn('파기하여야 한다',text)


if __name__=='__main__':unittest.main()
