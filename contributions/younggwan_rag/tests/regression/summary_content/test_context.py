import copy
import unittest

from rfp_rag.summary_content.context import DocumentContext, classify_claim_context, identify_project_candidates


def document(*paragraphs):
    return {'doc_id':'synthetic-document','sha256':'a'*64,
            'blocks':[{'block_id':str(i),'locator':f'paragraph {i}', 'text':text}
                      for i,text in enumerate(paragraphs)]}


def claim(doc, block_id, text=None):
    block=next(b for b in doc['blocks'] if b['block_id']==str(block_id))
    quote=block['text'] if text is None else text
    left=block['text'].index(quote)
    return {'text':'분류할 사실', 'citations':[{'quote':quote,'source_ranges':[
        {'collection':'blocks','block_id':block['block_id'],
         'char_start':left,'char_end':left+len(quote)}]}]}


class ContextTests(unittest.TestCase):
    def test_same_email_has_occurrence_specific_event(self):
        doc=document('1. 사업명 : 지역 데이터 통합 설계 용역',
            '□ [참고] 용역 사전설명회 계획 및 참가신청 안내',
            'ㅇ (참가신청)', '- 신청방법: 이메일신청(team@example.test)',
            '2. 서비스 현황', '운영 중인 관광정보 채널 목록',
            '6. 제안서 제출 방법 및 유의사항',
            '전자우편 : team@example.test 로 제출')
        before=copy.deepcopy(doc)
        first=classify_claim_context(doc,claim(doc,3))
        second=classify_claim_context(doc,claim(doc,7))
        self.assertEqual(first['event_scope'],'briefing')
        self.assertEqual(second['event_scope'],'tender')
        self.assertTrue(second['trusted_for_tender'])
        self.assertEqual(doc,before)

    def test_postcontract_documents_do_not_become_tender(self):
        doc=document('10. 착수 및 준공 관련 제출문서',
            '1) 계약 상대자는 착수 및 준공 시 다음의 서류를 문서로 제출한다.',
            '가. 착수 시 (계약 시작일로부터 7일 이내 제출)',
            '- 공정예정표, 보안각서, 현장대리인 선임계')
        c=classify_claim_context(doc,claim(doc,3))
        self.assertEqual(c['event_scope'],'execution')
        self.assertFalse(c['trusted_for_tender'])

    def test_other_project_appendix_is_not_main_project(self):
        doc=document('1. 사업명 : 공공 기록관리 자동화 체계 구축',
            '2. 사업기간 : 계약일부터 365일',
            '[별지 제5호서식] 소프트웨어사업 영향평가 검토결과서',
            '1. 기본정보','사업명','(또는 서비스명)',
            '지역 모바일 주차 앱 개편','사업기간','2020년 4월 ~ 2020년 9월')
        context=DocumentContext(doc)
        self.assertEqual(context.primary_project_candidates[0]['value'],'공공 기록관리 자동화 체계 구축')
        c=context.classify_claim(claim(doc,8))
        self.assertEqual(c['evidence_role'],'other_project_candidate')
        self.assertIn('different_project_name_in_form',c['warnings'])

    def test_subcontract_processing_days_are_not_project_duration(self):
        doc=document('<서식 4>','소프트웨어사업','[ ] 하도급','계약 승인신청서',
            '[ ] 재하도급','처리기간','10일')
        c=classify_claim_context(doc,claim(doc,6))
        self.assertIn(c['evidence_role'],{'source_form','agreement_clause'})
        self.assertFalse(c['trusted_for_tender'])

    def test_annex_impact_form_detects_another_project(self):
        doc=document('1. 사업명 : 항만 시설관리 통합 시스템 구축',
            '[붙임] 소프트웨어사업 영향평가 검토결과서',
            '사업명(또는서비스명)','지역 문화행사 모바일 앱 개선',
            '사업기간','2018년 1월 ~ 2018년 8월')
        c=classify_claim_context(doc,claim(doc,5))
        self.assertEqual(c['evidence_role'],'other_project_candidate')

    def test_same_project_annex_does_not_claim_different_project(self):
        doc=document('1. 사업명 : 항만 시설관리 통합 시스템 구축',
            '[붙임] 소프트웨어사업 영향평가 검토결과서',
            '사업명(또는서비스명)','항만 시설관리 통합 시스템 구축',
            '사업기간','2026년 1월 ~ 2026년 8월')
        c=classify_claim_context(doc,claim(doc,5))
        self.assertEqual(c['evidence_role'],'source_form')
        self.assertNotIn('different_project_name_in_form',c['warnings'])

    def test_new_business_section_ends_previous_form_role(self):
        doc=document('<서식 1>','행사 참가신청서','성명','1. 사업개요',
            '□ 입찰 참가자격','소프트웨어사업자 신고를 필한 업체')
        c=classify_claim_context(doc,claim(doc,5))
        self.assertEqual(c['evidence_role'],'project_fact')
        self.assertEqual(c['event_scope'],'tender')

    def test_agreement_name_in_tender_checklist_does_not_leak(self):
        doc=document('4. 입찰참가 등록신청 구비서류','청렴계약·인권경영 이행각서',
            '사업자등록증 사본','법인인감증명서','6. 제안서 제출 방법 및 유의사항',
            '제안서는 이메일로 제출한다.')
        c=classify_claim_context(doc,claim(doc,5))
        self.assertEqual(c['evidence_role'],'project_fact')
        self.assertEqual(c['event_scope'],'tender')

    def test_explicit_tender_notice_after_briefing_is_not_briefing_date(self):
        doc=document('□ 사전설명회 참가신청 안내','신청기간 : 5월 7일',
            '제안서 제출기한은 입찰공고문 참조')
        c=classify_claim_context(doc,claim(doc,2))
        self.assertEqual(c['event_scope'],'tender')

    def test_form_references_in_checklist_are_not_form_starts(self):
        doc=document('4. 제안서 제출 서류','<서식 18,19호>',
            '5. 협상적격자 선정','기술점수가 배점의 83% 이상인 업체를 선정한다.')
        c=classify_claim_context(doc,claim(doc,3))
        self.assertNotEqual(c['evidence_role'],'source_form')
        doc['blocks'][1]['text']='<서식 18호>'
        doc['blocks'][1]['table_id']='source-checklist'
        c=classify_claim_context(doc,claim(doc,3))
        self.assertNotEqual(c['evidence_role'],'source_form')

    def test_example_table_does_not_consume_next_binding_bullet(self):
        doc=document('< 자원 제공 수량 제안 예시 >','※ 단순 예시이며 실제와 다름',
            '가속기 종류','수량','100장',
            'o 공급자는 시간당 정산단가를 제안하여야 한다.',
            '* 자원 사용시간은 2시간 단위로 책정한다.')
        c=classify_claim_context(doc,claim(doc,6))
        self.assertNotEqual(c['evidence_role'],'example')

    def test_agreement_purpose_is_identified(self):
        doc=document('[별지 6]','개인(신용)정보처리 업무위탁 약정서',
            '제1조(목적)','본 약정은 위탁자가 개인정보 처리를 위탁할 때 필요한 사항을 규정한다.')
        c=classify_claim_context(doc,claim(doc,3))
        self.assertEqual(c['evidence_role'],'agreement_clause')
        self.assertFalse(c['trusted_for_tender'])

    def test_example_prices_are_classified_as_example(self):
        doc=document('<서식 제19호>','사업비 편성내역 요약서(예시)',
            '(단위:천원)','총 사업비','100%','부가가치세(면세)')
        c=classify_claim_context(doc,claim(doc,5))
        self.assertEqual(c['evidence_role'],'example')

    def test_pdf_page_multiple_sections_use_exact_line_location(self):
        text=('□ 용역 사전설명회 계획 및 참가신청 안내\n'
              '- 신청기한: 5월 7일 15시\n'
              '6. 제안서 제출 방법 및 유의사항\n'
              '제안서 제출기한은 입찰공고에 따른다.\n')
        doc=document(text)
        context=DocumentContext(doc)
        first=context.classify_claim(claim(doc,0,'- 신청기한: 5월 7일 15시'))
        second=context.classify_claim(claim(doc,0,'제안서 제출기한은 입찰공고에 따른다.'))
        self.assertEqual(first['event_scope'],'briefing')
        self.assertEqual(second['event_scope'],'tender')
        for resolved in second['context_source_ranges']:
            self.assertEqual(text[resolved['char_start']:resolved['char_end']],resolved['text'])

    def test_neutral_major_heading_stops_briefing_inheritance(self):
        doc=document('□ 용역 사전설명회 계획 및 참가신청 안내',
                     '신청기한 : 5월 7일', '3. 서비스 현황', '관광 채널과 상담 서비스')
        c=classify_claim_context(doc,claim(doc,3))
        self.assertEqual(c['event_scope'],'unknown')

    def test_invalid_or_absent_ranges_abstain(self):
        doc=document('제안서 제출 방법')
        bad={'citations':[{'quote':'제안서','source_ranges':[
            {'collection':'blocks','block_id':'0','char_start':-1,'char_end':5}]}]}
        self.assertEqual(classify_claim_context(doc,bad)['certainty'],'unknown')
        self.assertIn('source_range_not_resolved',classify_claim_context(doc,bad)['warnings'])
        self.assertEqual(classify_claim_context(doc,{})['event_scope'],'unknown')

    def test_identity_does_not_come_from_metadata(self):
        doc=document('목표 : 연구환경을 안정적으로 제공한다.')
        doc['file_name']='기관_2026_새사업_제안요청서.pdf'
        self.assertEqual(identify_project_candidates(doc),[])

    def test_multi_scope_composite_claim_is_not_certified_tender(self):
        doc=document('□ 용역 사전설명회 계획 및 참가신청 안내','신청기한 : 5월 7일',
                     '6. 제안서 제출 방법 및 유의사항','제출기한 : 입찰공고문 참조')
        a=claim(doc,1);b=claim(doc,3)
        a['citations']+=b['citations']
        c=classify_claim_context(doc,a)
        self.assertEqual(c['event_scope'],'unknown')
        self.assertIn('mixed_event_scopes',c['warnings'])
        self.assertFalse(c['trusted_for_tender'])


if __name__=='__main__':
    unittest.main()
