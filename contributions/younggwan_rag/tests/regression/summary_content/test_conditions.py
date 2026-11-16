"""Counterexamples for local source supplementation; no model or network."""
import copy
import unittest
from unittest.mock import patch

from rfp_rag.generation.evidence import EvidenceRegistry
from rfp_rag.summary_content.conditions import recover_conditions


def corpus(texts, *, page_mode=False):
    blocks = []
    chunks = []
    for index,text in enumerate(texts):
        block = {'block_id':f'b{index}', 'locator':f'PDF p.{index+1}' if page_mode else f'BodyText/Section0 / paragraph {index+1}',
                 'text':text}
        if page_mode:
            block['page']=index+1
        else:
            block['section']='BodyText/Section0'
        blocks.append(block)
        if not text.strip():
            continue
        span = {'collection':'blocks', 'block_id':block['block_id'],
            'char_start':0,'char_end':len(text),'chunk_start':0,'chunk_end':len(text),
            'locator':block['locator']}
        for key in ('page','section'):
            if key in block:
                span[key]=block[key]
        chunk = {'chunk_id':f'c{index}','doc_id':'document-test','canonical_doc_id':'document-test',
            'sha256':'a'*64,'file_name':'synthetic.hwp','text':text,
            'length_unit':'utf8_bytes','text_bytes':len(text.encode()),'kind':'prose',
            'source_method':'raw_parser','locator':block['locator'],'source_spans':[span]}
        for key in ('page','section'):
            if key in block:
                chunk[key]=block[key]
        chunks.append(chunk)
    document={'doc_id':'document-test','canonical_doc_id':'document-test','sha256':'a'*64,
        'file_name':'synthetic.hwp','blocks':blocks,'supplemental_blocks':[]}
    return document,chunks,EvidenceRegistry([document],chunks)


class ConditionsTests(unittest.TestCase):
    def recover(self, texts, **kwargs):
        doc,chunks,reg=corpus(texts,**kwargs)
        with patch('socket.socket.connect',side_effect=AssertionError('network prohibited')):
            return recover_conditions(doc,chunks,reg)

    def test_budget_parts_and_post_completion_duties_preserve_verbatim_numbers(self):
        texts=['□ 사업예산 : 912백만원 (부가세포함)',
               'ㅇ 개발예산 (612백만원) 및 HW구입(300백만원)',
               '□ 과업기간 : 계약체결일로부터 7개월(안정화 2개월 포함)',
               '※ 과업기간 이후에도 부당공제 후속절차 지원은 포함한다.']
        rows=self.recover(texts)
        self.assertTrue(any(r['field_id']=='budget_vat' and texts[1] in r['text'] for r in rows))
        self.assertTrue(any(r['field_id']=='duration' and texts[3] in r['text'] for r in rows))
        self.assertTrue(all(r['origin']=='local_source_extraction' for r in rows))
        self.assertTrue(all(r['semantic_quality_verified'] is False for r in rows))

    def test_source_ranges_on_pdf_page_are_line_exact_and_repeat_safe(self):
        line='□ 사업예산 : 137,500원 (부가가치세 제외)'
        page='I. 사업개요\n'+line+'\n상세 안내\n'+line
        rows=self.recover([page],page_mode=True)
        budget=[r for r in rows if r['field_id']=='budget_vat']
        self.assertEqual(len(budget),1)  # Repeated identical field text deduplicates.
        q=budget[0]['citations'][0]
        self.assertEqual(q['quote'],line)
        self.assertEqual(q['quote_start'],page.index(line))
        self.assertEqual(q['source_ranges'][0]['char_start'],page.index(line))
        self.assertEqual(page[q['quote_start']:q['quote_end']],line)

    def test_briefing_email_and_application_date_are_not_tender_routes(self):
        rows=self.recover(['□ 용역 사전설명회 계획 및 참가신청 안내',
            'ㅇ (참가신청)', '- 신청기간: ~ 7월 8일 15시',
            '- 신청방법: 이메일 신청(event@example.test)',
            '□ 제안서 제출 방법', '제안서는 전자우편 또는 우편 제출한다.'])
        tender=[r for r in rows if r['field_id'].startswith('submission')]
        self.assertFalse(any('event@example.test' in r['text'] for r in tender))
        self.assertFalse(any('7월 8일' in r['text'] for r in tender))

    def test_post_contract_report_email_is_not_tender_route(self):
        rows=self.recover(['□ 착수 및 준공 관련 제출문서',
            '제안서에 따라 착수계는 계약 시작 후 9일 이내 제출한다.',
            '결과보고서 등 산출물은 이메일 또는 USB로 제출한다.'])
        self.assertFalse(any(r['field_id'].startswith('submission') for r in rows))

    def test_external_notice_is_kept_without_inventing_a_deadline(self):
        rows=self.recover(['□ 입찰 및 제안서 제출',
            '제출서류 목록, 일시 및 장소 : 입찰공고문 참조'])
        deadlines=[r for r in rows if r['field_id']=='submission_deadline']
        self.assertTrue(deadlines)
        self.assertTrue(all(r['mode']=='notice_reference' for r in deadlines))
        self.assertFalse(any('202' in r['text'] for r in deadlines))

    def test_new_tender_heading_ends_prior_briefing_context(self):
        rows=self.recover(['□ 용역 사전설명회 계획 및 참가신청 안내',
            '- 신청방법: 이메일 신청(shared@example.test)',
            'Ⅱ. 입찰 안내', '제출기한은 입찰공고문에 명기된 기한에 따른다.',
            '제안서는 이메일(shared@example.test)로 제출한다.'])
        deadlines=[r for r in rows if r['field_id']=='submission_deadline']
        methods=[r for r in rows if r['field_id']=='submission_method']
        self.assertTrue(any('명기된 기한' in r['text'] for r in deadlines))
        self.assertTrue(any('제안서는 이메일' in r['text'] for r in methods))
        self.assertFalse(any('신청방법' in r['text'] for r in methods))

    def test_purpose_numbered_bullets_are_both_kept_and_scope_is_separate(self):
        rows=self.recover(['4. 과업목적',
            '1) 전문 교육 및 검토를 통한 업무의 효율적 추진',
            '2) 상담과 교육을 통한 신고오류 최소화 및 정확성 확보',
            '5. 과업범위',
            '전문 상담과 결과보고서 제공을 포함한 실무 지원을 수행한다.'])
        purpose=[r['text'] for r in rows if r['field_id']=='purpose']
        scope=[r['text'] for r in rows if r['field_id']=='scope_requirements']
        self.assertTrue(any('신고오류 최소화' in t for t in purpose))
        self.assertTrue(any('효율적 추진' in t for t in purpose))
        self.assertFalse(any('실무 지원을 수행' in t for t in purpose))
        self.assertTrue(any('실무 지원을 수행' in t for t in scope))

    def test_pdf_purpose_wrap_is_one_source_supported_statement(self):
        page='1 추진배경\nㅇ (추진목적) 노후 시스템 개편을 통해\n행정업무의 안정적인 운영기반을 마련한다.\nㅇ (기대효과) 서비스 만족도와 업무 효율성을 높인다.\n2 사업개요\nㅇ 사업예산 : 27,000원'
        rows=self.recover([page],page_mode=True)
        purpose=[r for r in rows if r['field_id']=='purpose']
        self.assertTrue(any('노후 시스템 개편을 통해\n행정업무의 안정적인' in r['text'] for r in purpose))
        self.assertFalse(any('27,000원' in r['text'] for r in purpose))

    def test_split_evaluation_table_cell_values_preserve_label_binding(self):
        rows=self.recover(['□ 평가 방법', 'o 평가 항목', '평가항목',
            '사업', '수행', '능력', '(7)', '재무건전성', 'o 재무건전성 평가',
            '전략 및 방법론', '(23)', '사업이해도'])
        table=[r for r in rows if r['extraction_rule']=='evaluation_table_label_and_score']
        self.assertTrue(any('사업\n수행\n능력\n(7)' in r['text'] for r in table))
        self.assertTrue(any('전략 및 방법론\n(23)' in r['text'] for r in table))

    def test_subcontract_approval_not_confused_with_tender_threshold(self):
        rows=self.recover(['□ 입찰 및 평가',
            '기술평가 점수가 배점한도의 87% 이상인 자를 협상적격자로 선정한다.',
            '하도급계약의 적정성은 평가점수가 89점 이상인 경우 승인한다.'])
        evaluation=[r for r in rows if r['field_id']=='evaluation']
        self.assertTrue(any('87%' in r['text'] for r in evaluation))
        self.assertFalse(any('89점' in r['text'] for r in evaluation))

    def test_actual_tender_subcontract_plan_requirement_is_recovered(self):
        rows=self.recover(['본 사업은 기술성 평가 시 하도급계획적정성을 평가항목에 포함하며 적정성 확인서를 제출해야 한다.'])
        self.assertTrue(any(r['field_id']=='evaluation' for r in rows))
        self.assertTrue(any(r['field_id']=='submission_documents' for r in rows))

    def test_empty_form_and_model_absence_are_not_budget_facts(self):
        rows=self.recover(['<서식 4> 하도급 금액 산출 명세서 부가세 포함 400원',
            '사업명', '                       ',
            '제공된 근거에서 사업예산 400원은 확인되지 않습니다.'])
        self.assertFalse(any(r['field_id']=='budget_vat' for r in rows))
        self.assertFalse(any(r['field_id']=='project_name' for r in rows))

    def test_irregular_price_is_never_reformatted(self):
        text='ㅇ 예정가격 : 금 1,234,000,0000원 (직접경비·부가세 포함)'
        rows=self.recover([text])
        self.assertTrue(any(text in r['text'] for r in rows))
        self.assertFalse(any('1,234,000,000원' in r['text'] for r in rows))

    def test_no_unknown_document_or_changed_chunk_is_accepted(self):
        doc,chunks,reg=corpus(['□ 사업예산 : 77,000원 (부가세 포함)'])
        chunks[0]['text']='□ 사업예산 : 99,000원 (부가세 포함)'
        with self.assertRaises(ValueError):
            recover_conditions(doc,chunks,reg)

    def test_split_source_across_chunks_keeps_both_literal_ranges(self):
        doc,chunks,_=corpus(['□ 사업예산 : 77,000원 (부가세 포함)'])
        original=chunks[0]
        mid=12
        separated=[]
        for number,(a,b) in enumerate([(0,mid),(mid,len(original['text']))]):
            c=copy.deepcopy(original);c['chunk_id']=f'split{number}'
            c['text']=original['text'][a:b];c['text_bytes']=len(c['text'].encode())
            c['source_spans'][0].update(char_start=a,char_end=b,chunk_start=0,chunk_end=b-a)
            separated.append(c)
        reg=EvidenceRegistry([doc],separated)
        rows=recover_conditions(doc,separated,reg)
        budget=next(r for r in rows if r['field_id']=='budget_vat')
        self.assertEqual(''.join(q['quote'] for q in budget['citations']),original['text'])
        self.assertEqual(len(budget['citations']),2)


if __name__=='__main__':
    unittest.main()
