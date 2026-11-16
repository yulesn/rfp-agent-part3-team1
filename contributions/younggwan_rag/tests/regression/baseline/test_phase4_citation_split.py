"""A joined noncontiguous quote becomes two exact quotes, never a fake substring."""
import copy
import unittest

from regression.baseline.test_phase4_evidence import block, document, chunk, answer
from rfp_rag.generation.evidence import EvidenceRegistry
from rfp_rag.generation.citation_recovery import validate_with_quote_recovery


HEADER='<부가세 포함, 금액단위 : 원>'
BODY='구       분\n금      액\n비     고\n용 역 비\n부가가치세 포함\n장  비\n구입비\nHW 구매 부문\n부가가치세 포함'
GAP='발주자\n한국주택금융공사\n직전도급자\n-\n하도급\n주는자\n사업명\n계약금액\n상  호\n대표자'


class CitationSplitTests(unittest.TestCase):
    def check(self,source,quote,*,did='d1',allowed=None):
        b=block(text=source);d=document(source=b);c=chunk(b)
        reg=EvidenceRegistry([d],[c]);payload=answer(quote=quote,did=did)
        original=copy.deepcopy(payload)
        before=reg.validate_answer(payload,[c],allowed)
        after=validate_with_quote_recovery(reg,payload,[c],allowed)
        self.assertEqual(payload,original)
        return before,after,d,c

    def test_reported_header_plus_table_body_is_two_exact_citations(self):
        before,r,d,c=self.check(HEADER+'\n'+GAP+'\n'+BODY+'\nSW 구매 부문\n부가가치세 포함',HEADER+'\n'+BODY)
        self.assertFalse(before['valid']);self.assertTrue(r['valid'])
        citations=r['claims'][0]['citations']
        self.assertEqual(len(citations),2)
        self.assertEqual([v['quote'] for v in citations],[HEADER,BODY])
        self.assertEqual(r['citation_recovery']['split_citations'],1)
        self.assertEqual(r['citation_recovery']['additional_api_calls'],0)
        for v in citations:
            self.assertIn(v['quote'],c['text'])
            for span in v['source_ranges']:
                self.assertEqual(d['blocks'][0]['text'][span['char_start']:span['char_end']],
                                 c['text'][span['chunk_start']:span['chunk_end']])
        self.assertNotIn(GAP,' '.join(v['quote'] for v in citations))

    def test_whitespace_inside_each_fragment_is_restored(self):
        _,r,_,_=self.check(HEADER+'\n'+GAP+'\n'+BODY,HEADER.replace(' ','')+'\n'+'\n'.join(s.replace(' ','') for s in BODY.splitlines()))
        self.assertTrue(r['valid']);self.assertEqual(r['claims'][0]['citations'][1]['quote'],BODY)

    def test_changed_numbers_vat_units_punctuation_and_words_are_rejected(self):
        source=HEADER+'\n'+GAP+'\n사업예산은 688백만원이며 부가세 포함입니다.'
        for tail in ['사업예산은 699백만원이며 부가세 포함입니다.','사업예산은 688백만원이며 부가세 별도입니다.',
                     '사업예산은 688달러이며 부가세 포함입니다.','사업예산은 688백만원이며 부가세 포함입니다!',
                     '사업예산은 688백만원으로 확정되었습니다.']:
            self.assertFalse(self.check(source,HEADER+'\n'+tail)[1]['valid'])

    def test_intra_line_omission_and_negation_removal_are_rejected(self):
        source=HEADER+'\n'+GAP+'\n사업예산은 688백만원이며 부가세 포함이 아닙니다.'
        r=self.check(source,HEADER+'\n사업예산은 688백만원이며 부가세 포함입니다.')[1]
        self.assertFalse(r['valid'])

    def test_reordered_fragments_are_rejected(self):
        self.assertFalse(self.check(HEADER+'\n'+GAP+'\n'+BODY,BODY+'\n'+HEADER)[1]['valid'])

    def test_duplicate_fragment_placements_are_ambiguous(self):
        source=HEADER+'\n'+GAP+'\n'+BODY+'\n여백\n'+BODY
        r=self.check(source,HEADER+'\n'+BODY)[1]
        self.assertFalse(r['valid'])
        self.assertEqual(r['citation_diagnostics'][0]['split_check'],'ambiguous_fragment_mapping')

    def test_several_omitted_blocks_become_separate_exact_citations(self):
        a='사업예산은 688백만원입니다.';b='사업기간은 계약일부터 5개월입니다.';c='안정화 기간은 1개월 포함입니다.'
        source=a+'\n생략 구간 하나\n'+b+'\n생략 구간 둘\n'+c
        r=self.check(source,a+'\n'+b+'\n'+c)[1]
        self.assertTrue(r['valid'])
        self.assertEqual([q['quote'] for q in r['claims'][0]['citations']],[a,b,c])
        self.assertEqual(len(r['citation_diagnostics'][0]['omitted_source_gaps']),2)
        self.assertFalse(r['semantic_entailment_verified'])

    def test_eight_fragments_are_bounded_and_ninth_is_rejected(self):
        pieces=[f'제출서류 {i}번의 원본과 사본을 모두 제출해야 합니다.' for i in range(9)]
        self.assertTrue(self.check('\n검토할 생략 조건\n'.join(pieces[:8]),'\n'.join(pieces[:8]))[1]['valid'])
        self.assertFalse(self.check('\n검토할 생략 조건\n'.join(pieces),'\n'.join(pieces))[1]['valid'])

    def test_intra_line_omission_stays_rejected_with_three_fragments(self):
        a='제안서 원본은 반드시 제출해야 합니다.'
        b='가격제안서는 별도로 밀봉한 경우에만 접수합니다.'
        c='제출서류는 원본대조필을 반드시 명기해야 합니다.'
        source=a+'\n누락구간 하나\n'+b+'\n누락구간 둘\n'+c
        self.assertFalse(self.check(source,a+'\n가격제안서는 접수합니다.\n'+c)[1]['valid'])

    def test_short_middle_fragment_cannot_bridge_long_blocks(self):
        a='사업예산은 650백만원으로 명시되어 있습니다.'
        c='사업기간은 계약일부터 10개월로 명시되어 있습니다.'
        self.assertFalse(self.check(a+'\n중간 조건 하나\n필수\n중간 조건 둘\n'+c,a+'\n필수\n'+c)[1]['valid'])

    def test_repeated_third_fragment_remains_ambiguous(self):
        a='원본 제안서는 반드시 한 부를 제출합니다.'
        b='발표자료는 반드시 열 부를 제출합니다.'
        c='가격제안서는 별도로 밀봉하여 제출합니다.'
        source=a+'\n원본 조건\n'+b+'\n평가 조건\n'+c+'\n반복 구간\n'+c
        self.assertFalse(self.check(source,a+'\n'+b+'\n'+c)[1]['valid'])

    def test_omitted_negation_is_visible_and_never_semantic_pass(self):
        a='제안서 제출 방식은 직접 방문 접수입니다.'
        b='참가 신청 서류는 원본을 제출해야 합니다.'
        c='가격입찰 서류는 별도로 제출해야 합니다.'
        source=a+'\n전자우편 제출은 허용되지 않습니다.\n'+b+'\n접수기한 이후 제출할 수 없습니다.\n'+c
        r=self.check(source,a+'\n'+b+'\n'+c)[1]
        self.assertTrue(r['valid'])
        diag=r['citation_diagnostics'][0]
        self.assertTrue(diag['requires_conditions_review'])
        self.assertIn('허용되지 않습니다',diag['omitted_source_gaps'][0]['preview'])
        self.assertFalse(r['semantic_entailment_verified'])

    def test_fragment_search_refuses_oversized_input(self):
        a='사업예산은 650백만원이며 부가세가 포함됩니다.'
        source=a+'\n'+'중복 아닌 설명 줄\n'*2050+'사업기간은 계약일부터 10개월입니다.'
        quote=a+'\n사업기간은 계약일부터 10개월입니다.'
        self.assertFalse(self.check(source,quote)[1]['valid'])

    def test_short_isolated_labels_do_not_enable_split(self):
        source='예산\n다른 문장\n부가세는 포함이며 예산은 688백만원입니다.'
        self.assertFalse(self.check(source,'예산\n부가세는 포함이며 예산은 688백만원입니다.')[1]['valid'])

    def test_scope_and_document_checks_still_apply(self):
        source=HEADER+'\n'+GAP+'\n'+BODY;quote=HEADER+'\n'+BODY
        self.assertFalse(self.check(source,quote,allowed=[])[1]['valid'])
        self.assertFalse(self.check(source,quote,did='other')[1]['valid'])

    def test_true_contiguous_quote_is_not_split(self):
        _,r,_,_=self.check(HEADER+'\n'+BODY,HEADER+'\n'+BODY)
        self.assertTrue(r['valid']);self.assertEqual(r['citation_recovery']['split_citations'],0)
        self.assertEqual(len(r['claims'][0]['citations']),1)

    def test_replaying_revalidated_output_keeps_original_usage(self):
        from regression.baseline import test_phase4_citation_recovery as fixtures
        from rfp_rag.generation.citation_recovery import replay_saved_result
        fixture=fixtures.ReplayTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        first=replay_saved_result(fixture.root,fixture.path,registry=fixture.reg)
        second=replay_saved_result(fixture.root,first['revalidated_result_path'],registry=fixture.reg)
        self.assertEqual(second['generation']['original_usage']['api_calls'],1)
        self.assertEqual(second['generation']['usage']['api_calls'],0)


    def test_reported_D02_form_list_is_three_exact_source_fragments(self):
        source='구분\n제출장소\n제출서류\n비고\n제안서\n제  출\n뷰티패션\n산업과\n(서소문2청사 18층)\n① 제안서 제출 공문 1부\n<서식 6호>\n② 사업수행능력평가서 1부\n- 제안사 일반현황 1부\n- 사업수행 조직 및 역할분담 체계 1부\n- 참여인력 현황 1부\n- 참여인력 이력사항 1부\n- 용역 수행실적(원본) 1부\n- 실적증명서 원본 각 1부\n- 제안사 경영상태 1부(신용평가등급확인서 등)\n-서약서 1부\n- 정량적 평가지표 자가진단표 1부\n<서식 7~16호>\n③ 사업내용제안서 및 제안서 요약본(발표자료) 각 11부 (원본 각1부, 평가용 각10부)\n※ 원 본  1부  : 표지에 제안기관명, 대표자명 기재 후 인감날인\n평가용 10부 : 표지에 제안기관명, 대표자명 기재하지 않고 빈칸 처리\n<서식\n17호>\n④ 사업내용제안서 및 발표자료 수록 USB 1개\n※ 제안서 원본(PDF), 발표자료(PPT 및 PDF) 수록\n-\n⑤ 가격제안서(산출내역서 포함) 1부\n※ 별도 밀봉 후 인감날인\n<서식 18,19호>'
        quote='① 제안서 제출 공문 1부\n<서식 6호>\n② 사업수행능력평가서 1부\n- 제안사 일반현황 1부\n- 사업수행 조직 및 역할분담 체계 1부\n- 참여인력 현황 1부\n- 참여인력 이력사항 1부\n- 용역 수행실적(원본) 1부\n- 실적증명서 원본 각 1부\n- 제안사 경영상태 1부(신용평가등급확인서 등)\n-서약서 1부\n- 정량적 평가지표 자가진단표 1부\n<서식 7~16호>\n③ 사업내용제안서 및 제안서 요약본(발표자료) 각 11부 (원본 각1부, 평가용 각10부)\n④ 사업내용제안서 및 발표자료 수록 USB 1개\n⑤ 가격제안서(산출내역서 포함) 1부'
        before,r,d,c=self.check(source,quote)
        self.assertFalse(before['valid'])
        self.assertTrue(r['valid'])
        self.assertEqual(r['citation_recovery']['additional_api_calls'],0)
        pieces=r['claims'][0]['citations']
        self.assertEqual(len(pieces),3)
        self.assertEqual(''.join(''.join(p['quote'].split()) for p in pieces),''.join(quote.split()))
        self.assertEqual(len(r['citation_diagnostics'][0]['omitted_source_gaps']),2)
        for p in pieces:
            self.assertIn(p['quote'],source)
            self.assertTrue(p['provenance_verified'])
        self.assertFalse(r['answer_correctness_verified'])
        for wrong in [quote.replace('각 11부','각 10부'),quote.replace('USB 1개','USB 2개'),quote.replace('1부','2부',1)]:
            self.assertFalse(self.check(source,wrong)[1]['valid'])


if __name__=='__main__':unittest.main()
