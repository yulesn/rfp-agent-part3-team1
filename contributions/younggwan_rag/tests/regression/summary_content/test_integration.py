"""Content regressions use original sources, not model scores or reference answers.

The portable tests invent a corpus.  The optional seven-document regression
uses saved, local artifacts only; it never requests fresh model responses.
Enable it with RFP_CONTENT_CORPUS_ROOT (contains data/) and RFP_CONTENT_RUN_ROOT
(contains documents/*/latest.json).  Neither path is sent to a network service.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from rfp_rag.generation.evidence import EvidenceRegistry
from rfp_rag.summary_fields.fields import FIELDS, FIELD_BY_ID
from rfp_rag.summary_content.pipeline import rebuild_document_summary


def normalized(value):
    return re.sub(r'\s+', '', value)


def field_text(result, field_id):
    field = next(f for f in result['fields'] if f['field_id'] == field_id)
    return '\n'.join(c['text'] for c in field.get('claims', []))


def rendered_text(result):
    return '\n'.join(field_text(result, f['id']) for f in FIELDS)


def source_fixture(lines, *, doc_id='invented-document-74', sha=None):
    """One exact block per paragraph: no synthetic text outside source spans."""
    sha = sha or hashlib.sha256('\n'.join(lines).encode()).hexdigest()
    blocks = []
    chunks = []
    for index, text in enumerate(lines):
        block = {'block_id':f'section0:paragraph:{index}', 'text':text,
                 'locator':f'BodyText/Section0 / paragraph {index}',
                 'section':'BodyText/Section0', 'paragraph':index}
        blocks.append(block)
        chunks.append({'chunk_id':f'{doc_id}:c{index}', 'doc_id':doc_id,
            'canonical_doc_id':doc_id, 'sha256':sha,
            'file_name':'independent-synthetic-rfp.hwp', 'text':text,
            'text_bytes':len(text.encode()), 'length_unit':'utf8_bytes',
            'kind':'prose', 'source_method':'raw_parser',
            'locator':block['locator'], 'section':block['section'],
            'source_spans':[{'collection':'blocks', 'block_id':block['block_id'],
                'locator':block['locator'], 'section':block['section'],
                'char_start':0, 'char_end':len(text),
                'chunk_start':0, 'chunk_end':len(text)}]})
    document = {'doc_id':doc_id, 'canonical_doc_id':doc_id, 'sha256':sha,
        'file_name':'independent-synthetic-rfp.hwp', 'blocks':blocks,
        'supplemental_blocks':[], 'is_duplicate_alias':False}
    registry = EvidenceRegistry([document],chunks)
    registry.validate_chunks(chunks,[doc_id])
    return document,chunks,registry


def make_claim(registry, chunks, field_id, paragraph, text=None, *, mode='fact'):
    chunk=chunks[paragraph]
    raw={'doc_id':chunk['doc_id'], 'category':FIELD_BY_ID[field_id]['category'],
         'text':f'[{field_id}|{mode}] '+(chunk['text'] if text is None else text),
         'citations':[{'chunk_id':chunk['chunk_id'],'quote':chunk['text']}]}
    checked=registry.validate_answer({'status':'answered','claims':[raw],
        'missing_information':[]},[chunk],[chunk['doc_id']])
    assert checked['valid'],checked
    return checked['claims'][0]


def saved_result(document,chunks,claims,*,execution='completed',coverage=1.0):
    fields=[]
    for spec in FIELDS:
        prefix='['+spec['id']+'|'
        selected=[]
        for claim in claims:
            if claim['text'].startswith(prefix):
                mode=claim['text'].split('|',1)[1].split(']',1)[0]
                selected.append({'text':claim['text'].split('] ',1)[1],
                    'mode':mode,'citations':copy.deepcopy(claim['citations'])})
        fields.append({'field_id':spec['id'],'label':spec['label'],
            'state':'evidence_found' if selected else 'not_found_in_processed_input',
            'claims':selected,'notes':[]})
    ids=[c['chunk_id'] for c in chunks]
    return {'version':'synthetic-test-run','doc_id':document['doc_id'],
        'source_sha256':document['sha256'],'run_fingerprint':'synthetic-input-run',
        'status':'partial','execution_status':execution,'answer_generated':bool(claims),
        'claims':copy.deepcopy(claims),'fields':fields,'rejected_claims':[],
        'source_chunk_ids':ids,'input_received_chunk_ids':ids if coverage==1 else ids[:1],
        'checked_chunk_ids':ids if coverage==1 else ids[:1],
        'input_chunk_coverage':coverage,'checked_chunk_coverage':coverage,
        'model_origin':{'all_live_or_live_cache':True},'missing_information':[],
        'batch_outcomes':[],'semantic_quality_verified':False}


class ContentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.network=patch('socket.socket.connect',side_effect=AssertionError('network prohibited'))
        self.network.start();self.addCleanup(self.network.stop)

    def rebuild(self,document,chunks,registry,claims,**options):
        result=saved_result(document,chunks,claims,**options)
        before=copy.deepcopy((result,document,chunks))
        derived=rebuild_document_summary(result,document,chunks,registry)
        self.assertEqual((result,document,chunks),before,'repair mutated an original artifact')
        self.assertEqual(derived['execution_status'],result['execution_status'])
        self.assertEqual(derived['generation_execution_status'],result['execution_status'])
        self.assertFalse(derived['semantic_quality_verified'])
        self.assertFalse(derived['human_review_completed'])
        self.assertIn('raw_model_result_fingerprint',derived['content_audit'])
        self.assertEqual(len(derived['fields']),11)
        return derived

    def test_post_contract_delivery_is_not_tender_submission(self):
        doc,chunks,registry=source_fixture([
            'Ⅰ. 과업 개요',
            '사업명 : 항만 데이터 정비 용역',
            '사업기간 : 계약체결일부터 2028. 2. 20.까지',
            'Ⅱ. 입찰 안내',
            '입찰 제출서류와 제출기한 및 제출방법은 입찰공고문 참조',
            'Ⅲ. 착수 및 준공 관련 제출문서',
            '가. 착수 시 (계약 시작일로부터 7일 이내 제출)',
            '착수계, 공정예정표, 보안각서, 책임자 선임계',
            '나. 준공 시 (준공 완료기한 내 제출)',
            '준공계, 준공검사원, 사업결과보고서를 문서로 제출한다.',
        ])
        claims=[make_claim(registry,chunks,'project_name',1),
                make_claim(registry,chunks,'submission_documents',7),
                make_claim(registry,chunks,'submission_deadline',6),
                make_claim(registry,chunks,'submission_method',9)]
        out=self.rebuild(doc,chunks,registry,claims)
        self.assertNotIn('착수계',field_text(out,'submission_documents'))
        self.assertNotIn('준공계',field_text(out,'submission_documents'))
        self.assertNotIn('7일',field_text(out,'submission_deadline'))
        self.assertNotIn('준공',field_text(out,'submission_method'))
        for fid in ('submission_documents','submission_deadline','submission_method'):
            self.assertIn('공고문',field_text(out,fid),fid)

    def test_other_project_appendix_does_not_override_canonical_project(self):
        doc,chunks,registry=source_fixture([
            'Ⅰ. 사업 개요',
            '사업명 : 항만 데이터 정비 용역',
            '사업기간 : 계약체결일로부터 270일',
            '※ 외부 연계 협의 지연 시 단계적 오픈 및 과업 조정 가능',
            'Ⅱ. 제안서 작성 안내',
            '제안서는 본 사업의 목적과 범위를 기준으로 작성한다.',
            '[붙임] 소프트웨어사업 영향평가 검토결과서',
            '1. 기본정보',
            '사업명 (또는 서비스명)',
            '산악안내 앱 구축',
            '사업기간 (또는 개발기간)',
            '2018년 5월 ~ 2018년 9월',
        ])
        claims=[make_claim(registry,chunks,'project_name',1),
                make_claim(registry,chunks,'duration',2),
                make_claim(registry,chunks,'project_name',9),
                make_claim(registry,chunks,'duration',11)]
        out=self.rebuild(doc,chunks,registry,claims)
        self.assertIn('항만 데이터',field_text(out,'project_name'))
        self.assertNotIn('산악안내',field_text(out,'project_name'))
        self.assertIn('270일',field_text(out,'duration'))
        self.assertNotIn('2018년',field_text(out,'duration'))
        self.assertIn('협의 지연',field_text(out,'duration'))

    def test_briefing_scope_not_email_string_controls_tender_filter(self):
        # The same email can legitimately be used for two different events.
        doc,chunks,registry=source_fixture([
            'Ⅰ. 사업 개요',
            '사업명 : 합성 연구플랫폼 ISP 수립',
            '□ [참고] 용역 사전설명회 계획 및 참가신청 안내',
            '신청기간 : 7월 9일(목) 15시',
            '신청방법 : 설명회 참가자는 liaison@example.org로 이메일 신청',
            'Ⅱ. 입찰 안내',
            '제출기한은 입찰공고문에 명기된 기한에 따른다.',
            '제안서 및 부속서류 제출 : 전자우편 또는 우편 제출',
            '제안서 제출 전자우편 : liaison@example.org',
            '우편 제출 시 제출마감 시한 이전 도착분에 한해 접수',
            'Ⅲ. 사업수행 관리',
            '착수계는 계약일로부터 10일 이내 제출하여야 한다.',
        ])
        claims=[make_claim(registry,chunks,'submission_deadline',3),
                make_claim(registry,chunks,'submission_method',4),
                make_claim(registry,chunks,'submission_method',8),
                make_claim(registry,chunks,'submission_deadline',11)]
        out=self.rebuild(doc,chunks,registry,claims)
        self.assertNotIn('7월 9일',field_text(out,'submission_deadline'))
        self.assertNotIn('10일',field_text(out,'submission_deadline'))
        self.assertIn('공고문',field_text(out,'submission_deadline'))
        method=field_text(out,'submission_method')
        self.assertNotIn('설명회',method)
        self.assertIn('liaison@example.org',method)

    def test_missing_budget_breakdown_and_stabilization_are_source_recovered(self):
        doc,chunks,registry=source_fixture([
            'Ⅰ. 사업 개요',
            '사업명 : 공공 연구정보 분석시스템 구축',
            '□ 사업예산 : 123백만원 (부가세포함)',
            'ㅇ 개발예산 (73백만원) 및 HW구입(50백만원)',
            '□ 사업기간 : 계약일로부터 7개월(안정화·모니터링 2개월 포함)',
            '※ 사업예산, 사업기간 및 입찰일정은 발주기관 내부 사정에 따라 조정될 수 있음',
        ])
        claims=[make_claim(registry,chunks,'budget_vat',2),
                make_claim(registry,chunks,'duration',4)]
        out=self.rebuild(doc,chunks,registry,claims)
        budget=normalized(field_text(out,'budget_vat'))
        self.assertIn('123백만원',budget)
        self.assertIn('73백만원',budget)
        self.assertIn('50백만원',budget)
        self.assertIn('부가세포함',budget)
        duration=normalized(field_text(out,'duration'))
        self.assertIn('7개월',duration)
        self.assertIn('2개월포함',duration)
        self.assertNotIn('9개월',duration)
        self.assertIn('조정',duration)
        self.assertGreater(out['content_audit']['local_extraction_count'],0)
        for claim in out['claims']:
            for cite in claim.get('citations',[]):
                self.assertIn(cite['quote'],registry.get_chunk(cite['chunk_id'])['text'])

    def test_unconfirmed_batch_statement_is_not_an_evidence_found_fact(self):
        doc,chunks,registry=source_fixture([
            'Ⅰ. 과업 범위',
            '요구사항 고유번호 SFR-001',
            '공공 데이터 분석 기능을 구현해야 한다.',
        ])
        claim=make_claim(registry,chunks,'evaluation',2,
            '제공된 원문 범위에서는 평가기준이 확인되지 않는다.')
        out=self.rebuild(doc,chunks,registry,[claim])
        f=next(f for f in out['fields'] if f['field_id']=='evaluation')
        self.assertEqual(f['claims'],[])
        self.assertNotEqual(f['state'],'evidence_found')
        self.assertTrue(out['content_audit']['filtered_claims'])

    def test_resource_example_table_does_not_swallow_following_binding_terms(self):
        doc,chunks,registry=source_fixture([
            'Ⅰ. 사업 개요',
            '사업명 : 공공 GPU 자원 공급 및 유지보수',
            '사업비 : 230백만원 (부가세 포함)',
            'Ⅱ. 자원 공급 요건',
            '공급자는 GPU 최소 120장 이상을 동적할당으로 제공한다.',
            '< 자원 제공 수량 제안 예시 >',
            '※ 단순 예시이며 실제와는 다름',
            '모델 A 40장 / 모델 B 80장 / 합계 120장',
            'o 공급자는 시간당 정산단가를 모델별로 제안하여야 한다.',
            '* 자원 사용시간은 2시간 단위로 책정',
            '※ 정산금액 누적 합산이 예산 한도에 도달할 경우 자원 제공 기간이 축소될 수 있음',
        ])
        out=self.rebuild(doc,chunks,registry,
            [make_claim(registry,chunks,'budget_vat',2)])
        self.assertIn('시간당 정산단가',field_text(out,'budget_vat'))
        self.assertIn('2시간 단위',field_text(out,'budget_vat'))
        self.assertNotIn('모델 A 40장',field_text(out,'budget_vat'))

    def test_agreement_name_in_tender_checklist_does_not_taint_later_tender_section(self):
        doc,chunks,registry=source_fixture([
            'Ⅰ. 사업 개요',
            '사업명 : 공공 연구정보 분석시스템 구축',
            'Ⅱ. 입찰 안내',
            '□ 입찰참가 등록신청 구비서류',
            '공동수급합의각서 및 협정서 1부',
            '□ 제안서 제출 방법 및 유의사항',
            '입찰공고문에 명기된 제출기한 내에 지정된 접수 장소에 제출하여야 한다.',
            '우편 제출 시 마감시한 이전 도착분에 한해 접수한다.',
        ])
        out=self.rebuild(doc,chunks,registry,[])
        self.assertIn('공고문',field_text(out,'submission_deadline'))

    def test_incomplete_generation_is_not_promoted_by_full_source_recovery(self):
        doc,chunks,registry=source_fixture([
            'Ⅰ. 사업 개요',
            '사업예산 : 91백만원 (부가세 포함)',
            '사업기간 : 계약체결일로부터 4개월',
        ])
        out=self.rebuild(doc,chunks,registry,[],execution='incomplete',coverage=0.25)
        self.assertEqual(out['execution_status'],'incomplete')
        self.assertEqual(out['input_chunk_coverage'],0.25)
        self.assertEqual(out['checked_chunk_coverage'],0.25)
        self.assertFalse(out.get('phase5_sample_validation_passed',False))

    def test_source_sha_or_document_scope_mismatch_is_rejected(self):
        doc,chunks,registry=source_fixture(['사업예산 : 75원 (부가세 포함)'])
        value=saved_result(doc,chunks,[])
        bad=copy.deepcopy(value);bad['source_sha256']='f'*64
        with self.assertRaises(ValueError):
            rebuild_document_summary(bad,doc,chunks,registry)
        other=copy.deepcopy(value);other['doc_id']='different-scope'
        with self.assertRaises(ValueError):
            rebuild_document_summary(other,doc,chunks,registry)

    def test_rejected_paraphrase_is_not_resurrected_by_category_repair(self):
        doc,chunks,registry=source_fixture([
            'Ⅰ. 사업 개요',
            '사업명 : 공공 연구정보 분석시스템 구축',
            '주요 범위 : 공공 데이터를 수집하고 분석결과를 시각화한다.',
        ])
        value=saved_result(doc,chunks,[])
        fabricated=make_claim(registry,chunks,'scope_requirements',2,
            '본 사업은 달 기지 건설 및 탐사로봇 개발을 수행한다.')
        fabricated['category']='other'
        value['rejected_claims']=[{'rejection_id':'synthetic-rejected-1',
            'batch_id':'synthetic-batch','claim_index':1,'field_id':'scope_requirements',
            'reason':'field_or_source_validation_failed',
            'raw_claim':fabricated,'raw_claim_is_unvalidated_diagnostic':True}]
        before=copy.deepcopy(value)
        out=rebuild_document_summary(value,doc,chunks,registry)
        self.assertEqual(value,before)
        self.assertNotIn('달 기지',rendered_text(out))
        self.assertNotIn('탐사로봇',rendered_text(out))


class SavedSevenDocumentRegressions(unittest.TestCase):
    """Optional checks bind current results to exact local original sources."""
    @classmethod
    def setUpClass(cls):
        source_path=os.environ.get('RFP_CONTENT_CORPUS_ROOT')
        run_path=os.environ.get('RFP_CONTENT_RUN_ROOT')
        if not source_path or not run_path:
            raise unittest.SkipTest('set RFP_CONTENT_CORPUS_ROOT and RFP_CONTENT_RUN_ROOT for saved-source regression')
        source_root=Path(source_path)
        cls.run_root=Path(run_path)
        with (source_root/'data/source_documents.jsonl').open(encoding='utf-8') as handle:
            cls.documents=[json.loads(s) for s in handle]
        with (source_root/'data/chunks.jsonl').open(encoding='utf-8') as handle:
            cls.chunks=[json.loads(s) for s in handle]
        cls.doc_by_id={d['doc_id']:d for d in cls.documents}
        cls.registry=EvidenceRegistry(cls.documents,cls.chunks)

    def saved(self,did):
        folder=self.run_root/json.loads((self.run_root/'documents'/did/'latest.json').read_text(encoding='utf-8'))['folder']
        result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        before=json.dumps(result,ensure_ascii=False,sort_keys=True)
        with patch('socket.socket.connect',side_effect=AssertionError('network prohibited')):
            derived=rebuild_document_summary(result,self.doc_by_id[did],
                [c for c in self.chunks if c['doc_id']==did],self.registry)
        self.assertEqual(json.dumps(result,ensure_ascii=False,sort_keys=True),before)
        self.assertFalse(derived['semantic_quality_verified'])
        self.assertFalse(derived['human_review_completed'])
        return derived

    def test_saved_payroll_does_not_call_execution_deliverables_tender_documents(self):
        out=self.saved('1Eq8gIdUnnLRbbEBiaU29wYji2YglNNuu')
        self.assertNotIn('착수계',field_text(out,'submission_documents'))
        self.assertNotIn('준공계',field_text(out,'submission_documents'))
        self.assertNotIn('7일',field_text(out,'submission_deadline'))
        self.assertIn('57,246,000',field_text(out,'budget_vat'))
        self.assertIn('실제',field_text(out,'budget_vat'))
        self.assertIn('과업기간 이후',field_text(out,'duration'))

    def test_saved_fashion_keeps_both_program_periods(self):
        out=self.saved('1Uy4SDf7e_NFKK9CbjL9uVO4ta_MQdeTt')
        duration=normalized(field_text(out,'duration'))
        self.assertIn('11.30',duration)
        self.assertIn('7.1',duration)
        self.assertIn('10개월',duration)
        self.assertIn('650,000,000',field_text(out,'budget_vat'))

    def test_saved_fashion_evaluation_retains_combined_threshold_and_weight_split(self):
        out=self.saved('1Uy4SDf7e_NFKK9CbjL9uVO4ta_MQdeTt')
        evaluation=normalized(field_text(out,'evaluation'))
        self.assertRegex(evaluation,r'정량.{0,12}20점')
        self.assertRegex(evaluation,r'정성.{0,12}70점')
        self.assertRegex(evaluation,r'가격.{0,12}10점')
        self.assertRegex(evaluation,r'종합평가점수.{0,35}70점이상')
        self.assertIn('추첨',evaluation)

    def test_saved_insight_keeps_development_hw_and_included_stabilization(self):
        out=self.saved('1G2at8Un7y8JIn5awuVxqV7OOKIj7zT4B')
        budget=normalized(field_text(out,'budget_vat'))
        for marker in ('688백만원','498백만원','190백만원'):
            self.assertIn(marker,budget)
        duration=normalized(field_text(out,'duration'))
        self.assertIn('5개월',duration)
        self.assertIn('1개월포함',duration)
        self.assertNotIn('처리기간은10일',duration)

    def test_saved_registration_removes_other_business_and_preserves_delay_condition(self):
        out=self.saved('1MxTCLhVvegbCCvqonlf95yO6pr4fYgaJ')
        self.assertNotIn('주택금융종합포털',field_text(out,'project_name'))
        self.assertNotIn('2020년',field_text(out,'duration'))
        self.assertIn('365일',field_text(out,'duration'))
        self.assertIn('협의 지연',field_text(out,'duration'))

    def test_saved_registration_budget_keeps_cap_estimated_volume_and_quantity_settlement(self):
        out=self.saved('1MxTCLhVvegbCCvqonlf95yO6pr4fYgaJ')
        budget=normalized(field_text(out,'budget_vat'))
        self.assertRegex(budget,r'280백만원?이내')
        self.assertIn('부가세포함',budget)
        self.assertIn('318,000',budget)
        self.assertIn('제공량',budget)
        self.assertIn('증감',budget)

    def test_saved_isp_uses_tender_notice_not_briefing_or_start_deadline(self):
        out=self.saved('161LuHCu_ZzUIDue5yGL_rRKMUe10FZ38')
        deadline=normalized(field_text(out,'submission_deadline'))
        self.assertNotIn('4월29일',deadline)
        self.assertNotIn('10일',deadline)
        self.assertIn('공고',deadline)
        self.assertIn('76.5',field_text(out,'evaluation'))

    def test_saved_union_retains_distinct_eighty_percent_threshold(self):
        out=self.saved('19K21s_OFwsevxfq9PJZFMSkNJVt9SPM0')
        evaluation=field_text(out,'evaluation')
        self.assertIn('80%',evaluation)
        self.assertNotIn('85%',evaluation)
        self.assertIn('1,189,000,0000',field_text(out,'budget_vat'))

    def test_saved_gpu_keeps_ratio_usage_settlement_and_eighty_five_percent(self):
        out=self.saved('1ZyM6Ipx-Zv-5E63Vb16dqhEy9VhsokLc')
        all_text=rendered_text(out)
        self.assertIn('672',field_text(out,'scope_requirements'))
        self.assertIn('1시간',field_text(out,'budget_vat'))
        self.assertIn('최대8개월',normalized(field_text(out,'duration')))
        self.assertIn('크레딧',all_text)
        self.assertIn('85%',field_text(out,'evaluation'))


if __name__=='__main__':
    unittest.main()
