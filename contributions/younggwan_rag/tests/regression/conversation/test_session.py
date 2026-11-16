"""Conversation regressions use invented businesses and source text only."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rfp_rag.retrieval.chunking import build_chunks
from rfp_rag.generation.evidence import EvidenceRegistry
from rfp_rag.conversation.session import ConversationSession, ConversationConfig
from rfp_rag.conversation.runtime import GroundedBackend, ScopedIndex, make_conversation


def fixture():
    docs=[]
    for did,title,agency,year in [('A','항만 데이터 분석시스템 구축','테스트공공기관',2028),
        ('B','공원 시설물 관리시스템 개선','테스트공공기관',2028),
        ('C','교육 자료 통합플랫폼 구축','시범연구재단',2029)]:
        text=f'{title}. 사업예산 137백만원, 부가세 포함. 사업기간 계약 후 8개월(안정화 2개월 포함).'
        docs.append({'doc_id':did,'canonical_doc_id':did,'sha256':hashlib.sha256(text.encode()).hexdigest(),
            'file_name':f'{agency}_{year}_{title}_제안요청서.hwp','title':title,
            'agency':agency,'document_year':year,'detected_format':'hwp',
            'blocks':[{'block_id':'p1','text':text,'locator':'BodyText/Section0 / paragraph 1',
                'section':'BodyText/Section0','kind':'paragraph'}],
            'supplemental_blocks':[],'reference_blocks':[],'tables':[]})
    alias=copy.deepcopy(docs[0]);alias.update(doc_id='COPY_A',canonical_doc_id='A',is_duplicate_alias=True)
    docs.append(alias)
    chunks,_=build_chunks(docs)
    return docs,chunks,EvidenceRegistry(docs,chunks)


class RoutingRegressionTests(unittest.TestCase):
    """Independent examples of the eight observed scope failure patterns."""
    def setUp(self):
        original=fixture()[0][0]
        names=[('A','전자계약시스템 구축','시범은행'),
            ('B','보안 대응 정보보호 컨설팅 용역','가상은행'),
            ('C','통합정보시스템 고도화','시범기술연구원'),
            ('D','통합정보시스템 고도화','가상정보재단'),
            ('E','정보화전략계획(ISP) 수립','해양개발기관'),
            ('F','중장기 정보화전략계획(ISP) 수립 용역','시범공제회'),
            ('G','디지털 연구재단 2040 정보화전략계획을 위한 ISP','연구재단'),
            ('H','조합 정보시스템 개선 사업(전면개편)','시범공제조합'),
            ('I','지역 정보시스템마스터플랜(ISMP) 수립','해양개발기관')]
        self.docs=[]
        for did,title,agency in names:
            d=copy.deepcopy(original);d.update(doc_id=did,canonical_doc_id=did,title=title,
                agency=agency,sha256=hashlib.sha256(did.encode()).hexdigest())
            self.docs.append(d)
        self.s=ConversationSession(self.docs)

    def test_electronic_project_name_does_not_select_first_comparison_column(self):
        self.s.select_documents(['A','B'])
        r=self.s.plan('전자계약시스템 구축과 보안 대응 정보보호 컨설팅 용역의 예산과 기간 비교')
        self.assertEqual(r['effective_doc_ids'],['A','B'])
        self.assertEqual(r['feature'],'comparison')

    def test_first_second_pronouns_are_words_and_keep_the_comparison_order(self):
        self.s.select_documents(['A','B'])
        for query,expected in [('전자세금계산서 제출 조건은?',['A','B']),
            ('전자와 후자의 예산 비교',['A','B']),('전자는 예산이 얼마야?',['A']),
            ('후자는 몇 개월인가?',['B'])]:
            self.assertEqual(self.s.plan(query)['effective_doc_ids'],expected)

    def test_short_title_inside_longer_named_project_is_not_an_extra_document(self):
        self.s.select_documents(['F','G'])
        r=self.s.plan('「시범공제회: 중장기 정보화전략계획(ISP) 수립 용역」 / '
                      '「연구재단: 디지털 연구재단 2040 정보화전략계획을 위한 ISP」 예산 비교')
        self.assertEqual(r['effective_doc_ids'],['F','G'])

    def test_full_and_abbreviated_titles_in_same_comparison_both_survive(self):
        self.s.select_documents(['E','I'])
        r=self.s.plan('「해양개발기관: 정보화전략계획(ISP) 수립」 / '
                      '「해양개발기관: 지역 정보시스템마스터플랜(ISMP) 수립」 비교')
        self.assertEqual(r['effective_doc_ids'],['E','I'])

    def test_duplicate_title_is_disambiguated_by_agency_and_selected_document(self):
        self.s.select_documents(['C'])
        self.assertEqual(self.s.plan('시범기술연구원 통합정보시스템 고도화 예산은?')['effective_doc_ids'],['C'])
        self.assertEqual(self.s.plan('통합정보시스템 고도화 기간은?')['effective_doc_ids'],['C'])
        self.s.reset()
        self.assertEqual(self.s.plan('통합정보시스템 고도화 예산은?')['reason'],'ambiguous_named_document')

    def test_named_clarification_without_parenthesized_acronym_resumes_budget(self):
        self.s.select_documents(['E','G'])
        self.assertEqual(self.s.plan('그 사업 예산은?')['reason'],'ambiguous_singular_reference')
        r=self.s.plan('먼저 정보화전략계획 수립을 말한 거야.')
        self.assertEqual(r['effective_doc_ids'],['E'])
        self.assertIn('예산',r['retrieval_plan']['content_question'])
        self.assertIn('예산',r['standalone_query'])

    def test_abbreviated_switch_uses_title_tokens_and_clears_old_question(self):
        self.s.plan('개발비 예산은?',doc_ids=['F'])
        r=self.s.plan('디지털 연구재단 2040 ISP로 바꿀게. 그 기간은?')
        self.assertEqual(r['effective_doc_ids'],['G'])
        self.assertNotIn('개발비',r['standalone_query'])
        self.assertEqual(r['retrieval_plan']['topics'],['duration'])

    def test_business_word_omission_and_bakkul_switch_keep_only_requested_field(self):
        self.s.plan('보안 대응 정보보호 컨설팅 용역 사업기간은?')
        r=self.s.plan('조합 정보시스템 개선 전면개편으로 바꿀게. 그건?')
        self.assertEqual(r['effective_doc_ids'],['H'])
        self.assertEqual(r['inherited_facet'],'duration')
        self.assertIn('사업기간',r['retrieval_plan']['content_question'])
        self.assertNotIn('보안 대응',r['standalone_query'])

    def test_two_businesses_after_switch_restore_recent_id_pair(self):
        self.s.plan('사업기간은?',doc_ids=['B'])
        self.s.plan('조합 정보시스템 개선 전면개편으로 바꿀게. 그건?')
        r=self.s.plan('두 사업의 입찰 방식도 비교해줘.')
        self.assertEqual(r['effective_doc_ids'],['B','H'])
        self.assertEqual(r['feature'],'comparison')

    def test_recent_pair_roundtrip_and_legacy_state_are_both_supported(self):
        self.s.plan('기간은?',doc_ids=['B'])
        self.s.plan('기간은?',doc_ids=['H'])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'state.json';self.s.save(p)
            restored=ConversationSession(self.docs);restored.restore(p)
            self.assertEqual(restored.plan('이 두 건의 기간 비교')['effective_doc_ids'],['B','H'])
            raw=self.s.state;raw.pop('recent_doc_ids');raw['version']='rfp-conversation-v1';p.write_text(json.dumps(raw))
            restored.restore(p);self.assertEqual(restored.active_doc_ids,['H'])

    def test_filter_change_and_reset_clear_recent_pair(self):
        self.s.plan('기간은?',doc_ids=['B']);self.s.plan('기간은?',doc_ids=['H'])
        self.s.set_filters({'agency':'시범공제조합'})
        self.assertEqual(self.s.plan('두 사업 기간 비교')['status'],'needs_clarification')
        self.s.reset();self.assertEqual(self.s.state['recent_doc_ids'],[])
        self.assertEqual(self.s.plan('ISP로 바꿀게')['status'],'needs_clarification')

    def test_indistinguishable_abbreviated_names_still_need_clarification(self):
        d=copy.deepcopy(self.docs[6]);d.update(doc_id='J',canonical_doc_id='J',
            title='디지털 연구재단 2040 정보화 전략 계획(ISP)')
        s=ConversationSession(self.docs+[d])
        self.assertEqual(s.plan('디지털 연구재단 2040 ISP로 바꿀게. 예산은?')['status'],'needs_clarification')

    def test_parenthesized_english_name_keeps_the_following_project_description(self):
        d=copy.deepcopy(self.docs[6]);d.update(doc_id='J',canonical_doc_id='J',agency='가상거래소',
            title='차세대플랫폼(Ocean PORTAL+) 구축 등을 위한 정보화전략계획 수립 용역')
        s=ConversationSession(self.docs+[d]);s.select_documents(['J'])
        self.assertEqual(s.plan('가상거래소 Ocean PORTAL+ 구축 등을 위한 정보화전략계획 수립 용역 예산은?')['effective_doc_ids'],['J'])


class Index:
    def __init__(self,docs,chunks):self.docs=docs;self.chunks=chunks;self.calls=[];self.inject_outside=False;self.empty=False
    def get_catalog(self):return [d for d in self.docs if d['canonical_doc_id']==d['doc_id']]
    def search(self,query,allowed_doc_ids=None,top_k=8,**kwargs):
        self.calls.append({'query':query,'scope':copy.deepcopy(allowed_doc_ids)})
        if self.empty:return []
        if self.inject_outside:return self.chunks
        return [c for c in self.chunks if c['doc_id'] in allowed_doc_ids][:top_k]


class MockLLM:
    def __init__(self,mode='valid',enabled=True):
        from types import SimpleNamespace
        self.config=SimpleNamespace(enabled=enabled);self.mode=mode;self.requests=[]
    def generate(self,task,question,evidence):
        self.requests.append((task,question,copy.deepcopy(evidence)))
        if self.mode=='error':raise RuntimeError('SECRET_PROVIDER_KEY')
        if self.mode=='poison':
            pass
        claims=[{'doc_id':c['doc_id'],'category':'overview','text':c['text'],
            'citations':[{'chunk_id':c['chunk_id'],'quote':'FAKE_QUOTATION' if self.mode=='bad_quote' else c['text']}]} for c in evidence]
        return {'payload':{'status':'answered','claims':claims,'missing_information':[]},
            'live_api':False,'execution_mode':'mock','usage':{},'cache_hit':False}


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.docs,self.chunks,self.registry=fixture()
        self.s=ConversationSession(self.docs)
        self.block=patch('socket.socket.connect',side_effect=AssertionError('network prohibited'))
        self.block.start();self.addCleanup(self.block.stop)

    def test_all_original_ids_including_aliases_are_accessible(self):
        self.assertEqual(len(self.s.list_documents(include_aliases=True)),4)
        for d in self.docs:
            self.s.select_documents(d['doc_id'])
            self.assertEqual(self.s.plan('사업기간은?')['effective_doc_ids'],[d['canonical_doc_id']])

    def test_unknown_id_does_not_widen_or_erase_selection(self):
        self.s.select_documents('A')
        result=self.s.ask('예산은?',doc_ids=['UNKNOWN'])
        self.assertEqual(result['status'],'needs_clarification')
        self.assertEqual(self.s.active_doc_ids,['A'])

    def test_focused_named_document_keeps_the_explicit_pair_available(self):
        self.s.select_documents(['A','B'])
        one=self.s.plan('항만 데이터 분석시스템 구축의 예산은?')
        self.assertEqual(one['effective_doc_ids'],['A'])
        self.assertEqual(self.s.active_doc_ids,['A','B'])
        self.assertEqual(self.s.plan('두 문서 기간을 비교해줘')['effective_doc_ids'],['A','B'])

    def test_more_than_two_recent_documents_require_an_explicit_pair(self):
        for did in ['A','B','C']:self.s.plan('기간은?',doc_ids=[did])
        self.assertEqual(self.s.plan('두 사업 기간을 비교해줘')['reason'],'recent_pair_ambiguous')
        self.assertEqual(self.s.plan('최근 두 사업 기간을 비교해줘')['effective_doc_ids'],['B','C'])

    def test_period_change_question_is_not_a_document_switch(self):
        self.s.select_documents('A')
        self.assertEqual(self.s.plan('과업기간을 바꿀 수 있어?')['effective_doc_ids'],['A'])

    def test_switch_to_a_shared_title_does_not_silently_choose_old_selection(self):
        d=copy.deepcopy(self.docs[0]);d.update(doc_id='D',canonical_doc_id='D',sha256='f'*64)
        s=ConversationSession(self.docs+[d]);s.select_documents('A')
        self.assertEqual(s.plan('항만 데이터 분석시스템 구축으로 바꿀게. 예산은?')['reason'],'ambiguous_named_document')

    def test_v2_state_rejects_missing_recent_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'state.json';self.s.save(p)
            raw=json.loads(p.read_text());raw.pop('recent_doc_ids');p.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):self.s.restore(p)

    def test_no_selection_and_pronoun_requires_clarification(self):
        self.assertEqual(self.s.plan('그건 몇 개월이야?')['reason'],'document_not_selected')
        self.assertEqual(self.s.active_doc_ids,[])

    def test_full_title_selects_document_and_pronoun_keeps_it(self):
        self.s.plan('항만 데이터 분석시스템 구축 예산은?')
        result=self.s.plan('그건 몇 개월짜리야? 안정화도 포함이야?')
        self.assertEqual(result['effective_doc_ids'],['A'])
        self.assertIn('항만 데이터 분석시스템 구축',result['standalone_query'])
        self.assertNotIn('직전 사용자 질문',result['standalone_query'])
        self.assertEqual(result['facet'],'duration')

    def test_same_facet_reuses_original_user_question_only_once(self):
        self.s.select_documents('A');self.s.plan('하드웨어 구매 예산이 얼마야?')
        self.s.plan('그건 부가세 포함이야?')
        result=self.s.plan('그럼 세금은?')
        self.assertNotIn('하드웨어 구매',result['standalone_query'])
        self.assertEqual(result['standalone_query'].count('직전 사용자 질문'),1)

    def test_switch_breaks_previous_project_and_topic_history(self):
        self.s.plan('항만 데이터 분석시스템 구축 하드웨어 예산은?')
        self.s.plan('공원 시설물 관리시스템 개선 사업기간은?')
        result=self.s.plan('그 안에 안정화가 들어가?')
        self.assertEqual(result['effective_doc_ids'],['B'])
        self.assertNotIn('항만',result['standalone_query'])
        self.assertNotIn('하드웨어',result['standalone_query'])

    def test_switch_without_target_cannot_fall_back_to_old_document(self):
        self.s.select_documents('A')
        self.assertEqual(self.s.ask('다른 문서로 바꿔줘')['reason'],'switch_target_missing')
        self.assertEqual(self.s.active_doc_ids,['A'])

    def test_pair_singular_is_ambiguous_and_first_second_are_explicit(self):
        self.s.select_documents(['A','B'])
        self.s.plan('두 사업 예산을 비교해줘')
        self.assertEqual(self.s.plan('그건 몇 개월이야?')['reason'],'ambiguous_singular_reference')
        self.assertEqual(self.s.plan('첫 번째 사업의 기간은?')['effective_doc_ids'],['A'])
        self.assertEqual(self.s.plan('그 안에 안정화 포함이야?')['effective_doc_ids'],['A'])
        self.assertEqual(self.s.plan('두 문서 각각 기간을 비교해줘')['effective_doc_ids'],['A','B'])
        self.assertEqual(self.s.plan('두 번째 사업은?')['effective_doc_ids'],['B'])

    def test_deictic_variants_are_not_implicit_multi_document_comparisons(self):
        for reference in ['그건','그것','이거','이것','여기','거기','이 문서','해당 사업']:
            s=ConversationSession(self.docs);s.select_documents(['A','B'])
            self.assertEqual(s.plan(reference+' 예산은?')['reason'],'ambiguous_singular_reference')

    def test_mention_of_other_business_in_current_document_needs_confirmation(self):
        self.s.select_documents('A')
        r=self.s.plan('이 문서에 공원 시설물 관리시스템 개선이 기재되어 있는데 왜 그래?')
        self.assertEqual(r['reason'],'mentioned_project_or_switch')
        self.assertEqual(self.s.active_doc_ids,['A'])

    def test_numbered_clarification_resumes_original_question(self):
        pending=self.s.ask('테스트공공기관 사업예산은?')
        self.assertEqual(len(pending['candidate_documents']),2)
        result=self.s.plan('2번')
        self.assertEqual(result['effective_doc_ids'],['B'])
        self.assertIn('사업예산',result['standalone_query'])
        self.assertNotIn('질문: 2번',result['standalone_query'])

    def test_comparison_requires_two_documents(self):
        self.s.select_documents('A')
        self.assertEqual(self.s.ask('두 사업 예산 비교')['reason'],'comparison_needs_multiple_documents')

    def test_explicit_selection_conflict_is_clarified(self):
        self.s.select_documents('A')
        r=self.s.ask('공원 시설물 관리시스템 개선 예산은?',doc_ids=['A'])
        self.assertEqual(r['reason'],'explicit_selection_conflicts_with_question')
        self.assertEqual(self.s.active_doc_ids,['A'])

    def test_filters_persist_and_block_cross_agency_switch(self):
        self.s.set_filters({'agency':'테스트공공기관','document_year':2028})
        self.s.plan('항만 데이터 분석시스템 구축 예산은?')
        r=self.s.ask('교육 자료 통합플랫폼 구축 기간은?')
        self.assertEqual(r['reason'],'named_document_outside_filters')
        self.assertEqual(self.s.active_doc_ids,['A'])
        self.assertEqual(self.s.plan('그건 몇 개월이야?')['filters']['document_year'],2028)

    def test_explicit_filter_change_clears_old_context(self):
        self.s.select_documents('A');self.s.plan('항만 하드웨어 예산은?')
        self.s.set_filters({'document_year':2029})
        self.assertEqual(self.s.active_doc_ids,[])
        self.assertEqual(self.s.state['history'],[])
        r=self.s.plan('교육 자료 통합플랫폼 구축 기간은?')
        self.assertEqual(r['effective_doc_ids'],['C'])
        self.assertNotIn('하드웨어',r['standalone_query'])

    def test_empty_filter_match_does_not_search_globally(self):
        r=self.s.ask('예산은?',filters={'document_year':2099})
        self.assertEqual(r['reason'],'empty_filtered_scope')

    def test_wrong_year_is_uncertain_not_missing_document(self):
        r=self.s.ask('2031년 항만 데이터 분석시스템 구축 사업기간은?')
        self.assertEqual(r['reason'],'year_scope_ambiguous')

    def test_duplicate_titles_need_explicit_year_or_id(self):
        d=copy.deepcopy(self.docs[0]);d.update(doc_id='D',canonical_doc_id='D',document_year=2030,sha256='f'*64)
        s=ConversationSession(self.docs+[d])
        self.assertEqual(s.plan('항만 데이터 분석시스템 구축 예산은?')['reason'],'ambiguous_named_document')

    def test_multiple_full_titles_are_resolved_as_comparison(self):
        r=self.s.plan('항만 데이터 분석시스템 구축과 공원 시설물 관리시스템 개선의 예산 비교')
        self.assertEqual(set(r['effective_doc_ids']),{'A','B'})
        self.assertEqual(r['feature'],'comparison')

    def test_comparison_word_in_project_name_is_not_comparison_intent(self):
        docs=copy.deepcopy(self.docs)
        docs[0]['title']='국가 통계 비교분석 플랫폼 구축'
        s=ConversationSession(docs)
        r=s.plan('국가 통계 비교분석 플랫폼 구축 예산은?')
        self.assertEqual(r['status'],'ready');self.assertEqual(r['effective_doc_ids'],['A'])
        self.assertEqual(r['feature'],'qa');self.assertEqual(r['facet'],'budget')

    def test_state_copy_cannot_mutate_session(self):
        self.s.select_documents('A');value=self.s.state;value['active_doc_ids'].append('B')
        self.assertEqual(self.s.active_doc_ids,['A'])

    def test_history_is_bounded_and_reset_clears_every_reference(self):
        self.s.select_documents('A')
        for _ in range(30):self.s.plan('그건 예산이 얼마야?')
        self.assertLessEqual(len(self.s.state['history']),8)
        self.assertLess(len(self.s.plan('그럼 세금은?')['standalone_query']),3500)
        self.s.reset();self.assertEqual(self.s.state['history'],[])
        self.assertEqual(self.s.ask('그건?')['status'],'needs_clarification')

    def test_invalid_config_and_filters_fail_early(self):
        with self.assertRaises(ValueError):ConversationConfig(max_history_turns=False)
        self.assertEqual(self.s.ask('예산은?',filters={'unsupported':1})['reason'],'invalid_scope')
        with self.assertRaises(ValueError):self.s.set_filters({'document_year':True})

    def test_state_roundtrip_has_only_user_history_and_rejects_source_drift(self):
        self.s.select_documents('A');self.s.plan('예산은?')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'state.json';self.s.save(p)
            other=ConversationSession(self.docs);other.restore(p)
            self.assertEqual(other.state,self.s.state)
            raw=json.loads(p.read_text());raw['history'][0]['assistant']='UNTRUSTED'
            p.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):other.restore(p)
            self.s.save(p);raw=json.loads(p.read_text());raw['catalog_fingerprint']='changed';p.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):other.restore(p)

    def test_pending_question_survives_state_roundtrip(self):
        self.s.ask('테스트공공기관 예산은?')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'pending.json';self.s.save(p)
            other=ConversationSession(self.docs);other.restore(p)
            result=other.plan('1번')
            self.assertEqual(result['effective_doc_ids'],['A'])
            self.assertIn('예산',result['standalone_query'])

    def test_invalid_candidate_number_preserves_pending_question(self):
        self.s.ask('테스트공공기관 예산은?')
        r=self.s.plan('9번')
        self.assertEqual(r['reason'],'invalid_candidate_number')
        self.assertIn('예산',self.s.plan('1번')['standalone_query'])

    def test_filter_conflict_in_same_turn_is_atomic(self):
        self.s.select_documents('A');before=self.s.state
        r=self.s.plan('기간은?',doc_ids=['B'],filters={'document_year':2029})
        self.assertEqual(r['reason'],'invalid_scope')
        self.assertEqual(self.s.active_doc_ids,before['active_doc_ids'])
        self.assertEqual(self.s.filters,{})

    def test_sessions_do_not_share_selected_documents_or_history(self):
        other=ConversationSession(self.docs)
        self.s.plan('예산은?',doc_ids=['A']);other.plan('기간은?',doc_ids=['B'])
        self.assertEqual(self.s.active_doc_ids,['A'])
        self.assertEqual(other.active_doc_ids,['B'])
        self.assertNotIn('예산',other.plan('그 안에 안정화가 포함돼?')['standalone_query'])

    def test_all_selection_supports_large_catalog_without_rewrite_overflow(self):
        docs=[]
        for i in range(120):
            row=copy.deepcopy(self.docs[0]);row.update(doc_id=f'doc-{i}',canonical_doc_id=f'doc-{i}',
                sha256=hashlib.sha256(str(i).encode()).hexdigest(),title=f'독립된 가상사업 {i} 공공 인프라 운영',
                file_name=f'independent-{i}.pdf')
            docs.append(row)
        s=ConversationSession(docs);s.select_documents('all')
        r=s.plan('전체 문서의 기간을 비교해 줘')
        self.assertEqual(len(r['effective_doc_ids']),120)
        self.assertLessEqual(len(r['standalone_query']),3500)


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.docs,self.chunks,self.registry=fixture();self.index=Index(self.docs,self.chunks)
        self.env={'root':Path(tempfile.gettempdir())/'rfp-phase6-test','documents':self.docs,
            'chunks':self.chunks,'registry':self.registry,'index':self.index}
        self.block=patch('socket.socket.connect',side_effect=AssertionError('network prohibited'))
        self.block.start();self.addCleanup(self.block.stop)

    def session(self,llm=None):
        return ConversationSession(self.docs,backend=GroundedBackend(self.env,llm or MockLLM()))

    def test_every_turn_retrieves_fresh_evidence_inside_its_document(self):
        llm=MockLLM();s=self.session(llm)
        s.ask('예산은?',doc_ids=['A']);s.ask('그건 몇 개월이야?')
        r=s.ask('기간은?',doc_ids=['B'])
        self.assertEqual([set(c['doc_id'] for c in evidence) for _,_,evidence in llm.requests],[{'A'},{'A'},{'B'}])
        self.assertTrue(r['answer_generated']);self.assertTrue(r['validation']['valid'])

    def test_assistant_poison_never_enters_rewritten_query(self):
        grounded=GroundedBackend(self.env,MockLLM())
        def backend(q,*,doc_ids,feature):
            result=grounded(q,doc_ids=doc_ids,feature=feature)
            result['markdown']+='\nPOISON_PREVIOUS_ANSWER_900M'
            return result
        s=ConversationSession(self.docs,backend=backend)
        s.ask('사업예산은?',doc_ids=['A']);r=s.ask('그럼 VAT는?')
        self.assertNotIn('POISON',r['conversation']['standalone_query'])
        self.assertNotIn('POISON',json.dumps(s.state))

    def test_ambiguous_turn_calls_neither_search_nor_llm(self):
        llm=MockLLM();s=self.session(llm);s.select_documents(['A','B'])
        r=s.ask('그건 몇 개월이야?')
        self.assertEqual(r['status'],'needs_clarification')
        self.assertEqual(self.index.calls,[]);self.assertEqual(llm.requests,[])

    def test_unresolved_switch_and_its_followup_never_use_the_old_document(self):
        llm=MockLLM();s=self.session(llm)
        s.ask('예산은?',doc_ids=['A']);calls=len(self.index.calls)
        r=s.ask('새로운 미등록 플랫폼으로 바꿀게. 그건?')
        self.assertEqual(r['status'],'needs_clarification')
        r=s.ask('그 예산은 얼마야?')
        self.assertEqual(r['reason'],'switch_target_still_pending')
        self.assertEqual(r['effective_doc_ids'],[])
        self.assertEqual(len(self.index.calls),calls)
        self.assertEqual(len(llm.requests),1)

    def test_out_of_scope_index_hit_fails_before_generation(self):
        llm=MockLLM();s=self.session(llm);self.index.inject_outside=True
        r=s.ask('예산은?',doc_ids=['A'])
        self.assertFalse(r['answer_generated']);self.assertEqual(r['claims'],[])
        self.assertEqual(llm.requests,[])

    def test_fabricated_quote_and_provider_errors_are_not_displayed(self):
        for mode in ['bad_quote','error']:
            r=self.session(MockLLM(mode)).ask('예산은?',doc_ids=['A'])
            self.assertFalse(r['answer_generated']);self.assertEqual(r['claims'],[])
            self.assertNotIn('FAKE_QUOTATION',r['markdown'])
            self.assertNotIn('SECRET_PROVIDER_KEY',r['markdown'])

    def test_bad_backend_cannot_return_other_document_claims(self):
        def bad(q,**kw):return {'active_doc_ids':['B'],'claims':[],
            'markdown':'WRONG_DOCUMENT_PRIVATE_TEXT','answer_generated':True}
        s=ConversationSession(self.docs,backend=bad)
        r=s.ask('예산은?',doc_ids=['A'])
        self.assertFalse(r['answer_generated']);self.assertNotIn('PRIVATE',r['markdown'])

    def test_empty_retrieval_is_not_generated(self):
        self.index.empty=True;llm=MockLLM()
        r=self.session(llm).ask('최종 낙찰업체를 알려줘',doc_ids=['A'])
        self.assertEqual(r['status'],'insufficient_evidence');self.assertEqual(llm.requests,[])

    def test_comparison_keeps_per_document_quotas_and_focus(self):
        s=self.session();r=s.ask('두 사업 예산 비교',doc_ids=['A','B'])
        self.assertEqual({c['doc_id'] for c in r['claims']},{'A','B'})
        self.assertTrue(r['retrieval']['document_quotas_applied_before_packing'])
        r=s.ask('첫 번째 사업의 예산은?')
        self.assertEqual({c['doc_id'] for c in r['claims']},{'A'})

    def test_backend_does_not_reinfer_comparison_from_embedded_title(self):
        llm=MockLLM()
        backend=GroundedBackend(self.env,llm)
        r=backend('질문: 사업예산은?\n대상 사업: 국가 통계 비교분석 플랫폼 구축',doc_ids=['A'],feature='qa')
        self.assertTrue(r['answer_generated'])
        self.assertEqual(llm.requests[-1][0],'qa')
        self.assertIn('비교분석',llm.requests[-1][1])

    def test_scoped_index_cannot_be_widened(self):
        index=ScopedIndex(self.index,self.registry,['A'])
        with self.assertRaises(ValueError):index.search('예산',allowed_doc_ids=['B'])
        with self.assertRaises(ValueError):ScopedIndex(self.index,self.registry,[])

    def test_default_factory_imports_no_sdk_or_key_and_calls_no_network(self):
        s=make_conversation(self.env)
        r=s.ask('예산은?',doc_ids=['A'])
        self.assertEqual(r['status'],'retrieved');self.assertFalse(r['answer_generated'])
        self.assertEqual(s.llm.usage_report()['api_calls'],0)
        self.assertTrue(r['source_chunks'])

    def test_live_scope_not_approved_is_blocked_before_llm(self):
        with self.assertRaises(ValueError):make_conversation(self.env,enabled=True)
        s=make_conversation(self.env,enabled=True,approved_doc_ids=['A'])
        r=s.ask('예산은?',doc_ids=['B'])
        self.assertEqual(r['execution_status'],'not_approved')
        self.assertEqual(s.llm.usage_report()['api_calls'],0)

    def test_backend_shares_one_budget_client_across_scope_switches(self):
        s=make_conversation(self.env)
        client=s.llm
        s.ask('예산은?',doc_ids=['A']);s.ask('기간은?',doc_ids=['B'])
        self.assertIs(s.llm,client)

    def test_selection_only_turn_does_not_generate(self):
        llm=MockLLM();s=self.session(llm)
        s.select_documents('A')
        r=s.ask('공원 시설물 관리시스템 개선 문서로 바꿔줘')
        self.assertEqual(r['status'],'selected')
        self.assertEqual(s.active_doc_ids,['B'])
        self.assertEqual(llm.requests,[])
        self.assertEqual(self.index.calls,[])


if __name__=='__main__':unittest.main()
