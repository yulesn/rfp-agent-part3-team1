import copy
import inspect
import tempfile
import unittest
from pathlib import Path

from rfp_rag.retrieval.session import SearchSession


class FakeIndex:
    def __init__(self):
        self.calls = []
        self.hits = [{'chunk_id': 'A1', 'text': 'SECRET_GOLD_VALUE_777777', 'doc_id': 'A'}]
        self.catalog = [
            {'doc_id': 'A', 'file_name': '한국투자공사_2025_내부 인공지능 플랫폼 및 인프라 구축_제안요청서.pdf',
             'agency': '한국투자공사', 'document_year': 2025, 'title': '내부 인공지능 플랫폼 및 인프라 구축'},
            {'doc_id': 'B', 'file_name': '한국투자공사_2019_차세대 투자시스템 구축_제안요청서.pdf',
             'agency': '한국투자공사', 'document_year': 2019, 'title': '차세대 투자시스템 구축'},
            {'doc_id': 'C', 'file_name': '서울특별시_2026_2027년 침수예측 정보시스템 유지관리_제안요청서.pdf',
             'agency': '서울특별시', 'document_year': 2026, 'title': '2027년 침수예측 정보시스템 유지관리'},
        ]

    def get_catalog(self):
        return copy.deepcopy(self.catalog)

    def resolve_doc_ids(self, ids):
        aliases = {row['doc_id']: row['doc_id'] for row in self.catalog}
        aliases['A-copy'] = 'A'
        result = []
        for value in ([ids] if isinstance(ids, str) else ids):
            if value not in aliases:
                raise ValueError('unknown document')
            if aliases[value] not in result:
                result.append(aliases[value])
        return result

    def search(self, query, **kwargs):
        self.calls.append({'query': query, **kwargs})
        return copy.deepcopy(self.hits)

    def get_document_chunks(self, doc_id):
        return [{'chunk_id': f'{doc_id}{i}', 'doc_id': doc_id, 'text': str(i)} for i in range(12)]


class SessionTests(unittest.TestCase):
    def test_two_sessions_sharing_index_do_not_share_selection_or_history(self):
        index = FakeIndex()
        a, b = SearchSession(index, ['A']), SearchSession(index, ['B'])
        a.ask('예산은?')
        self.assertEqual(a.active_doc_ids, ['A'])
        self.assertEqual(b.active_doc_ids, ['B'])
        self.assertEqual(b.query_history, [])
        a.active_doc_ids.append('B')
        a.query_history.append('유출')
        self.assertEqual(a.active_doc_ids, ['A'])
        self.assertEqual(a.query_history, ['예산은?'])

    def test_alias_selection_passes_canonical_scope_to_all_methods(self):
        index = FakeIndex(); session = SearchSession(index, ['A-copy'])
        for method in ['bm25', 'char_tfidf', 'rrf', 'mmr']:
            result = session.ask('사업 예산은?', method=method)
            self.assertEqual(index.calls[-1]['allowed_doc_ids'], ['A'])
            self.assertEqual(result['status'], 'retrieved')
            self.assertFalse(result['answer_generated'])
            self.assertFalse(result['language_model_used'])

    def test_invalid_selection_preserves_previous_scope_and_does_not_query(self):
        index = FakeIndex(); session = SearchSession(index, ['A'])
        with self.assertRaises(ValueError):
            session.select_documents(['UNKNOWN'])
        self.assertEqual(session.active_doc_ids, ['A'])
        self.assertEqual(index.calls, [])

    def test_selection_switch_discards_old_query_for_rewriting(self):
        index = FakeIndex(); session = SearchSession(index, ['A'])
        session.ask('옛 사업의 GPU 서버 수량은?')
        session.select_documents(['B'])
        result = session.ask('그 사업의 기간은?')
        self.assertNotIn('GPU', result['standalone_query'])
        self.assertEqual(index.calls[-1]['allowed_doc_ids'], ['B'])

    def test_rewrite_uses_only_previous_user_question_never_retrieved_answer(self):
        index = FakeIndex(); session = SearchSession(index, ['A'])
        session.ask('GPU 서버 수량은?')
        result = session.ask('그 장비의 유지보수 조건은?')
        self.assertIn('GPU 서버 수량은?', result['standalone_query'])
        self.assertNotIn('SECRET_GOLD_VALUE_777777', result['standalone_query'])
        self.assertEqual(result['query_history'], ['GPU 서버 수량은?', '그 장비의 유지보수 조건은?'])

    def test_same_agency_without_project_selection_requests_clarification(self):
        index = FakeIndex(); session = SearchSession(index)
        result = session.ask('한국투자공사 시스템 사업의 예산은?')
        self.assertEqual(result['status'], 'needs_clarification')
        self.assertEqual({d['doc_id'] for d in result['candidate_documents']}, {'A', 'B'})
        self.assertEqual(result['active_doc_ids'], [])
        self.assertEqual(index.calls, [])

    def test_explicit_single_project_title_selects_new_scope(self):
        index = FakeIndex(); session = SearchSession(index, ['A'])
        result = session.ask('이번에는 차세대 투자시스템 구축으로 바꿀게. 기간은?')
        self.assertEqual(result['active_doc_ids'], ['B'])
        self.assertEqual(index.calls[-1]['allowed_doc_ids'], ['B'])

    def test_writing_year_different_from_business_year_is_not_rejected(self):
        index = FakeIndex(); session = SearchSession(index)
        result = session.ask('2027년 침수예측 정보시스템 유지관리 사업기간은?')
        self.assertEqual(result['status'], 'retrieved')
        self.assertEqual(result['active_doc_ids'], ['C'])
        self.assertIn('2027년', index.calls[-1]['query'])

    def test_unmatched_explicit_business_year_requests_confirmation_not_absence(self):
        index = FakeIndex(); session = SearchSession(index)
        result = session.ask('2025년 침수예측 정보시스템 유지관리 사업의 예산은?')
        self.assertEqual(result['status'], 'needs_clarification')
        self.assertIn('부재를 확정하지 않습니다', result['reason'])
        self.assertEqual(index.calls, [])

    def test_explicit_comparison_adds_both_named_projects_to_active_scope(self):
        index = FakeIndex(); session = SearchSession(index, ['A'])
        result = session.ask('내부 인공지능 플랫폼 및 인프라 구축과 차세대 투자시스템 구축의 기간을 비교해줘')
        self.assertEqual(set(result['active_doc_ids']), {'A', 'B'})
        self.assertEqual(set(index.calls[-1]['allowed_doc_ids']), {'A', 'B'})

    def test_comparison_combines_exact_and_year_omitted_project_titles(self):
        for initial in (None, ['A'], ['B']):
            with self.subTest(initial=initial):
                index = FakeIndex(); session = SearchSession(index, initial)
                result = session.ask('내부 인공지능 플랫폼 및 인프라 구축과 침수예측 정보시스템 유지관리의 예산을 비교해줘')
                self.assertEqual(set(result['active_doc_ids']), {'A', 'C'})
                self.assertEqual(set(index.calls[-1]['allowed_doc_ids']), {'A', 'C'})

    def test_comparison_year_omission_with_multiple_candidates_requests_selection(self):
        index = FakeIndex()
        index.catalog.append({'doc_id': 'D', 'agency': '서울특별시', 'document_year': 2027,
                              'title': '2028년 침수예측 정보시스템 유지관리',
                              'file_name': '서울특별시_2027_2028년 침수예측 정보시스템 유지관리.pdf'})
        session = SearchSession(index, ['B'])
        result = session.ask('내부 인공지능 플랫폼 및 인프라 구축과 침수예측 정보시스템 유지관리의 예산을 비교해줘')
        self.assertEqual(result['status'], 'needs_clarification')
        self.assertEqual(session.active_doc_ids, ['B'])
        self.assertEqual({row['doc_id'] for row in result['candidate_documents']}, {'A', 'C', 'D'})
        self.assertEqual(index.calls, [])

    def test_comparison_exact_year_does_not_add_other_year_from_nested_title_match(self):
        index = FakeIndex()
        index.catalog.append({'doc_id': 'D', 'agency': '서울특별시', 'document_year': 2027,
                              'title': '2028년 침수예측 정보시스템 유지관리',
                              'file_name': '서울특별시_2027_2028년 침수예측 정보시스템 유지관리.pdf'})
        result = SearchSession(index).ask('내부 인공지능 플랫폼 및 인프라 구축과 2027년 침수예측 정보시스템 유지관리의 예산을 비교해줘')
        self.assertEqual(set(result['active_doc_ids']), {'A', 'C'})
        self.assertEqual(set(index.calls[-1]['allowed_doc_ids']), {'A', 'C'})

    def test_caller_candidate_ambiguity_is_not_converted_to_gold_selection(self):
        index = FakeIndex(); session = SearchSession(index)
        result = session.ask('그 사업의 예산은?', candidate_doc_ids=['A-copy', 'B'])
        self.assertEqual(result['status'], 'needs_clarification')
        self.assertEqual(result['active_doc_ids'], [])
        self.assertEqual(index.calls, [])

    def test_empty_candidate_scope_does_not_expand_to_global_search(self):
        index = FakeIndex(); session = SearchSession(index)
        result = session.ask('예산은?', candidate_doc_ids=[])
        self.assertEqual(result['status'], 'insufficient_evidence')
        self.assertEqual(index.calls, [])

    def test_empty_candidate_scope_also_prevents_exact_title_escape(self):
        index = FakeIndex(); session = SearchSession(index)
        result = session.ask('내부 인공지능 플랫폼 및 인프라 구축 예산은?', candidate_doc_ids=[])
        self.assertEqual(result['status'], 'insufficient_evidence')
        self.assertEqual(index.calls, [])

    def test_named_project_outside_candidate_scope_is_not_selected(self):
        index = FakeIndex(); session = SearchSession(index)
        result = session.ask('내부 인공지능 플랫폼 및 인프라 구축 예산은?', candidate_doc_ids=['B'])
        self.assertEqual(result['status'], 'needs_clarification')
        self.assertEqual(result['active_doc_ids'], [])
        self.assertEqual(index.calls, [])

    def test_empty_results_do_not_assert_condition_absent(self):
        index = FakeIndex(); index.hits = []
        result = SearchSession(index, ['A']).ask('참가자격은?')
        self.assertEqual(result['status'], 'insufficient_evidence')
        self.assertIn('확정은 아닙니다', result['reason'])
        self.assertNotIn('문서에 없습니다', result['reason'])

    def test_blank_query_does_not_search_or_pollute_history(self):
        index = FakeIndex(); session = SearchSession(index)
        result = session.ask('  ')
        self.assertEqual(result['status'], 'needs_clarification')
        self.assertEqual(result['query_history'], [])
        self.assertEqual(index.calls, [])

    def test_full_document_route_is_not_top_k_and_updates_selection(self):
        index = FakeIndex(); session = SearchSession(index, ['B'])
        result = session.get_full_document_chunks('A-copy')
        self.assertEqual(result['status'], 'document_chunks')
        self.assertEqual(len(result['chunks']), 12)
        self.assertEqual(result['structural_chunk_coverage'], 1.0)
        self.assertFalse(result['full_original_semantic_coverage_verified'])
        self.assertEqual(session.active_doc_ids, ['A'])
        self.assertEqual(index.calls, [])

    def test_no_gold_or_assistant_history_parameter(self):
        signature = inspect.signature(SearchSession.ask)
        self.assertNotIn('expected_value', signature.parameters)
        self.assertNotIn('required_facts', signature.parameters)
        self.assertNotIn('assistant_answers', signature.parameters)
        with self.assertRaises(TypeError):
            SearchSession(FakeIndex()).ask('질문', expected_value='정답')

    def test_generic_unselected_query_can_search_globally_without_fabricated_scope(self):
        index = FakeIndex(); result = SearchSession(index).ask('개인정보 영향평가 관련 조건을 찾아줘')
        self.assertIsNone(index.calls[-1]['allowed_doc_ids'])
        self.assertEqual(result['active_doc_ids'], [])
        self.assertEqual(result['status'], 'retrieved')


class SessionIndexIntegrationTests(unittest.TestCase):
    def test_real_index_scope_aliases_history_and_full_document_route(self):
        from rfp_rag.retrieval.retrieval import LocalSearchIndex

        chunks = []
        examples = [
            ('A', '내부 인공지능 플랫폼 구축', ['A', 'A-copy'],
             '제안서는 온라인으로 제출한다. 사업 예산은 1억원이다.'),
            ('B', '차세대 투자시스템 구축', ['B'],
             '차세대 투자시스템의 유지관리 사업 예산은 3억원이다.'),
        ]
        for doc_id, title, aliases, text in examples:
            for sequence in range(2):
                chunks.append({'chunk_id': f'{doc_id}-{sequence}', 'doc_id': doc_id,
                               'canonical_doc_id': doc_id, 'alias_doc_ids': aliases,
                               'sha256': ('a' if doc_id == 'A' else 'b') * 64,
                               'file_name': f'한국투자공사_2025_{title}_제안요청서.pdf',
                               'agency': '한국투자공사', 'document_year': 2025,
                               'text': text, 'metadata_context': {'title': title}})
        with tempfile.TemporaryDirectory(prefix='session-index-test-') as directory:
            with LocalSearchIndex(Path(directory) / 'index') as index:
                index.build(chunks)
                session = SearchSession(index)
                self.assertEqual(session.ask('한국투자공사 사업 예산은?')['status'],
                                 'needs_clarification')
                session.select_documents(['A-copy'])
                for method in ['bm25', 'char_tfidf', 'rrf', 'mmr']:
                    result = session.ask('온라인 제출 방법은?', method=method)
                    self.assertEqual(result['status'], 'retrieved')
                    self.assertEqual({hit['doc_id'] for hit in result['hits']}, {'A'})
                    self.assertEqual(result['active_doc_ids'], ['A'])
                followup = session.ask('그 제출 방식은?')
                self.assertIn('직전 사용자 질문:', followup['standalone_query'])
                self.assertNotIn('1억원', followup['standalone_query'])
                self.assertEqual(session.get_full_document_chunks('A-copy')['returned_chunks'], 2)
                self.assertEqual(SearchSession(index).active_doc_ids, [])


if __name__ == '__main__':
    unittest.main()
