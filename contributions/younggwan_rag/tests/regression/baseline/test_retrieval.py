import hashlib
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rfp_rag.retrieval.retrieval import LocalSearchIndex, normalize_retrieval_text


def chunk(chunk_id, doc_id, text, aliases=None, **updates):
    value = {'chunk_id': chunk_id, 'doc_id': doc_id, 'canonical_doc_id': doc_id,
             'alias_doc_ids': aliases or [doc_id], 'sha256': hashlib.sha256(doc_id.encode()).hexdigest(),
             'file_name': doc_id + '.hwp', 'agency': '기관' + doc_id, 'document_year': 2026,
             'kind': 'prose', 'text': text, 'search_text': text, 'locator': 'BodyText/Section0 / paragraph 1',
             'table_ids': [], 'source_method': 'raw_parser', 'quality_flags': [],
             'length_unit': 'utf8_bytes', 'text_bytes': len(text.encode()),
             'source_spans': [{'block_id': 'b1', 'locator': 'BodyText/Section0 / paragraph 1',
                               'char_start': 0, 'char_end': len(text), 'chunk_start': 0,
                               'chunk_end': len(text), 'collection': 'blocks'}]}
    value.update(updates)
    return value


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'index'
        self.index = LocalSearchIndex(self.path)

    def tearDown(self):
        self.index.close()
        self.temp.cleanup()

    def build(self, rows, **config):
        return self.index.build(rows, config or None)

    def test_spaced_hangul_restoration_is_search_only(self):
        original = '사 업 비 : 100,000,000원 (부가가치세 포함)'
        self.build([chunk('a1', 'a', original), chunk('b1', 'b', '수행기간 12개월')])
        result = self.index.search('예산 부가세', method='bm25')
        self.assertEqual(result[0]['chunk_id'], 'a1')
        self.assertEqual(result[0]['text'], original)
        self.assertEqual(result[0]['source_spans'][0]['char_end'], len(original))
        self.assertIn('사업비', normalize_retrieval_text(original))

    def test_korean_compounds_retrieve_without_morphological_service(self):
        self.build([chunk('a1', 'a', '사업수행기간은 계약일부터 8개월이다.'),
                    chunk('b1', 'b', '주요 산출물은 보고서와 프로그램이다.')], min_df=1)
        for method in ['bm25', 'char_tfidf', 'rrf']:
            with self.subTest(method=method):
                self.assertEqual(self.index.search('수행기간 알려줘', method=method)[0]['doc_id'], 'a')

    def test_declared_budget_expansion_does_not_invent_answer(self):
        self.build([chunk('a1', 'a', '용역비 800만원 이내'), chunk('b1', 'b', '업무 개요와 주요 산출물')])
        result = self.index.search('예산', method='bm25')
        self.assertEqual(result[0]['doc_id'], 'a')
        self.assertEqual(result[0]['text'], '용역비 800만원 이내')
        self.assertNotIn('예산', result[0]['text'])

    def test_numeric_commas_and_vat_qualifier_survive(self):
        self.build([chunk('a1', 'a', '사업예산 100,000,000원 부가가치세 제외'),
                    chunk('b1', 'b', '사업예산 200,000,000원 부가가치세 포함')])
        result = self.index.search('100000000 부가세 제외', method='bm25', top_k=1)
        self.assertEqual(result[0]['doc_id'], 'a')
        self.assertIn('100,000,000원', result[0]['text'])
        self.assertIn('제외', result[0]['text'])

    def test_scope_is_applied_before_candidate_truncation_in_all_methods(self):
        rows = [chunk(f'b{i}', f'b{i}', '예산 부가세 예산 부가세 사업비 100원') for i in range(12)]
        rows.append(chunk('a1', 'a', '예산 조건은 별도 확인'))
        self.build(rows)
        for method in ['bm25', 'char_tfidf', 'rrf', 'mmr']:
            with self.subTest(method=method):
                result = self.index.search('예산 부가세', top_k=1, fetch_k=1,
                                           allowed_doc_ids=['a'], method=method)
                self.assertEqual([r['doc_id'] for r in result], ['a'])

    def test_alias_resolution_keeps_content_indexed_once(self):
        result = self.build([chunk('a1', 'a', '사업기간 8개월', ['a', 'a-copy']),
                             chunk('b1', 'b', '사업기간 10개월')])
        self.assertEqual(result['chunk_count'], 2)
        self.assertEqual(result['canonical_document_count'], 2)
        self.assertEqual(self.index.resolve_doc_ids(['a-copy', 'a', 'a-copy']), ['a'])
        hits = self.index.search('기간', allowed_doc_ids=['a-copy'])
        self.assertEqual([r['chunk_id'] for r in hits], ['a1'])

    def test_unknown_or_empty_scope_never_falls_back_to_global(self):
        self.build([chunk('a1', 'a', '예산 10원')])
        for method in ['bm25', 'char_tfidf', 'rrf', 'mmr']:
            self.assertEqual(self.index.search('예산', allowed_doc_ids=[], method=method), [])
            with self.assertRaises(ValueError):
                self.index.search('예산', allowed_doc_ids=['not-a-document'], method=method)
        with self.assertRaises(ValueError):
            self.index.resolve_doc_ids(None)

    def test_full_document_route_preserves_build_sequence_and_alias(self):
        self.build([chunk('a-z', 'a', '첫 구간', ['a', 'copy']),
                    chunk('b1', 'b', '다른 문서'), chunk('a-a', 'a', '둘째 구간', ['a', 'copy'])])
        self.assertEqual([c['chunk_id'] for c in self.index.get_document_chunks('copy')], ['a-z', 'a-a'])
        catalog = self.index.get_catalog()
        self.assertEqual(len(catalog), 2)
        self.assertIn('copy', next(d for d in catalog if d['doc_id'] == 'a')['alias_doc_ids'])
        catalog[0]['alias_doc_ids'].append('not-real')
        with self.assertRaises(ValueError):
            self.index.resolve_doc_ids(['not-real'])

    def test_persisted_reload_is_deterministic(self):
        rows = [chunk('a1', 'a', '예산 10원 VAT 포함'), chunk('b1', 'b', '사업기간 12개월')]
        metadata = self.build(rows)
        before = self.index.search('예산 VAT', method='rrf')
        self.index.close()
        reloaded = LocalSearchIndex(self.path)
        try:
            after = reloaded.search('예산 VAT', method='rrf')
            self.assertEqual([r['chunk_id'] for r in before], [r['chunk_id'] for r in after])
            self.assertEqual([r['score'] for r in before], [r['score'] for r in after])
            self.assertEqual(reloaded.get_metadata()['index_fingerprint'], metadata['index_fingerprint'])
        finally:
            reloaded.close()

    def test_same_content_and_config_reuses_valid_cache(self):
        rows = [chunk('a1', 'a', '예산 10원')]
        first = self.build(rows)
        second = self.build(rows)
        self.assertFalse(first['cache_reused'])
        self.assertTrue(second['cache_reused'])
        self.assertEqual(first['index_fingerprint'], second['index_fingerprint'])

    def test_changed_text_with_same_chunk_id_rebuilds_instead_of_stale_vectors(self):
        first = self.build([chunk('a1', 'a', 'oldtermzz 예산')])
        second = self.build([chunk('a1', 'a', 'newtermyy 기간')])
        self.assertNotEqual(first['chunks_fingerprint'], second['chunks_fingerprint'])
        self.assertFalse(second['cache_reused'])
        self.assertEqual(self.index.search('oldtermzz', method='bm25'), [])
        self.assertEqual(self.index.search('newtermyy', method='bm25')[0]['text'], 'newtermyy 기간')

    def test_actual_rrf_combines_ranks_not_concatenation(self):
        self.build([chunk('a1', 'a', '갑 예산'), chunk('b1', 'b', '을 기간'), chunk('c1', 'c', '병 자격')])
        with patch.object(self.index, '_bm25_candidates', return_value=[(1, 10), (2, 8)]), \
             patch.object(self.index, '_char_candidates', return_value=[(2, .9), (3, .7)]):
            hits = self.index.search('비교 질문', method='rrf', top_k=3)
        self.assertEqual([r['chunk_id'] for r in hits], ['b1', 'a1', 'c1'])
        self.assertAlmostEqual(hits[0]['score'], 1 / 62 + 1 / 61)
        self.assertEqual(hits[0]['component_ranks'], {'bm25': 2, 'char_tfidf': 1})

    def test_rrf_zero_component_weight_is_respected(self):
        self.build([chunk('a1', 'a', '갑 예산'), chunk('b1', 'b', '을 기간'), chunk('c1', 'c', '병 자격')],
                   char_tfidf_weight=0)
        with patch.object(self.index, '_bm25_candidates', return_value=[(1, 10), (2, 8)]), \
             patch.object(self.index, '_char_candidates', return_value=[(2, .9), (3, .7)]):
            hits = self.index.search('비교 질문', method='rrf', top_k=3)
        self.assertEqual([r['chunk_id'] for r in hits], ['a1', 'b1'])
        self.assertAlmostEqual(hits[0]['score'], 1 / 61)

    def test_mmr_selects_diverse_evidence_over_duplicate(self):
        self.build([chunk('a1', 'a', '예산 분석 설계 검토'), chunk('a2', 'a', '예산 분석 설계 검토'),
                    chunk('b1', 'b', '별도 제출 서류 등록')], min_df=1)
        ranks = [(1, 3), (2, 2), (3, 1)]
        with patch.object(self.index, '_bm25_candidates', return_value=ranks), \
             patch.object(self.index, '_char_candidates', return_value=ranks):
            hits = self.index.search('예산 제출', method='mmr', top_k=2, mmr_lambda=.2)
        self.assertEqual([r['chunk_id'] for r in hits], ['a1', 'b1'])
        self.assertIn('mmr', hits[0]['score_kind'])
        self.assertIn('rrf_score', hits[0])

    def test_corrupt_artifact_is_rejected_then_explicit_build_recovers(self):
        rows = [chunk('a1', 'a', '예산 10원')]
        self.build(rows)
        self.index.close()
        with (self.path / 'char_idf.npy').open('ab') as handle:
            handle.write(b'changed')
        with self.assertRaises(RuntimeError):
            LocalSearchIndex(self.path).search('예산')
        result = self.build(rows)
        self.assertFalse(result['cache_reused'])
        self.assertEqual(self.index.search('예산')[0]['chunk_id'], 'a1')

    def test_corrupt_config_cannot_load_with_old_vector_metadata(self):
        self.build([chunk('a1', 'a', '예산 10원')])
        self.index.close()
        path = self.path / 'index_meta.json'
        value = json.loads(path.read_text())
        value['config']['ngram_range'] = [3, 5]
        path.write_text(json.dumps(value))
        with self.assertRaises(RuntimeError):
            self.index.search('예산')

    def test_empty_punctuation_and_no_match_queries_return_empty(self):
        self.build([chunk('a1', 'a', '예산 10원')])
        for query in ['', '   ', '!!!']:
            self.assertEqual(self.index.search(query), [])
        self.assertEqual(self.index.search('zxqwplmn987654321', method='bm25'), [])
        self.assertEqual(self.index.search('예산', top_k=0), [])

    def test_invalid_config_and_mixed_source_identity_are_rejected(self):
        for config in [{'max_features': 0}, {'ngram_range': [4, 2]}, {'bm25_weight': 0, 'char_tfidf_weight': 0}, {'unknown': True}]:
            with self.assertRaises(ValueError):
                self.index.build([chunk('a1', 'a', '예산')], config)
        with self.assertRaises(ValueError):
            self.build([chunk('a1', 'a', '예산'), chunk('a1', 'b', '기간')])
        with self.assertRaises(ValueError):
            self.build([chunk('a1', 'a', '예산'), chunk('a2', 'a', '기간', sha256='f' * 64)])
        with self.assertRaises(ValueError):
            self.build([chunk('a1', 'a', '예산', canonical_doc_id='other')])
        with self.assertRaises(ValueError):
            self.index.build([])

    def test_scores_are_labels_not_confidence_and_original_metadata_remains(self):
        source = chunk('a1', 'a', '사업예산 10원', quality_flags=['PDF_TABLES_HEURISTIC'])
        self.build([source])
        for method in ['bm25', 'char_tfidf', 'rrf', 'mmr']:
            hit = self.index.search('예산', method=method)[0]
            self.assertFalse(hit['score_is_confidence'])
            self.assertIn('not_confidence', hit['score_kind'])
            self.assertEqual(hit['source_spans'], source['source_spans'])
            self.assertEqual(hit['quality_flags'], ['PDF_TABLES_HEURISTIC'])

    def test_build_and_search_require_no_network(self):
        with patch.object(socket, 'socket', side_effect=AssertionError('network must not be used')):
            self.build([chunk('a1', 'a', '예산 10원')])
            self.assertEqual(self.index.search('예산')[0]['chunk_id'], 'a1')


if __name__ == '__main__':
    unittest.main()
