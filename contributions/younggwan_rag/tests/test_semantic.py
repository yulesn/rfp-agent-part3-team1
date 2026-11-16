"""Semantic search correctness and spending boundaries on synthetic vectors."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rfp_rag.retrieval.retrieval import LocalSearchIndex
from rfp_rag.retrieval.semantic import (DEFAULT_SEMANTIC_CONFIG, SemanticSearchIndex,
    build_semantic_index, normalize_semantic_config)


TEXTS = ['초록색 열매 재배 조건', '매월 시스템 정기점검 제공', '다른 기관의 유지보수 의무']
QUERY = '서버 돌봄 주기는?'


class Embeddings:
    def __init__(self, bad=None):
        self.embeddings = self; self.calls = []; self.bad = bad

    def create(self, **body):
        self.calls.append(deepcopy(body))
        if self.bad == 'failure':
            error = RuntimeError('SECRET_API_KEY_OR_PROVIDER_BODY'); error.status_code = 429; raise error
        vectors = {TEXTS[0]: [0, 1, 0], TEXTS[1]: [1, 0, 0], TEXTS[2]: [1, 0, .01], QUERY: [1, 0, 0]}
        rows = [SimpleNamespace(index=i, embedding=vectors.get(t, [0, 0, 1])) for i, t in enumerate(body['input'])]
        if self.bad == 'nan':
            rows[0].embedding = [float('nan'), 0, 0]
        elif self.bad == 'zero':
            rows[0].embedding = [0, 0, 0]
        elif self.bad == 'dimension':
            rows[0].embedding = [1, 0]
        elif self.bad == 'duplicate_index':
            rows[-1].index = rows[0].index
        elif self.bad == 'missing_row':
            rows.pop()
        return SimpleNamespace(model='wrong-model' if self.bad == 'model' else body['model'],
                               data=list(reversed(rows)), usage=SimpleNamespace(prompt_tokens=17))


class SemanticTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.chunks = [dict(chunk_id=f'c{i}', doc_id='A' if i < 2 else 'B',
                            canonical_doc_id='A' if i < 2 else 'B',
                            alias_doc_ids=['A', 'A_alias'] if i < 2 else ['B'],
                            sha256=hashlib.sha256(('A' if i < 2 else 'B').encode()).hexdigest(),
                            text=text, source_spans=[{'preserved': i}], locator={'page': i + 1})
                       for i, text in enumerate(TEXTS)]
        self.lexical = LocalSearchIndex(self.root / 'index')
        self.lexical.build(self.chunks, {'min_df': 1}); self.addCleanup(self.lexical.close)
        self.config = normalize_semantic_config({'enabled': True, 'dimensions': 3, 'batch_size': 2})
        self.client = Embeddings()
        blocked = patch('socket.socket.connect', side_effect=AssertionError('paid network prohibited'))
        blocked.start(); self.addCleanup(blocked.stop)

    def build(self, **kwargs):
        return build_semantic_index(self.root, self.chunks, self.lexical, self.config,
                                    live=True, client=kwargs.pop('client', self.client), **kwargs)

    def open(self, **kwargs):
        return SemanticSearchIndex(self.lexical, self.root, self.chunks, self.config,
                                   client=kwargs.pop('client', self.client), **kwargs)

    def test_build_batches_correct_model_input_dimensions_and_original_text(self):
        report = self.build()
        self.assertEqual(report['usage']['api_calls'], 2)
        self.assertEqual(report['usage']['mock_calls'], 2)
        self.assertEqual(report['usage']['live_api_calls'], 0)
        self.assertEqual(report['usage']['reported_input_tokens'], 34)
        self.assertEqual([t for c in self.client.calls for t in c['input']], TEXTS)
        self.assertTrue(all(c['model'] == 'text-embedding-3-large' and c['dimensions'] == 3
                            and c['encoding_format'] == 'float' for c in self.client.calls))

    def test_semantic_synonym_retrieval_and_source_provenance(self):
        self.build(); index = self.open(allow_query_api=True)
        result = index.search(QUERY, allowed_doc_ids=['A_alias'], top_k=1)
        self.assertEqual(result[0]['chunk_id'], 'c1')
        self.assertEqual(result[0]['text'], TEXTS[1])
        self.assertEqual(result[0]['source_spans'], self.chunks[1]['source_spans'])
        self.assertEqual(result[0]['locator'], self.chunks[1]['locator'])
        self.assertEqual(result[0]['component_ranks']['semantic'], 1)
        self.assertFalse(result[0]['score_is_confidence'])

    def test_dense_scope_before_top_k(self):
        self.build(); index = self.open(allow_query_api=True)
        # Out-of-scope rows cannot consume the top-k pool.
        hits = index.search(QUERY, allowed_doc_ids=['B'], top_k=1, fetch_k=1, method='semantic')
        self.assertEqual([h['doc_id'] for h in hits], ['B'])
        self.assertEqual(index.search(QUERY, allowed_doc_ids=['A'], top_k=1,
                                     fetch_k=1, method='semantic')[0]['chunk_id'], 'c1')

    def test_open_and_disallowed_query_make_no_calls(self):
        self.build(); before = len(self.client.calls)
        with patch.dict(os.environ, {}, clear=True):
            index = self.open()
            self.assertEqual(index.get_metadata()['semantic_embedding_model'], 'text-embedding-3-large')
            with self.assertRaisesRegex(ValueError, 'live-search'):
                index.search(QUERY, allowed_doc_ids=['A'])
        self.assertEqual(len(self.client.calls), before)

    def test_empty_scope_unknown_id_and_bad_queries_never_call(self):
        self.build(); index = self.open(allow_query_api=True); before = len(self.client.calls)
        for kwargs in ({'allowed_doc_ids': []}, {'top_k': 0}, {}):
            query = '' if not kwargs else QUERY
            self.assertEqual(index.search(query, **kwargs), [])
        for kwargs in ({'allowed_doc_ids': ['unknown']}, {'top_k': -1}, {'fetch_k': 0},
                       {'method': 'bad'}, {'mmr_lambda': float('nan')}):
            with self.assertRaises(ValueError):
                index.search(QUERY, **kwargs)
        with self.assertRaises(ValueError):
            index.search('a' * 4001)
        self.assertEqual(len(self.client.calls), before)

    def test_explicit_lexical_method_preserves_baseline_without_api(self):
        self.build(); index = self.open(); before = len(self.client.calls)
        for method in ('bm25', 'char_tfidf'):
            self.assertEqual(index.search('시스템', method=method, allowed_doc_ids=['A']),
                             self.lexical.search('시스템', method=method, allowed_doc_ids=['A']))
        self.assertEqual(len(self.client.calls), before)

    def test_query_cache_and_separate_usage(self):
        self.build(); index = self.open(allow_query_api=True)
        index.search(QUERY, allowed_doc_ids=['A']); index.search(QUERY, allowed_doc_ids=['B'])
        usage = index.usage_report()
        self.assertEqual(usage['api_calls'], 1); self.assertEqual(usage['mock_calls'], 1)
        self.assertEqual(usage['cache_hits'], 1); self.assertEqual(usage['live_api_calls'], 0)
        self.assertEqual(usage['attempted_input_bytes'], len(QUERY.encode()))
        self.assertEqual(usage['reported_input_tokens'], 17)

    def test_completed_build_reuse_has_zero_new_calls_and_offline_allowed(self):
        self.build(); before = len(self.client.calls)
        with patch.dict(os.environ, {}, clear=True):
            report = build_semantic_index(self.root, self.chunks, self.lexical, self.config)
        self.assertTrue(report['reused']); self.assertEqual(report['api_calls_this_run'], 0)
        self.assertEqual(len(self.client.calls), before)

    def test_resume_completed_batches_without_rebilling(self):
        self.build(); (self.root / 'semantic/metadata.json').unlink()
        before = len(self.client.calls); result = self.build()
        self.assertEqual(result['api_calls_this_run'], 0)
        self.assertEqual(len(self.client.calls), before)
        self.open()

    def test_progress_callback_can_interrupt_without_rebilling_saved_batch(self):
        reports = []
        def stop_after_first(report):
            reports.append(report)
            raise RuntimeError('synthetic UI interruption')
        with self.assertRaisesRegex(RuntimeError, 'UI interruption'):
            self.build(progress=stop_after_first)
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(reports, [{'completed_batches': 1, 'total_batches': 2, 'reused_batch': False}])
        reports.clear(); self.build(progress=reports.append)
        self.assertEqual(len(self.client.calls), 2)
        self.assertEqual(reports, [{'completed_batches': 1, 'total_batches': 2, 'reused_batch': True},
                                   {'completed_batches': 2, 'total_batches': 2, 'reused_batch': False}])

    def test_failed_build_does_not_retry_and_provider_body_is_redacted(self):
        failing = Embeddings('failure')
        with self.assertRaisesRegex(RuntimeError, '자동 재시도'):
            self.build(client=failing)
        self.assertEqual(len(failing.calls), 1)
        with self.assertRaises(ValueError):
            self.build()
        self.assertEqual(self.client.calls, [])
        self.assertNotIn('SECRET_API_KEY_OR_PROVIDER_BODY', '\n'.join(
            p.read_text() for p in (self.root / 'semantic_build').glob('*.json')))

    def test_unfinished_build_reservation_never_retries(self):
        self.build(); (self.root / 'semantic/metadata.json').unlink()
        (self.root / 'semantic_build/000000.result.json').unlink()
        before = len(self.client.calls)
        with self.assertRaisesRegex(ValueError, '자동 재요청'):
            self.build()
        self.assertEqual(len(self.client.calls), before)

    def test_build_budget_rejected_before_sdk_or_reservation(self):
        for key in ('max_build_calls', 'max_build_input_bytes'):
            config = {**self.config, key: 1}
            with self.assertRaises(ValueError):
                build_semantic_index(self.root, self.chunks, self.lexical, config, live=True, client=self.client)
        self.assertFalse((self.root / 'semantic_build').exists()); self.assertEqual(self.client.calls, [])

    def test_disabled_build_and_missing_live_never_call(self):
        for live, config in [(False, self.config), (True, {**self.config, 'enabled': False})]:
            with self.assertRaises(ValueError):
                build_semantic_index(self.root, self.chunks, self.lexical, config, live=live, client=self.client)
        self.assertEqual(self.client.calls, [])

    def test_wrong_order_source_model_and_dimension_rejected(self):
        self.build()
        changed = deepcopy(self.chunks); changed[0]['text'] += ' changed'
        for chunks, config in [(list(reversed(self.chunks)), self.config), (changed, self.config),
                               (self.chunks, {**self.config, 'dimensions': 2})]:
            with self.assertRaises(ValueError):
                SemanticSearchIndex(self.lexical, self.root, chunks, config)
        for config in ({'model': 'other'}, {'dimensions': 3073}, {'weight': 0}, {'enabled': 'true'},
                       {'max_query_calls': True}, {'unsupported': 1}):
            with self.assertRaises(ValueError):
                normalize_semantic_config(config)

    def test_damaged_vector_file_fails_closed(self):
        self.build(); path = self.root / 'semantic/vectors.npy'
        path.write_bytes(path.read_bytes() + b'damaged')
        with self.assertRaises(ValueError):
            self.open()

    def test_bad_responses_never_publish_index(self):
        for bad in ('nan', 'zero', 'dimension', 'duplicate_index', 'missing_row', 'model'):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as folder:
                failing = Embeddings(bad)
                with self.assertRaises(RuntimeError):
                    build_semantic_index(folder, self.chunks, self.lexical, self.config, live=True, client=failing)
                self.assertFalse((Path(folder) / 'semantic/metadata.json').exists())
                self.assertEqual(len(failing.calls), 1)

    def test_query_budget_persists_across_index_instances(self):
        self.config['max_query_calls'] = 1
        self.build(); first = self.open(allow_query_api=True)
        first.search(QUERY, allowed_doc_ids=['A'])
        second = self.open(allow_query_api=True); before = len(self.client.calls)
        with self.assertRaisesRegex(ValueError, '예산'):
            second.search('다른 질문', allowed_doc_ids=['A'])
        self.assertEqual(len(self.client.calls), before)

    def test_failed_query_is_not_automatically_retried_on_reopen(self):
        self.build(); failing = Embeddings('failure'); index = self.open(client=failing, allow_query_api=True)
        with self.assertRaisesRegex(RuntimeError, '자동 재시도'):
            index.search(QUERY, allowed_doc_ids=['A'])
        reopened = self.open(allow_query_api=True); before = len(self.client.calls)
        with self.assertRaisesRegex(ValueError, '자동 재요청'):
            reopened.search(QUERY, allowed_doc_ids=['A'])
        self.assertEqual(len(self.client.calls), before)
        self.assertEqual(index.usage_report()['failed_attempts'], 1)

    def test_mmr_ranking_is_deterministic_and_scoped(self):
        self.build(); index = self.open(allow_query_api=True)
        left = index.search(QUERY, allowed_doc_ids=['A'], method='mmr', top_k=2)
        right = index.search(QUERY, allowed_doc_ids=['A'], method='mmr', top_k=2)
        self.assertEqual(left, right)
        self.assertTrue(all(h['doc_id'] == 'A' for h in left))
        self.assertTrue(all(np.isfinite(h['score']) for h in left))


if __name__ == '__main__':
    unittest.main()
