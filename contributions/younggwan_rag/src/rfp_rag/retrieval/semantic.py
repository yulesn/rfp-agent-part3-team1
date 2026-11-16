"""Opt-in OpenAI embeddings beside the unchanged lexical search baseline.

Opening an index never imports the SDK or makes a request. Document embedding
requires ``live=True``; query embedding requires ``allow_query_api=True``.
Write-ahead reservations prevent automatic rebilling of interrupted build batches.
All saved vectors bind to the exact ordered source chunks and lexical index.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

import numpy as np

from .retrieval import _hash_file, _json_bytes


VERSION = 'rfp-semantic-index-v1'
DEFAULT_SEMANTIC_CONFIG = {
    'enabled': False, 'model': 'text-embedding-3-large', 'dimensions': 3072,
    'batch_size': 32, 'max_build_calls': 1000, 'max_build_input_bytes': 100000000,
    'max_query_calls': 64, 'max_query_input_bytes': 256000, 'weight': 1.0,
}


def normalize_semantic_config(config=None):
    supplied = dict(config or {})
    if set(supplied) - set(DEFAULT_SEMANTIC_CONFIG):
        raise ValueError('지원하지 않는 의미 검색 설정입니다.')
    value = {**DEFAULT_SEMANTIC_CONFIG, **supplied}
    if type(value['enabled']) is not bool:
        raise ValueError('semantic.enabled는 true/false여야 합니다.')
    if value['model'] != 'text-embedding-3-large':
        raise ValueError('이 프로필의 임베딩 모델은 text-embedding-3-large입니다.')
    for key in ('dimensions', 'batch_size', 'max_build_calls', 'max_build_input_bytes',
                'max_query_calls', 'max_query_input_bytes'):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(key + ' must be a positive integer.')
    if value['dimensions'] > 3072 or value['batch_size'] > 32:
        raise ValueError('dimensions <= 3072, batch_size <= 32로 지정하세요.')
    if type(value['weight']) not in (int, float) or not math.isfinite(value['weight']) or value['weight'] <= 0:
        raise ValueError('semantic.weight must be finite and positive.')
    return value


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _atomic_array(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def _lock(path):
    """Release OS locks on process exit, including Windows development hosts."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        handle.seek(0, 2)
        if not handle.tell():
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError('다른 의미 검색 작업이 진행 중입니다.') from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _binding(chunks, lexical, config):
    rows = [{'chunk_id': c['chunk_id'], 'doc_id': c['doc_id'], 'sha256': c['sha256'],
             'text_sha256': hashlib.sha256(c['text'].encode('utf-8')).hexdigest()} for c in chunks]
    if len({c['chunk_id'] for c in rows}) != len(rows) or not rows:
        raise ValueError('의미 검색 청크 ID가 중복되었거나 비어 있습니다.')
    meta = lexical.get_metadata()
    lexical._ensure_loaded()
    if [c['chunk_id'] for c in rows] != lexical._chunk_ids:
        raise ValueError('의미 검색 청크 순서가 로컬 인덱스와 다릅니다.')
    return {'version': VERSION, 'model': config['model'], 'dimensions': config['dimensions'],
            'config': config, 'lexical_index_fingerprint': meta['index_fingerprint'],
            'chunks_fingerprint': meta['chunks_fingerprint'], 'rows': rows,
            'semantic_module_sha256': _hash_file(__file__)}


def _array(value, count, dimensions):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (count, dimensions) or not np.isfinite(value).all():
        raise ValueError('임베딩 응답의 차원, 개수 또는 유한성 검사가 실패했습니다.')
    norms = np.linalg.norm(value.astype(np.float64), axis=1)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError('임베딩에 유효하지 않은 0 벡터가 있습니다.')
    return (value / norms[:, None]).astype(np.float32)


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _client(client):
    if client is not None:
        return client
    if not os.environ.get('OPENAI_API_KEY', '').strip():
        raise ValueError('의미 검색 API 실행에는 OPENAI_API_KEY가 필요합니다.')
    from openai import OpenAI
    return OpenAI(max_retries=0, timeout=60.0)


def _response(client, texts, config):
    response = client.embeddings.create(model=config['model'], input=texts,
                                        dimensions=config['dimensions'], encoding_format='float')
    if _get(response, 'model', config['model']) != config['model']:
        raise ValueError('임베딩 응답 모델이 요청 모델과 다릅니다.')
    data = _get(response, 'data')
    if not isinstance(data, list) or len(data) != len(texts):
        raise ValueError('임베딩 응답 개수가 요청과 다릅니다.')
    indices = [_get(row, 'index') for row in data]
    if any(type(i) is not int for i in indices) or sorted(indices) != list(range(len(texts))):
        raise ValueError('임베딩 응답의 순서 식별자가 올바르지 않습니다.')
    ordered = sorted(data, key=lambda row: _get(row, 'index'))
    vectors = _array([_get(row, 'embedding') for row in ordered], len(texts), config['dimensions'])
    tokens = _get(_get(response, 'usage'), 'prompt_tokens')
    if tokens is not None and (type(tokens) is not int or tokens < 0):
        raise ValueError('임베딩 사용량이 올바르지 않습니다.')
    return vectors, tokens


def _safe_failure(error):
    status = getattr(error, 'status_code', None)
    return {'status': 'error', 'http_status': status if type(status) is int else None,
            'message': '임베딩 요청 또는 응답 검사에 실패했습니다. 자동 재시도하지 않습니다.'}


def _load_index(root, chunks, lexical, config):
    folder = Path(root) / 'semantic'
    try:
        meta = _read(folder / 'metadata.json')
        expected = _binding(chunks, lexical, config)
        if meta.get('binding') != expected or meta.get('fingerprint') != _digest({k: v for k, v in meta.items() if k != 'fingerprint'}):
            raise ValueError('의미 검색 인덱스의 원문·모델·설정 바인딩이 다릅니다.')
        if _hash_file(folder / 'vectors.npy') != meta['vectors_sha256']:
            raise ValueError('의미 검색 벡터 파일이 변경되었습니다.')
        vectors = np.load(folder / 'vectors.npy', allow_pickle=False)
        checked = _array(vectors, len(chunks), config['dimensions'])
        if not np.allclose(checked, vectors, rtol=1e-5, atol=1e-6):
            raise ValueError('의미 검색 벡터가 정규화되지 않았습니다.')
        return meta, vectors
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError('의미 검색 인덱스가 없거나 원문·모델·파일이 다릅니다. '
                         'rfp-rag embed --workspace <D 경로> --live 로 구축하세요.') from error


def build_semantic_index(root, chunks, lexical_index, config, *, live=False, client=None, progress=None):
    """Embed exact chunk text once, resuming only already completed batches.

    An attempted batch without a valid saved result is never resent. A caller
    must deliberately choose a new workspace after inspecting that failure.
    Synthetic clients are recorded separately from real OpenAI usage.
    """
    config = normalize_semantic_config(config); chunks = list(chunks); root = Path(root)
    if progress is not None and not callable(progress):
        raise TypeError('progress must be callable or None.')
    if type(live) is not bool or not config['enabled']:
        raise ValueError('semantic.enabled=true인 D 프로필이 필요합니다.')
    if (root / 'semantic/metadata.json').exists():
        meta, _ = _load_index(root, chunks, lexical_index, config)
        return {**meta, 'reused': True, 'api_calls_this_run': 0}
    if not live:
        raise ValueError('문서 임베딩은 --live를 명시해야 실행됩니다.')
    texts = [c['text'] for c in chunks]
    # UTF-8 bytes provide a conservative token upper bound. Batches stay below
    # 32 * 8191 bytes and therefore below the API's aggregate token limit.
    if any(not text.strip() or len(text.encode('utf-8')) > 8191 for text in texts):
        raise ValueError('임베딩 입력은 비어 있지 않은 8191 UTF-8 바이트 이하 청크여야 합니다.')
    count = math.ceil(len(texts) / config['batch_size'])
    total_bytes = sum(len(text.encode('utf-8')) for text in texts)
    if count > config['max_build_calls'] or total_bytes > config['max_build_input_bytes']:
        raise ValueError('문서 임베딩 호출 수 또는 입력 바이트 예산을 초과합니다.')
    binding = _binding(chunks, lexical_index, config)
    plan = {'binding': binding, 'input_bytes': total_bytes, 'planned_calls': count,
            'execution_origin': 'mock' if client is not None else 'live_openai'}
    plan['fingerprint'] = _digest(plan)
    ledger = root / 'semantic_build'; ledger.mkdir(parents=True, exist_ok=True)
    this_run = 0
    with _lock(ledger / '.lock'):
        plan_path = ledger / 'plan.json'
        if plan_path.exists():
            if _read(plan_path) != plan:
                raise ValueError('기존 문서 임베딩 계획과 다릅니다. 새 workspace를 지정하세요.')
        else:
            _write_new(plan_path, plan)
        sdk = None; arrays = []; reported_tokens = 0; unknown_usage = 0
        for batch_number, start in enumerate(range(0, len(texts), config['batch_size'])):
            batch = texts[start:start + config['batch_size']]
            stem = f'{batch_number:06d}'
            attempt_path = ledger / (stem + '.attempt.json')
            result_path = ledger / (stem + '.result.json')
            vector_path = ledger / (stem + '.npy')
            attempt = {'batch': batch_number, 'plan_fingerprint': plan['fingerprint'],
                       'input_fingerprint': _digest(batch),
                       'input_bytes': sum(len(t.encode('utf-8')) for t in batch)}
            reused_batch = attempt_path.exists()
            if attempt_path.exists():
                if _read(attempt_path) != attempt or not result_path.exists():
                    raise ValueError('결과가 없거나 실패한 임베딩 시도가 있습니다. 자동 재요청하지 않습니다. 새 workspace를 지정하세요.')
                result = _read(result_path)
                if result.get('status') != 'completed' or result.get('attempt_fingerprint') != _digest(attempt) or not vector_path.exists():
                    raise ValueError('실패한 임베딩 배치는 자동 재시도하지 않습니다. 새 workspace를 지정하세요.')
                if _hash_file(vector_path) != result.get('vectors_sha256'):
                    raise ValueError('저장된 임베딩 배치가 변경되었습니다.')
                array = _array(np.load(vector_path, allow_pickle=False), len(batch), config['dimensions'])
            else:
                sdk = sdk or _client(client)
                _write_new(attempt_path, attempt)
                this_run += 1
                try:
                    array, tokens = _response(sdk, batch, config)
                    _atomic_array(vector_path, array)
                    result = {'status': 'completed', 'attempt_fingerprint': _digest(attempt),
                              'vectors_sha256': _hash_file(vector_path), 'reported_input_tokens': tokens}
                    _write_new(result_path, result)
                except Exception as error:
                    if not result_path.exists():
                        _write_new(result_path, {**_safe_failure(error), 'attempt_fingerprint': _digest(attempt)})
                    raise RuntimeError('문서 임베딩 요청 또는 응답 검사에 실패했습니다. '
                                       'semantic_build 기록을 확인하세요. 자동 재시도하지 않습니다.') from None
            tokens = result.get('reported_input_tokens')
            if tokens is None:
                unknown_usage += 1
            else:
                reported_tokens += tokens
            arrays.append(array)
            if progress is not None:
                progress({'completed_batches': batch_number + 1, 'total_batches': count,
                          'reused_batch': reused_batch})
        folder = root / 'semantic'; folder.mkdir(exist_ok=True)
        _atomic_array(folder / 'vectors.npy', np.concatenate(arrays, axis=0))
        meta = {'version': VERSION, 'binding': binding, 'chunk_count': len(chunks),
                'vectors_sha256': _hash_file(folder / 'vectors.npy'),
                'usage': {'api_calls': count, 'live_api_calls': count if client is None else 0,
                          'mock_calls': count if client is not None else 0, 'input_bytes': total_bytes,
                          'reported_input_tokens': reported_tokens, 'usage_unknown_attempts': unknown_usage},
                'semantic_accuracy': None, 'independent_quality_verified': False}
        meta['fingerprint'] = _digest(meta)
        _atomic_json(folder / 'metadata.json', meta)
    return {**meta, 'reused': False, 'api_calls_this_run': this_run}


class SemanticSearchIndex:
    """Three-component scoped RRF using immutable source chunks and dense vectors."""
    def __init__(self, lexical_index, root, chunks, config, *, allow_query_api=False, client=None):
        if type(allow_query_api) is not bool:
            raise ValueError('allow_query_api는 true/false여야 합니다.')
        self.lexical = lexical_index; self.root = Path(root); self.chunks = list(chunks)
        self.config = normalize_semantic_config(config)
        self.allow_query_api = allow_query_api; self.client = client; self._sdk = None
        self._cache = OrderedDict()
        self._usage = {key: 0 for key in ('api_calls', 'live_api_calls', 'mock_calls', 'failed_attempts',
                                        'input_bytes', 'reported_input_tokens', 'usage_unknown_attempts', 'cache_hits')}
        self.metadata, self.vectors = _load_index(root, self.chunks, lexical_index, self.config)
        self.by_id = {c['chunk_id']: c for c in self.chunks}
        self.positions = {c['chunk_id']: i for i, c in enumerate(self.chunks)}

    def __getattr__(self, name):
        return getattr(self.lexical, name)

    def close(self):
        self._cache.clear(); self.lexical.close()

    def usage_report(self):
        return {**copy.deepcopy(self._usage), 'attempted_input_bytes': self._usage['input_bytes'],
                'max_calls': self.config['max_query_calls'], 'allow_query_api': self.allow_query_api}

    def get_metadata(self):
        value = self.lexical.get_metadata()
        value['lexical_index_fingerprint'] = value['index_fingerprint']
        value['index_fingerprint'] = _digest([value['index_fingerprint'], self.metadata['fingerprint']])
        value['semantic_index_fingerprint'] = self.metadata['fingerprint']
        value['semantic_embedding_model'] = self.config['model']
        value['semantic_embedding_dimensions'] = self.config['dimensions']
        value['semantic_query_api_enabled'] = self.allow_query_api
        value['semantic_scores_are_confidence'] = False
        return value

    def _query_vector(self, query):
        if not self.allow_query_api:
            raise ValueError('D 프로필의 의미 검색에는 --live-search(serve) 또는 --live(search/ask/validate)가 필요합니다.')
        key = _digest([self.config['model'], self.config['dimensions'], query])
        if key in self._cache:
            self._usage['cache_hits'] += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        sdk = self._sdk or _client(self.client)
        ledger = self.root / 'semantic_outputs'; ledger.mkdir(parents=True, exist_ok=True)
        input_bytes = len(query.encode('utf-8'))
        if input_bytes > 8191:
            raise ValueError('의미 검색 질문은 8191 UTF-8 바이트 이하여야 합니다.')
        with _lock(ledger / '.lock'):
            paths = list(ledger.glob('*.attempt.json'))
            attempts = [_read(path) for path in paths]
            for path, previous in zip(paths, attempts):
                result_path = path.with_name(path.name.replace('.attempt.json', '.result.json'))
                if previous.get('query_fingerprint') == key and (
                        not result_path.exists() or _read(result_path).get('status') != 'completed'):
                    raise ValueError('결과가 없거나 실패한 동일 질문 임베딩을 자동 재요청하지 않습니다. '
                                     'semantic_outputs 기록을 확인하고 새 workspace를 지정하세요.')
            if len(attempts) >= self.config['max_query_calls'] or sum(r['input_bytes'] for r in attempts) + input_bytes > self.config['max_query_input_bytes']:
                raise ValueError('의미 검색 질문 임베딩 예산을 초과했습니다. 새 workspace 또는 예산 설정을 사용하세요.')
            call = uuid.uuid4().hex
            attempt = {'query_fingerprint': key, 'input_bytes': input_bytes,
                       'execution_origin': 'mock' if self.client is not None else 'live_openai',
                       'model': self.config['model'], 'dimensions': self.config['dimensions']}
            _write_new(ledger / (call + '.attempt.json'), attempt)
        self._sdk = sdk
        self._usage['api_calls'] += 1
        self._usage['mock_calls' if self.client is not None else 'live_api_calls'] += 1
        self._usage['input_bytes'] += input_bytes
        try:
            vector, tokens = _response(sdk, [query], self.config)
            result = {'status': 'completed', 'attempt_fingerprint': _digest(attempt),
                      'reported_input_tokens': tokens}
            _write_new(ledger / (call + '.result.json'), result)
        except Exception as error:
            self._usage['failed_attempts'] += 1
            self._usage['usage_unknown_attempts'] += 1
            if not (ledger / (call + '.result.json')).exists():
                _write_new(ledger / (call + '.result.json'), {**_safe_failure(error), 'attempt_fingerprint': _digest(attempt)})
            raise RuntimeError('질문 임베딩 요청 또는 응답 검사에 실패했습니다. 자동 재시도하지 않습니다.') from None
        if tokens is None:
            self._usage['usage_unknown_attempts'] += 1
        else:
            self._usage['reported_input_tokens'] += tokens
        self._cache[key] = vector[0]
        if len(self._cache) > 64:
            self._cache.popitem(last=False)
        return vector[0]

    def search(self, query, top_k=5, allowed_doc_ids=None, method='rrf', fetch_k=30, mmr_lambda=.75):
        if method in {'bm25', 'char_tfidf'}:
            return self.lexical.search(query, top_k=top_k, allowed_doc_ids=allowed_doc_ids,
                                       method=method, fetch_k=fetch_k, mmr_lambda=mmr_lambda)
        if not isinstance(query, str):
            raise TypeError('Search query must be a string.')
        if type(top_k) is not int or top_k < 0 or type(fetch_k) is not int or fetch_k < 1:
            raise ValueError('top_k >= 0, fetch_k >= 1 정수로 지정하세요.')
        if method not in {'rrf', 'mmr', 'semantic'}:
            raise ValueError('Unknown search method: ' + str(method))
        if type(mmr_lambda) not in (int, float) or not math.isfinite(mmr_lambda) or not 0 <= mmr_lambda <= 1:
            raise ValueError('mmr_lambda must be between 0 and 1.')
        scope = None if allowed_doc_ids is None else self.resolve_doc_ids(allowed_doc_ids)
        if scope == [] or top_k == 0 or not query.strip():
            return []
        lexical_meta = self.lexical.get_metadata()
        if len(query) > lexical_meta['config']['query_max_chars']:
            raise ValueError('Search query is longer than query_max_chars.')
        indices = [i for i, chunk in enumerate(self.chunks) if scope is None or chunk['doc_id'] in scope]
        if not indices:
            return []
        vector = self._query_vector(query)
        similarities = self.vectors[indices] @ vector
        fetch_k = max(top_k, fetch_k)
        dense = sorted([(self.chunks[i]['chunk_id'], float(score)) for i, score in zip(indices, similarities)
                        if score > 0], key=lambda pair: (-pair[1], pair[0]))[:fetch_k]
        components = [('semantic', dense, self.config['weight'])]
        config = lexical_meta['config']
        if method != 'semantic':
            for name, weight in [('bm25', config['bm25_weight']), ('char_tfidf', config['char_tfidf_weight'])]:
                hits = self.lexical.search(query, top_k=fetch_k, allowed_doc_ids=scope, method=name,
                                           fetch_k=fetch_k, mmr_lambda=mmr_lambda)
                components.append((name, [(h['chunk_id'], h['score']) for h in hits], weight))
        scores = {}; details = {}
        for name, ranking, weight in components:
            if not weight:
                continue
            for rank, (chunk_id, score) in enumerate(ranking, 1):
                scores[chunk_id] = scores.get(chunk_id, 0) + weight / (config['rrf_k'] + rank)
                value = details.setdefault(chunk_id, {'component_ranks': {}, 'component_scores': {}})
                value['component_ranks'][name] = rank; value['component_scores'][name] = score
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        if method == 'mmr' and ranked:
            pool = ranked[:fetch_k]; maximum = pool[0][1]; chosen = []; remaining = list(pool)
            while remaining and len(chosen) < top_k:
                def objective(pair):
                    redundancy = max((float(self.vectors[self.positions[pair[0]]] @ self.vectors[self.positions[c[0]]])
                                      for c in chosen), default=0)
                    return mmr_lambda * pair[1] / maximum - (1 - mmr_lambda) * max(0, redundancy)
                item = min(remaining, key=lambda pair: (-objective(pair), pair[0]))
                chosen.append((item[0], objective(item))); remaining.remove(item)
            ranked = chosen
        hits = []
        for rank, (chunk_id, score) in enumerate(ranked[:top_k], 1):
            hit = copy.deepcopy(self.by_id[chunk_id])
            hit.update(score=float(score), rank=rank, method='hybrid_' + method,
                       score_kind='hybrid_reciprocal_rank_not_confidence', score_is_confidence=False,
                       semantic_embedding_model=self.config['model'], **details[chunk_id])
            hits.append(hit)
        return hits
