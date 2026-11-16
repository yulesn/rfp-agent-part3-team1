"""Local lexical retrieval: SQLite FTS5 BM25, character TF-IDF, RRF and MMR.

Character TF-IDF is a lexical vector baseline, not a semantic embedding model.
No evaluation files, API keys, network clients, model downloads or pickle loads
are used. Original chunk text is never changed by retrieval preprocessing.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import unicodedata

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


INDEX_SCHEMA = 'rfp-local-search-v1'
TOKENIZER_VERSION = 'ko_word_2_3gram_v1'
ARTIFACT_NAMES = ('chunks.sqlite3', 'char_vectors.npz', 'char_idf.npy', 'char_vocabulary.json')
DEFAULT_TERM_EXPANSIONS = {
    'budget': ['예산', '사업비', '용역비', '소요예산', '예정가격', '사업금액', '추정금액'],
    'duration': ['수행기간', '사업기간', '용역기간', '계약기간', '과업기간'],
    'submissiondeadline': ['제출기한', '마감', '접수기한', '제출마감', '접수마감'],
    'submissiondocuments': ['제출서류', '구비서류', '제출자료', '제안서류'],
    'submissionmethod': ['제출방법', '제출방식', '접수방법'],
    'qualification': ['참가자격', '입찰자격', '입찰참가', '제안자격'],
    'vat': ['부가세', '부가가치세', 'vat'],
}
DEFAULT_CONFIG = {
    'ngram_range': [2, 4], 'max_features': 100000, 'min_df': 2,
    'rrf_k': 60, 'bm25_weight': 1.0, 'char_tfidf_weight': 1.0,
    'fts_metadata_weight': 0.25, 'query_max_chars': 4000,
    'term_expansions': DEFAULT_TERM_EXPANSIONS,
}
_STOP_WORDS = {'알려줘', '알려주세요', '알려', '해줘', '해주세요', '무엇인가요', '무엇', '어떻게', '대해서'}
_SPACED_HANGUL = re.compile(r'(?<![가-힣])(?:[가-힣]\s+){2,}[가-힣](?![가-힣])')
_WORDS = re.compile(r'[가-힣]+|[a-z]+|\d+(?:\.\d+)?')


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _dependency_versions():
    return {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn')}


def normalize_retrieval_text(text):
    """Normalize only the search view; source strings and source offsets stay intact."""
    text = unicodedata.normalize('NFKC', str(text)).lower()
    text = _SPACED_HANGUL.sub(lambda match: re.sub(r'\s+', '', match.group(0)), text)
    text = re.sub(r'(?<=\d)[,，](?=\d)', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def _expansion_tags(normalized, config):
    compact = re.sub(r'\s+', '', normalized)
    return ['s' + category for category, alternatives in config['term_expansions'].items()
            if any(normalize_retrieval_text(term).replace(' ', '') in compact for term in alternatives)]


def tokenize_search(text, config=None):
    """Korean words plus 2/3-character subwords and declared fixed domain tags."""
    config = config or DEFAULT_CONFIG
    normalized = normalize_retrieval_text(text)
    terms = []
    for word in _WORDS.findall(normalized):
        if word in _STOP_WORDS:
            continue
        if word[0].isdigit():
            terms.append('n' + word.replace('.', 'd'))
        else:
            terms.append('w' + word)
            if re.fullmatch(r'[가-힣]+', word):
                for width in (2, 3):
                    terms.extend('g' + word[i:i + width] for i in range(len(word) - width + 1))
    terms.extend(_expansion_tags(normalized, config))
    return terms


def _char_view(text, config):
    normalized = normalize_retrieval_text(text)
    tags = _expansion_tags(normalized, config)
    return normalized + (' ' + ' '.join(tags) if tags else '')


def _normalize_config(config):
    supplied = dict(config or {})
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError('Unknown retrieval config fields: ' + ', '.join(sorted(unknown)))
    value = json.loads(json.dumps(DEFAULT_CONFIG))
    value.update(supplied)
    interval = value['ngram_range']
    if not isinstance(interval, (list, tuple)) or len(interval) != 2 or not all(isinstance(n, int) for n in interval) or not 1 <= interval[0] <= interval[1] <= 8:
        raise ValueError('ngram_range must be a positive increasing pair, up to 8.')
    value['ngram_range'] = list(interval)
    for key in ('max_features', 'min_df', 'query_max_chars'):
        if not isinstance(value[key], int) or isinstance(value[key], bool) or value[key] < 1:
            raise ValueError(key + ' must be a positive integer.')
    for key in ('rrf_k', 'bm25_weight', 'char_tfidf_weight', 'fts_metadata_weight'):
        if not isinstance(value[key], (int, float)) or not math.isfinite(value[key]) or value[key] < 0:
            raise ValueError(key + ' must be finite and nonnegative.')
    if not value['bm25_weight'] and not value['char_tfidf_weight']:
        raise ValueError('At least one RRF component weight must be positive.')
    expansions = value['term_expansions']
    if not isinstance(expansions, dict) or any(not re.fullmatch(r'[a-z]+', key) for key in expansions):
        raise ValueError('term_expansions must map lowercase Latin category names to term lists.')
    if any(not isinstance(terms, list) or not terms or any(not isinstance(t, str) or not t.strip() for t in terms) for terms in expansions.values()):
        raise ValueError('Each term expansion must contain nonempty string alternatives.')
    value['term_expansions'] = {key: expansions[key] for key in sorted(expansions)}
    return value


def _vectorizer(config, vocabulary=None, min_df=None):
    return TfidfVectorizer(analyzer='char_wb', ngram_range=tuple(config['ngram_range']),
                           max_features=config['max_features'], min_df=min_df if min_df is not None else config['min_df'],
                           lowercase=False, norm='l2', dtype=np.float32, vocabulary=vocabulary)


class LocalSearchIndex:
    """Persisted, content/version-checked local search index.

    ``allowed_doc_ids=None`` searches the indexed corpus. An empty scope returns
    no hits. Unknown IDs raise instead of silently widening the search scope.
    All component candidate pools apply the scope before ranking/truncation.
    """

    def __init__(self, index_dir):
        self.index_dir = Path(index_dir).resolve()
        self._connection = None
        self._meta = None
        self._matrix = None
        self._vectorizer = None
        self._row_doc_ids = None
        self._chunk_ids = None

    def close(self):
        if self._connection is not None:
            self._connection.close()
        self._connection = None
        self._meta = None
        self._matrix = None
        self._vectorizer = None
        self._row_doc_ids = None
        self._chunk_ids = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _load(self):
        self.close()
        metadata_path = self.index_dir / 'index_meta.json'
        try:
            meta = json.loads(metadata_path.read_text(encoding='utf-8'))
            if meta['schema_version'] != INDEX_SCHEMA or meta['tokenizer_version'] != TOKENIZER_VERSION:
                raise ValueError('Index schema/tokenizer version changed.')
            if meta['retrieval_module_sha256'] != _hash_file(__file__):
                raise ValueError('Retrieval implementation changed.')
            if meta['dependencies'] != _dependency_versions():
                raise ValueError('Retrieval dependency versions changed.')
            if hashlib.sha256(_json_bytes(meta['config'])).hexdigest() != meta['config_fingerprint']:
                raise ValueError('Index config fingerprint mismatch.')
            for name in ARTIFACT_NAMES:
                if _hash_file(self.index_dir / name) != meta['files_sha256'][name]:
                    raise ValueError('Index artifact fingerprint mismatch: ' + name)
            matrix = sparse.load_npz(self.index_dir / 'char_vectors.npz').tocsr()
            vocabulary = json.loads((self.index_dir / 'char_vocabulary.json').read_text(encoding='utf-8'))
            idf = np.load(self.index_dir / 'char_idf.npy', allow_pickle=False)
            if matrix.shape != (meta['chunk_count'], len(vocabulary)) or idf.shape != (len(vocabulary),):
                raise ValueError('Index matrix dimensions disagree with stored metadata.')
            if not np.isfinite(matrix.data).all() or not np.isfinite(idf).all():
                raise ValueError('Index vectors contain nonfinite values.')
            connection = sqlite3.connect(f'file:{self.index_dir / "chunks.sqlite3"}?mode=ro', uri=True)
            rows = connection.execute('SELECT row_id,chunk_id,doc_id FROM chunks ORDER BY row_id').fetchall()
            if len(rows) != meta['chunk_count'] or [r[0] for r in rows] != list(range(1, len(rows) + 1)):
                connection.close()
                raise ValueError('Index row ordering is inconsistent.')
            vectorizer = _vectorizer(meta['config'], vocabulary=vocabulary)
            vectorizer.idf_ = idf
            self._connection = connection
            self._matrix = matrix
            self._vectorizer = vectorizer
            self._meta = meta
            self._row_doc_ids = np.array([r[2] for r in rows], dtype=object)
            self._chunk_ids = [r[1] for r in rows]
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
            self.close()
            raise RuntimeError('Local search index is missing, stale, or damaged; rebuild it from chunks. ' + str(error)) from error

    def _ensure_loaded(self):
        if self._connection is None:
            self._load()

    def build(self, chunks, config=None):
        config = _normalize_config(config)
        chunks = list(chunks)
        if not chunks:
            raise ValueError('Cannot build a search index without chunks.')
        alias_map, catalog, ids = {}, {}, set()
        fingerprint = hashlib.sha256()
        for chunk in chunks:
            required = ['chunk_id', 'doc_id', 'sha256', 'text']
            if any(not isinstance(chunk.get(key), str) or not chunk[key] for key in required):
                raise ValueError('Each chunk needs nonempty string chunk_id/doc_id/sha256/text.')
            if not chunk['text'].strip() or not re.fullmatch(r'[0-9a-fA-F]{64}', chunk['sha256']):
                raise ValueError('Chunks must contain text and a valid original SHA256.')
            if chunk['chunk_id'] in ids:
                raise ValueError('Duplicate chunk_id: ' + chunk['chunk_id'])
            ids.add(chunk['chunk_id'])
            canonical = chunk.get('canonical_doc_id', chunk['doc_id'])
            if canonical != chunk['doc_id']:
                raise ValueError('Only canonical chunk content may be indexed.')
            aliases = chunk.get('alias_doc_ids') or [canonical]
            for alias in [canonical] + list(aliases):
                if not isinstance(alias, str) or not alias:
                    raise ValueError('Alias document IDs must be nonempty strings.')
                if alias in alias_map and alias_map[alias] != canonical:
                    raise ValueError('An alias cannot point to multiple canonical documents.')
                alias_map[alias] = canonical
            if canonical in catalog and catalog[canonical]['sha256'] != chunk['sha256']:
                raise ValueError('A document ID cannot mix original content versions.')
            if canonical not in catalog:
                context = chunk.get('metadata_context') or {}
                catalog[canonical] = {'doc_id': canonical, 'canonical_doc_id': canonical,
                                      'file_name': chunk.get('file_name', canonical),
                                      'agency': chunk.get('agency'), 'document_year': chunk.get('document_year'),
                                      'title': context.get('title'), 'sha256': chunk['sha256'], 'alias_doc_ids': []}
            encoded = _json_bytes(chunk)
            fingerprint.update(len(encoded).to_bytes(8, 'big')); fingerprint.update(encoded)
        for alias, canonical in alias_map.items():
            catalog[canonical]['alias_doc_ids'].append(alias)
        for entry in catalog.values():
            entry['alias_doc_ids'].sort()
        chunks_fingerprint = fingerprint.hexdigest()
        config_fingerprint = hashlib.sha256(_json_bytes(config)).hexdigest()
        module_hash = _hash_file(__file__)
        dependency_versions = _dependency_versions()
        logical_fingerprint = hashlib.sha256(_json_bytes([INDEX_SCHEMA, chunks_fingerprint, config_fingerprint,
                                                       module_hash, dependency_versions])).hexdigest()
        metadata_path = self.index_dir / 'index_meta.json'
        if metadata_path.is_file():
            try:
                previous = json.loads(metadata_path.read_text(encoding='utf-8'))
                if previous.get('index_fingerprint') == logical_fingerprint:
                    self._load()
                    return self._build_result(reused=True)
            except (OSError, ValueError, TypeError, RuntimeError):
                pass
        self.close()
        self.index_dir.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix='rfp-index-build-', dir=self.index_dir.parent))
        try:
            connection = sqlite3.connect(stage / 'chunks.sqlite3')
            try:
                connection.execute('PRAGMA journal_mode=DELETE')
                connection.execute('CREATE TABLE chunks(row_id INTEGER PRIMARY KEY,chunk_id TEXT UNIQUE NOT NULL,doc_id TEXT NOT NULL,sequence INTEGER NOT NULL,data TEXT NOT NULL)')
                connection.execute('CREATE INDEX chunk_document_idx ON chunks(doc_id,sequence)')
                connection.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(body,metadata,tokenize='unicode61 remove_diacritics 0')")
                rows, fts_rows, char_texts = [], [], []
                for sequence, chunk in enumerate(chunks):
                    row_id = sequence + 1
                    rows.append((row_id, chunk['chunk_id'], chunk['doc_id'], sequence, _json_bytes(chunk).decode('utf-8')))
                    search_text = chunk.get('search_text') or chunk['text']
                    metadata = search_text.replace(chunk['text'], ' ', 1) if chunk['text'] in search_text else search_text
                    fts_rows.append((row_id, ' '.join(tokenize_search(chunk['text'], config)),
                                     ' '.join(tokenize_search(metadata, config))))
                    char_texts.append(_char_view(search_text, config))
                connection.executemany('INSERT INTO chunks VALUES (?,?,?,?,?)', rows)
                connection.executemany('INSERT INTO chunks_fts(rowid,body,metadata) VALUES (?,?,?)', fts_rows)
                connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
                connection.commit()
            finally:
                connection.close()
            effective_min_df = min(config['min_df'], len(chunks))
            vectorizer = _vectorizer(config, min_df=effective_min_df)
            try:
                matrix = vectorizer.fit_transform(char_texts).tocsr()
            except ValueError as error:
                if effective_min_df == 1:
                    raise ValueError('No character vocabulary could be built from chunks.') from error
                effective_min_df = 1
                vectorizer = _vectorizer(config, min_df=1)
                matrix = vectorizer.fit_transform(char_texts).tocsr()
            sparse.save_npz(stage / 'char_vectors.npz', matrix, compressed=True)
            np.save(stage / 'char_idf.npy', vectorizer.idf_.astype(np.float32), allow_pickle=False)
            serial_vocabulary = {term: int(index) for term, index in vectorizer.vocabulary_.items()}
            (stage / 'char_vocabulary.json').write_text(json.dumps(serial_vocabulary, ensure_ascii=False, sort_keys=True), encoding='utf-8')
            meta = {'schema_version': INDEX_SCHEMA, 'tokenizer_version': TOKENIZER_VERSION,
                    'retrieval_module_sha256': module_hash, 'dependencies': dependency_versions,
                    'chunks_fingerprint': chunks_fingerprint, 'config_fingerprint': config_fingerprint,
                    'index_fingerprint': logical_fingerprint, 'config': config,
                    'chunk_count': len(chunks), 'canonical_document_count': len(catalog),
                    'feature_count': matrix.shape[1], 'effective_min_df': effective_min_df,
                    'alias_map': alias_map, 'catalog': list(catalog.values()),
                    'files_sha256': {name: _hash_file(stage / name) for name in ARTIFACT_NAMES},
                    'score_semantics': {'bm25': 'negative SQLite BM25 rank, larger is better',
                                        'char_tfidf': 'character lexical TF-IDF cosine similarity',
                                        'rrf': 'weighted reciprocal rank fusion',
                                        'mmr': 'relevance/diversity objective using lexical TF-IDF redundancy'},
                    'scores_are_confidence': False, 'semantic_embedding_model': None}
            (stage / 'index_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            self.index_dir.mkdir(parents=True, exist_ok=True)
            for name in ARTIFACT_NAMES + ('index_meta.json',):
                os.replace(stage / name, self.index_dir / name)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        self._load()
        return self._build_result(reused=False)

    def _build_result(self, reused):
        return {key: self._meta[key] for key in ['schema_version', 'chunk_count', 'canonical_document_count', 'feature_count',
                                                'index_fingerprint', 'chunks_fingerprint', 'config_fingerprint', 'config']} | {'cache_reused': reused}

    def get_metadata(self):
        self._ensure_loaded()
        return json.loads(json.dumps(self._meta))

    def get_catalog(self):
        self._ensure_loaded()
        return json.loads(json.dumps(self._meta['catalog']))

    def resolve_doc_ids(self, alias_ids):
        self._ensure_loaded()
        if isinstance(alias_ids, str):
            alias_ids = [alias_ids]
        if alias_ids is None:
            raise ValueError('Specify document IDs explicitly; None is not a selected scope.')
        resolved = []
        for alias in alias_ids:
            if alias not in self._meta['alias_map']:
                raise ValueError('Unknown document ID: ' + str(alias))
            canonical = self._meta['alias_map'][alias]
            if canonical not in resolved:
                resolved.append(canonical)
        return resolved

    def get_document_chunks(self, doc_id):
        canonical = self.resolve_doc_ids([doc_id])[0]
        return [json.loads(row[0]) for row in self._connection.execute('SELECT data FROM chunks WHERE doc_id=? ORDER BY sequence', (canonical,))]

    def _scope(self, allowed_doc_ids):
        if allowed_doc_ids is None:
            return None
        return self.resolve_doc_ids(allowed_doc_ids)

    def _bm25_candidates(self, query, count, scope):
        terms = sorted(set(tokenize_search(query, self._meta['config'])))
        if not terms:
            return []
        # Generated terms are alphanumeric only; quoted MATCH atoms cannot become operators.
        expression = ' OR '.join('"' + term + '"' for term in terms[:256])
        where, parameters = '', [self._meta['config']['fts_metadata_weight'], expression]
        if scope is not None:
            where = ' AND c.doc_id IN (' + ','.join('?' for _ in scope) + ')'
            parameters.extend(scope)
        parameters.append(count)
        query_sql = ('SELECT c.row_id,bm25(chunks_fts,1.0,?) AS relevance FROM chunks_fts '
                     'JOIN chunks c ON c.row_id=chunks_fts.rowid WHERE chunks_fts MATCH ?' + where +
                     ' ORDER BY relevance ASC,c.chunk_id ASC LIMIT ?')
        return [(int(row_id), -float(score)) for row_id, score in self._connection.execute(query_sql, parameters)]

    def _char_candidates(self, query, count, scope):
        vector = self._vectorizer.transform([_char_view(query, self._meta['config'])])
        if not vector.nnz:
            return []
        if scope is None:
            matrix, row_indices = self._matrix, None
        else:
            row_indices = np.flatnonzero(np.isin(self._row_doc_ids, np.array(scope, dtype=object)))
            if not len(row_indices):
                return []
            matrix = self._matrix[row_indices]
        similarities = (matrix @ vector.T).tocoo()
        candidates = []
        for row, score in zip(similarities.row, similarities.data):
            if score > 0:
                actual = int(row if row_indices is None else row_indices[row])
                candidates.append((actual + 1, float(score)))
        candidates.sort(key=lambda pair: (-pair[1], self._chunk_ids[pair[0] - 1]))
        return candidates[:count]

    def _fusion(self, bm25, characters):
        config = self._meta['config']
        values, details = {}, {}
        for name, ranking, weight in [('bm25', bm25, config['bm25_weight']), ('char_tfidf', characters, config['char_tfidf_weight'])]:
            if not weight:
                continue
            for rank, (row_id, score) in enumerate(ranking, 1):
                values[row_id] = values.get(row_id, 0.0) + weight / (config['rrf_k'] + rank)
                details.setdefault(row_id, {'component_ranks': {}, 'component_scores': {}})
                details[row_id]['component_ranks'][name] = rank
                details[row_id]['component_scores'][name] = score
        ranking = sorted(values.items(), key=lambda pair: (-pair[1], self._chunk_ids[pair[0] - 1]))
        return ranking, details

    def search(self, query, top_k=5, allowed_doc_ids=None, method='rrf', fetch_k=30, mmr_lambda=0.75):
        self._ensure_loaded()
        if not isinstance(query, str):
            raise TypeError('Search query must be a string.')
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
            raise ValueError('top_k must be a nonnegative integer.')
        if not isinstance(fetch_k, int) or isinstance(fetch_k, bool) or fetch_k < 1:
            raise ValueError('fetch_k must be a positive integer.')
        if method not in {'bm25', 'char_tfidf', 'rrf', 'mmr'}:
            raise ValueError('Unknown search method: ' + str(method))
        if not isinstance(mmr_lambda, (int, float)) or not math.isfinite(mmr_lambda) or not 0 <= mmr_lambda <= 1:
            raise ValueError('mmr_lambda must be between 0 and 1.')
        scope = self._scope(allowed_doc_ids)
        if scope == [] or top_k == 0 or not query.strip() or not tokenize_search(query, self._meta['config']):
            return []
        if len(query) > self._meta['config']['query_max_chars']:
            raise ValueError('Search query is longer than query_max_chars.')
        fetch_k = max(fetch_k, top_k)
        details = {}
        if method == 'bm25':
            ranked = self._bm25_candidates(query, top_k, scope)
            score_kind = 'bm25_rank_score_not_confidence'
        elif method == 'char_tfidf':
            ranked = self._char_candidates(query, top_k, scope)
            score_kind = 'lexical_character_tfidf_cosine_not_confidence'
        else:
            bm25 = self._bm25_candidates(query, fetch_k, scope)
            characters = self._char_candidates(query, fetch_k, scope)
            ranked, details = self._fusion(bm25, characters)
            score_kind = 'weighted_reciprocal_rank_fusion_not_confidence'
            if method == 'mmr' and ranked:
                candidates = ranked[:fetch_k]
                row_ids = [row_id for row_id, _ in candidates]
                vectors = self._matrix[np.array(row_ids) - 1]
                relevance = np.array([score for _, score in candidates], dtype=np.float64)
                relevance /= relevance.max()
                redundancy = np.zeros(len(candidates), dtype=np.float64)
                remaining, selected = set(range(len(candidates))), []
                while remaining and len(selected) < top_k:
                    def objective(index):
                        return mmr_lambda * relevance[index] - (1 - mmr_lambda) * redundancy[index]
                    choice = min(remaining, key=lambda i: (-objective(i), self._chunk_ids[row_ids[i] - 1]))
                    value = objective(choice)
                    selected.append((row_ids[choice], float(value)))
                    remaining.remove(choice)
                    similarity = (vectors @ vectors[choice].T).toarray().ravel()
                    redundancy = np.maximum(redundancy, similarity)
                for row_id, raw_score in candidates:
                    details[row_id]['rrf_score'] = raw_score
                ranked = selected
                score_kind = 'mmr_relevance_diversity_objective_not_confidence'
        results = []
        for rank, (row_id, score) in enumerate(ranked[:top_k], 1):
            row = self._connection.execute('SELECT data FROM chunks WHERE row_id=?', (row_id,)).fetchone()
            hit = json.loads(row[0])
            hit.update({'score': float(score), 'rank': rank, 'method': method,
                        'score_kind': score_kind, 'score_is_confidence': False})
            hit.update(details.get(row_id, {}))
            results.append(hit)
        return results
