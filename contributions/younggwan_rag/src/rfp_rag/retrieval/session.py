"""문서 선택과 사용자 질문 이력을 보존하는 로컬 검색 세션.

검색 결과를 정답으로 생성하거나 과거 assistant 답변을 근거로 재사용하지 않는다.
문서 선택은 명시적 ID와 보수적인 메타데이터 일치만 사용한다. 연도 불일치를
문서 부재로 판단하지 않으며, 불명확한 후보는 선택 질문으로 남긴다.
"""
from __future__ import annotations

import re
import unicodedata


def _compact(value):
    text = unicodedata.normalize('NFC', str(value or '')).casefold()
    return ''.join(character for character in text if character.isalnum())


def _project_title(document):
    if isinstance(document.get('title'), str) and document['title'].strip():
        return document['title'].strip()
    name = re.sub(r'\.[^.]+$', '', str(document.get('file_name') or ''))
    parts = name.split('_')
    if len(parts) > 1 and _compact(parts[0]) == _compact(document.get('agency')):
        parts.pop(0)
    if parts and re.fullmatch(r'(?:19|20)\d{2}', parts[0]):
        parts.pop(0)
    kept = []
    for part in parts:
        if re.fullmatch(r'(?:제안요청서(?: 및 과업지시서)?|과업내용서|과업지시서|안|수정(?:본)?|최종|재공고|사전규격|사본\d+|v?\d+(?:\.\d+)*)', part, flags=re.I):
            continue
        kept.append(part)
    return ' '.join(kept) or name


def _title_variants(title):
    variants = [_compact(title)]
    without_year = re.sub(r'^(?:19|20)\d{2}\s*년(?:도)?\s*', '', title)
    variants.append(_compact(without_year))
    without_suffix = re.sub(r'\s+(?:사업|용역)$', '', without_year)
    variants.append(_compact(without_suffix))
    # 짧은 보통명사만으로 사업을 자동 확정하지 않는다.
    return list(dict.fromkeys(value for value in variants if len(value) >= 8))


def _is_comparison(query):
    return bool(re.search(r'비교|차이|둘\s*다|두\s*사업', query))


def _ambiguous_title_matches(matches):
    # 같은/포함 관계의 이름만으로 여러 문서가 맞으면 서로 다른 사업 언급이라고
    # 확정하지 않는다. 비교 요청도 후보 문서를 모두 자동 선택하는 근거는 아니다.
    return any(doc_id != other_id and (variant in other or other in variant)
               for index, (doc_id, variant) in enumerate(matches)
               for other_id, other in matches[index + 1:])


class SearchSession:
    """인덱스를 공유해도 선택 문서와 대화 상태는 세션마다 독립적이다.

    ``ask``는 검색만 실행하며 status는 retrieved/needs_clarification/
    insufficient_evidence 중 하나다. retrieved는 생성 답변의 정확성을 뜻하지 않는다.
    ``candidate_doc_ids``는 선택이 없는 세션에 사용자 인터페이스/문서 선택기가
    제공한 후보만 받는다. 현재 선택이 있으면 그 선택을 우선한다. 평가 정답의
    target_doc_ids를 이 인수에 주입하면 안 된다.
    """

    def __init__(self, index, active_doc_ids=None):
        self.index = index
        self._active = ()
        self._history = []
        self._rewrite_start = 0
        if active_doc_ids is not None:
            self.select_documents(active_doc_ids)

    @property
    def active_doc_ids(self):
        return list(self._active)

    @property
    def query_history(self):
        return [turn['user'] for turn in self._history]

    def _catalog(self):
        getter = getattr(self.index, 'get_catalog', None)
        if getter is None:
            return []
        rows = getter()
        if isinstance(rows, dict):
            rows = list(rows.values())
        return [row for row in rows if isinstance(row, dict) and row.get('doc_id')]

    def _canonical_ids(self, ids):
        requested = [ids] if isinstance(ids, str) else list(ids)
        if any(not isinstance(value, str) or not value.strip() for value in requested):
            raise ValueError('문서 ID는 비어 있지 않은 문자열이어야 합니다.')
        canonical = list(self.index.resolve_doc_ids(requested))
        if requested and not canonical:
            raise ValueError('선택한 문서 ID를 인덱스에서 찾지 못했습니다.')
        return list(dict.fromkeys(canonical))

    def select_documents(self, ids):
        """선택을 바꾸면 과거 사업의 질문을 새 검색어에 덧붙이지 않는다.

        빈 목록 또는 None은 명시적인 선택 해제다. 잘못된 ID는 이전 선택을 유지한
        채 ValueError로 끝나며, 범위 없는 전체 검색으로 바꾸지 않는다.
        """
        canonical = [] if ids is None else self._canonical_ids(ids)
        chosen = tuple(canonical)
        if chosen != self._active:
            self._active = chosen
            self._rewrite_start = len(self._history)
        return self.active_doc_ids

    def _named_candidates(self, query):
        normalized = _compact(query)
        catalog = self._catalog()
        exact, shorter, agencies = [], [], []
        exact_matches, shorter_matches = [], []
        for row in catalog:
            title = _project_title(row)
            variants = _title_variants(title)
            if variants and variants[0] in normalized:
                exact.append(row['doc_id'])
                exact_matches.append((row['doc_id'], variants[0]))
            elif any(variant in normalized for variant in variants[1:]):
                shorter.append(row['doc_id'])
                shorter_matches.append((row['doc_id'], variants[1:]))
            agency = _compact(row.get('agency'))
            agency_variants = [agency]
            if '특별시' in str(row.get('agency', '')):
                agency_variants.append(_compact(str(row['agency']).replace('특별시', '시')))
            if any(len(value) >= 3 and value in normalized for value in agency_variants):
                agencies.append(row['doc_id'])
        if exact:
            if _is_comparison(query):
                # 완전한 A 사업명과 연도를 생략한 B 사업명이 섞인 비교도 보존한다.
                # A 이름 내부에서 우연히 일치한 다른 연도의 문서는 추가하지 않는다.
                remaining = normalized
                for variant in sorted({value for _, value in exact_matches}, key=len, reverse=True):
                    remaining = remaining.replace(variant, '')
                independent = [(doc_id, max((value for value in variants if value in remaining),
                                           key=len))
                               for doc_id, variants in shorter_matches
                               if any(value in remaining for value in variants)]
                if _ambiguous_title_matches(exact_matches) or _ambiguous_title_matches(independent):
                    return list(dict.fromkeys(exact + [value for value, _ in independent])), 'ambiguous_comparison_titles'
                if independent:
                    return list(dict.fromkeys(exact + [value for value, _ in independent])), 'explicit_comparison_titles'
            return list(dict.fromkeys(exact)), 'explicit_project_title'
        if shorter:
            if _is_comparison(query):
                matches = [(doc_id, max((value for value in variants if value in normalized), key=len))
                           for doc_id, variants in shorter_matches]
                if _ambiguous_title_matches(matches):
                    return list(dict.fromkeys(shorter)), 'ambiguous_comparison_titles'
            return list(dict.fromkeys(shorter)), 'project_title_without_optional_year'
        return list(dict.fromkeys(agencies)), 'agency_only' if agencies else None

    def _candidate_rows(self, ids):
        by_id = {row['doc_id']: row for row in self._catalog()}
        return [{'doc_id': value, 'file_name': by_id.get(value, {}).get('file_name'),
                 'agency': by_id.get(value, {}).get('agency'),
                 'document_year': by_id.get(value, {}).get('document_year'),
                 'project_title': _project_title(by_id[value]) if value in by_id else None}
                for value in ids]

    def _standalone_query(self, query):
        parts = [query]
        catalog = {row['doc_id']: row for row in self._catalog()}
        titles = [_project_title(catalog[doc_id]) for doc_id in self._active if doc_id in catalog]
        if titles:
            parts.append('현재 선택 사업: ' + '; '.join(titles))
        refers_back = bool(re.match(r'^(?:그(?:건|것|중|럼| 사업| 문서| 장비)?|해당|이거|이 사업|거기)', query))
        # 검색 질의에는 같은 선택 범위의 직전 사용자 발화만 보조 문맥으로 제공한다.
        # standalone_query 자체를 다시 누적하지 않아 검색어가 재귀적으로 커지지 않는다.
        previous = self._history[self._rewrite_start:]
        if refers_back and previous:
            last = previous[-1]
            if tuple(last['active_doc_ids']) == self._active:
                parts.append('직전 사용자 질문: ' + last['user'][:1200])
        return '\n'.join(parts)

    def _result(self, status, query, standalone_query=None, hits=None, candidates=None,
                reason=None, selection_reason=None):
        return {'status': status, 'query': query, 'standalone_query': standalone_query or query,
                'active_doc_ids': self.active_doc_ids, 'hits': list(hits or []),
                'candidate_documents': list(candidates or []), 'reason': reason,
                'selection_reason': selection_reason, 'query_history': self.query_history,
                'retrieval_only': True, 'answer_generated': False,
                'language_model_used': False}

    def ask(self, query, top_k=5, method='rrf', *, candidate_doc_ids=None):
        """사용자 질문과 실제 현재 선택만으로 검색한다. 정답 데이터 인수는 없다."""
        if not isinstance(query, str):
            raise TypeError('질문은 문자열이어야 합니다.')
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError('top_k는 1 이상의 정수여야 합니다.')
        query = query.strip()
        if not query:
            return self._result('needs_clarification', query, reason='질문을 입력해 주세요.')

        selection_reason = None
        named, naming_basis = self._named_candidates(query)
        explicit_names = naming_basis in {'explicit_project_title', 'project_title_without_optional_year',
                                          'explicit_comparison_titles', 'ambiguous_comparison_titles'}
        supplied_candidates = None
        if not self._active and candidate_doc_ids is not None:
            supplied_candidates = self._canonical_ids(candidate_doc_ids)
            if not supplied_candidates:
                self._history.append({'user': query, 'active_doc_ids': self.active_doc_ids})
                return self._result('insufficient_evidence', query,
                                    reason='제공된 문서 후보 범위에서 검색할 문서를 확보하지 못했습니다. 문서의 부재를 확정한 것은 아닙니다.')
            if explicit_names and named:
                in_scope = [value for value in named if value in supplied_candidates]
                if not in_scope:
                    self._history.append({'user': query, 'active_doc_ids': self.active_doc_ids})
                    return self._result('needs_clarification', query,
                                        candidates=self._candidate_rows(supplied_candidates),
                                        reason='질문에서 명시한 사업과 제공된 문서 후보가 다릅니다. 후보 밖 문서로 자동 전환하지 않고 범위를 확인합니다.')
                named = in_scope
        if naming_basis == 'ambiguous_comparison_titles':
            self._history.append({'user': query, 'active_doc_ids': self.active_doc_ids})
            return self._result('needs_clarification', query,
                                candidates=self._candidate_rows(self._canonical_ids(named)),
                                reason='비교할 사업 이름과 일치하는 문서 후보가 겹칩니다. 연도나 문서 ID로 비교 대상을 선택해 주세요.')
        if naming_basis == 'project_title_without_optional_year' and named:
            requested_years = set(re.findall(r'(?<!\d)((?:19|20)\d{2})\s*년', query))
            catalog = {row['doc_id']: row for row in self._catalog()}
            known_years = set()
            for value in named:
                row = catalog.get(value, {})
                known_years.update(re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)', _project_title(row)))
                if row.get('document_year') is not None:
                    known_years.add(str(row['document_year']))
            if requested_years and known_years and not requested_years & known_years:
                self._history.append({'user': query, 'active_doc_ids': self.active_doc_ids})
                return self._result('needs_clarification', query,
                                    candidates=self._candidate_rows(self._canonical_ids(named)),
                                    reason='관련 사업은 찾았지만 질문 연도와 사업·문서 연도의 관계가 불명확합니다. 문서 또는 연도 기준을 확인해 주세요. 문서 부재를 확정하지 않습니다.')
        if explicit_names and len(named) == 1:
            canonical = self._canonical_ids(named)
            if tuple(canonical) != self._active:
                self.select_documents(canonical)
                selection_reason = naming_basis
        elif explicit_names and len(named) > 1:
            if _is_comparison(query):
                self.select_documents(named)
                selection_reason = 'explicit_comparison_projects'
            else:
                self._history.append({'user': query, 'active_doc_ids': self.active_doc_ids})
                return self._result('needs_clarification', query,
                                    candidates=self._candidate_rows(self._canonical_ids(named)),
                                    reason='사업명과 일치하는 문서가 여러 개입니다. 현재 선택을 임의 변경하지 않고 문서 범위를 확인합니다.')
        elif not self._active:
            candidates = supplied_candidates if supplied_candidates is not None else self._canonical_ids(named)
            if len(candidates) > 1:
                self._history.append({'user': query, 'active_doc_ids': self.active_doc_ids})
                return self._result('needs_clarification', query,
                                    candidates=self._candidate_rows(candidates),
                                    reason='관련 문서 후보가 여러 개입니다. 사업명이나 문서를 선택해 주세요.')
            if candidate_doc_ids is not None:
                if not candidates:
                    self._history.append({'user': query, 'active_doc_ids': self.active_doc_ids})
                    return self._result('insufficient_evidence', query,
                                        reason='제공된 문서 후보 범위에서 검색할 문서를 확보하지 못했습니다. 문서의 부재를 확정한 것은 아닙니다.')
                self.select_documents(candidates)
                selection_reason = 'caller_supplied_single_candidate'
            elif explicit_names and candidates:
                self.select_documents(candidates)
                selection_reason = naming_basis

        standalone = self._standalone_query(query)
        scope = self.active_doc_ids if self._active else None
        hits = self.index.search(standalone, top_k=top_k, allowed_doc_ids=scope, method=method)
        self._history.append({'user': query, 'active_doc_ids': self.active_doc_ids})
        if hits:
            return self._result('retrieved', query, standalone_query=standalone, hits=hits,
                                selection_reason=selection_reason)
        return self._result('insufficient_evidence', query, standalone_query=standalone,
                            reason='현재 검색 범위와 질의에서 근거 후보를 찾지 못했습니다. 문서에 해당 조건이 없다는 확정은 아닙니다.',
                            selection_reason=selection_reason)

    def get_full_document_chunks(self, doc_id):
        """전체 요약 준비용 경로. top_k 없이 선택 문서의 모든 청크를 가져온다."""
        canonical = self._canonical_ids([doc_id])
        if len(canonical) != 1:
            raise ValueError('전체 문서 경로에는 문서 한 개를 지정해야 합니다.')
        chunks = list(self.index.get_document_chunks(canonical[0]))
        self.select_documents(canonical)
        return {'status': 'document_chunks' if chunks else 'insufficient_evidence',
                'doc_id': canonical[0], 'active_doc_ids': self.active_doc_ids,
                'chunks': chunks, 'returned_chunks': len(chunks),
                'total_indexed_chunks': len(chunks),
                'structural_chunk_coverage': 1.0 if chunks else None,
                'coverage_basis': 'all_chunks_returned_by_index_for_selected_document',
                'full_original_semantic_coverage_verified': False,
                'answer_generated': False, 'language_model_used': False}
