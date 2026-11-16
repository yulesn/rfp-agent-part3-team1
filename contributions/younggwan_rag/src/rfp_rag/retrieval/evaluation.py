"""Source-grounded chunk coverage and local retrieval experiments.

Gold text is used only to label source intervals after retrieval. Search receives
the user's original question and, in the context protocol, only user-selected
document IDs. This module does not score generated answers.
"""
from __future__ import annotations

from array import array
from bisect import bisect_left
from collections import defaultdict
import math
import statistics
import time


def _compact(text):
    return ''.join(character for character in text if not character.isspace())


def _positions(text):
    return array('I', (i for i, character in enumerate(text) if not character.isspace()))


def _union_size(intervals):
    intervals = sorted((int(a), int(b)) for a, b in intervals if b > a)
    if not intervals:
        return 0
    total = 0
    start, end = intervals[0]
    for a, b in intervals[1:]:
        if a <= end:
            end = max(end, b)
        else:
            total += end - start
            start, end = a, b
    return total + end - start


def _link_id(case_id, fact_id, evidence_kind, evidence_index):
    return f'{case_id}|{evidence_kind}|{fact_id or "-"}|{evidence_index}'


class _SourceMap:
    def __init__(self, documents):
        self.documents = {}
        self.blocks = {}
        self.references = {}
        self.position_cache = {}
        self.section_cache = {}
        for document in documents:
            doc_id = document['doc_id']
            if doc_id in self.documents:
                raise ValueError(f'Duplicate source doc_id: {doc_id}')
            self.documents[doc_id] = document
            blocks = {}
            for collection in ['blocks', 'supplemental_blocks']:
                for block in document.get(collection, []):
                    key = collection, block['block_id']
                    if key in blocks:
                        raise ValueError(f'Duplicate source block: {doc_id} {key}')
                    blocks[key] = block
            self.blocks[doc_id] = blocks
            self.references[doc_id] = {b['block_id']: b for b in document.get('reference_blocks', [])}

    def canonical(self, doc_id):
        document = self.documents.get(doc_id)
        if document is None:
            raise ValueError('Source document is missing')
        canonical_id = document.get('canonical_doc_id', doc_id)
        canonical = self.documents.get(canonical_id)
        if canonical is None or canonical['sha256'] != document['sha256']:
            raise ValueError('Canonical alias is absent or has different source bytes')
        return canonical_id, canonical

    def block_positions(self, doc_id, collection, block_id):
        key = doc_id, collection, block_id
        if key not in self.position_cache:
            self.position_cache[key] = _positions(self.blocks[doc_id][(collection, block_id)]['text'])
        return self.position_cache[key]

    def section(self, doc_id, reference_id):
        key = doc_id, reference_id
        if key in self.section_cache:
            return self.section_cache[key]
        reference = self.references[doc_id][reference_id]
        section_id = reference.get('section', reference['locator'])
        parts = [b for b in self.documents[doc_id].get('blocks', []) if b.get('section') == section_id]
        normal_parts, spans, offset = [], [], 0
        for block in parts:
            compact = _compact(block['text'])
            normal_parts.append(compact)
            spans.append((offset, offset + len(compact), block))
            offset += len(compact)
        # No approximate diff/alignment or reference injection is permitted.
        if ''.join(normal_parts) != _compact(reference['text']):
            raise ValueError('Reference view and primary section text do not have identical non-whitespace content')
        result = reference, _positions(reference['text']), spans
        self.section_cache[key] = result
        return result

    def candidate_fragments(self, doc_id, candidate, quote):
        block_id = candidate.get('block_id')
        start, end = candidate.get('char_start'), candidate.get('char_end')
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end:
            raise ValueError('Source span offsets are missing or invalid')
        if candidate.get('reference_view'):
            reference, reference_positions, section_spans = self.section(doc_id, block_id)
            if end > len(reference['text']) or _compact(reference['text'][start:end]) != _compact(quote):
                raise ValueError('Reference source span no longer matches the gold quotation')
            lower, upper = bisect_left(reference_positions, start), bisect_left(reference_positions, end)
            fragments = []
            for a, b, block in section_spans:
                left, right = max(a, lower), min(b, upper)
                if left >= right:
                    continue
                positions = self.block_positions(doc_id, 'blocks', block['block_id'])
                i, j = left - a, right - a
                fragments.append({'collection': 'blocks', 'block_id': block['block_id'],
                                  'locator': block['locator'], 'char_start': positions[i],
                                  'char_end': positions[j - 1] + 1, 'normalized_start': i,
                                  'normalized_end': j, 'target_start': left - lower,
                                  'target_end': right - lower})
            return fragments, upper - lower
        collections = [collection for collection in ['blocks', 'supplemental_blocks']
                       if (collection, block_id) in self.blocks[doc_id]]
        if len(collections) != 1:
            raise ValueError('Source block is missing or ambiguous across collections')
        collection = collections[0]
        block = self.blocks[doc_id][(collection, block_id)]
        if end > len(block['text']) or _compact(block['text'][start:end]) != _compact(quote):
            raise ValueError('Primary source span no longer matches the gold quotation')
        positions = self.block_positions(doc_id, collection, block_id)
        left, right = bisect_left(positions, start), bisect_left(positions, end)
        return [{'collection': collection, 'block_id': block_id, 'locator': block['locator'],
                 'char_start': start, 'char_end': end, 'normalized_start': left,
                 'normalized_end': right, 'target_start': 0, 'target_end': right - left}], right - left


def audit_chunk_coverage(documents, chunks, evidence_validation):
    """Map all gold evidence links to exact primary-source chunk intervals.

    Every alternative of an ambiguous original quotation is retained. A link can
    be supported by any valid alternative, but no arbitrary location is selected.
    Whitespace has zero weight; punctuation, digits and Korean text retain weight.
    Returned records are evaluation labels and must never be sent to a retriever.
    """
    source = _SourceMap(documents)
    span_index = defaultdict(list)
    issues = []
    chunk_ids = set()
    for chunk in chunks:
        cid = chunk.get('chunk_id')
        if not cid or cid in chunk_ids:
            raise ValueError('Chunk IDs must be present and unique')
        chunk_ids.add(cid)
        doc_id = chunk.get('doc_id')
        try:
            canonical_id, document = source.canonical(doc_id)
        except ValueError as error:
            issues.append({'chunk_id': cid, 'reason': str(error)})
            continue
        if canonical_id != doc_id or chunk.get('sha256') != document['sha256']:
            issues.append({'chunk_id': cid, 'reason': 'Chunk document identity or source hash is invalid'})
            continue
        chunk_positions = _positions(chunk.get('text', ''))
        covered_chunk_intervals = []
        pending_spans = []
        previous_issue_count = len(issues)
        for span in chunk.get('source_spans', []):
            collection = span.get('collection', 'blocks')
            block = source.blocks[doc_id].get((collection, span.get('block_id')))
            if block is None:
                issues.append({'chunk_id': cid, 'reason': 'Missing primary block or forbidden reference-view indexing'})
                continue
            if collection == 'supplemental_blocks' and (block.get('review_status') != 'visual_verified'
                    or block.get('source_method') != 'reviewed_visual_transcription'):
                issues.append({'chunk_id': cid, 'reason': 'Unreviewed supplemental/OCR text cannot become indexed evidence'})
                continue
            a, b, c, d = (span.get(k) for k in ['char_start', 'char_end', 'chunk_start', 'chunk_end'])
            valid = all(isinstance(v, int) for v in [a, b, c, d])
            valid = valid and 0 <= a <= b <= len(block['text']) and 0 <= c <= d <= len(chunk['text'])
            valid = valid and block['text'][a:b] == chunk['text'][c:d] and span.get('locator') == block['locator']
            if not valid:
                issues.append({'chunk_id': cid, 'reason': 'Chunk text and exact original source offsets disagree', 'block_id': span.get('block_id')})
                continue
            pending_spans.append(((doc_id, collection, span['block_id']), (a, b, cid)))
            covered_chunk_intervals.append((bisect_left(chunk_positions, c), bisect_left(chunk_positions, d)))
        if not chunk_positions or _union_size(covered_chunk_intervals) != len(chunk_positions):
            issues.append({'chunk_id': cid, 'reason': 'Chunk contains non-whitespace text without an exact primary-source span'})
        # A partly valid chunk still contains unsupported material. Its valid
        # fragments cannot make an otherwise invalid indexed chunk look useful.
        if len(issues) == previous_issue_count:
            for key, interval in pending_spans:
                span_index[key].append(interval)

    labels = []
    for original in evidence_validation.get('records', []):
        label = {key: original.get(key) for key in ['case_id', 'fact_id', 'evidence_kind', 'evidence_index', 'doc_id', 'quote']}
        label['link_id'] = _link_id(label['case_id'], label['fact_id'], label['evidence_kind'], label['evidence_index'])
        label.update(mapped=False, fully_covered=False, all_alternatives_covered=False,
                     canonical_doc_id=None, alternatives=[], source_ambiguous=not original.get('location_resolved', False))
        labels.append(label)
        try:
            canonical_id, document = source.canonical(original['doc_id'])
            label['canonical_doc_id'] = canonical_id
            if document['sha256'] != original.get('gold_sha256') or document['sha256'] != original.get('source_sha256'):
                raise ValueError('Gold/source/chunk source hash disagreement')
            if original.get('match_mode') not in {'exact', 'normalized_whitespace'}:
                raise ValueError('Original evidence did not have a valid source match')
            candidates = original.get('candidates') or [original]
            for number, candidate in enumerate(candidates):
                fragments, required = source.candidate_fragments(canonical_id, candidate, original['quote'])
                if required < 1:
                    raise ValueError('A quotation must contain non-whitespace source text')
                coverage = defaultdict(list)
                for fragment in fragments:
                    positions = source.block_positions(canonical_id, fragment['collection'], fragment['block_id'])
                    for a, b, cid in span_index[(canonical_id, fragment['collection'], fragment['block_id'])]:
                        left, right = max(a, fragment['char_start']), min(b, fragment['char_end'])
                        if left >= right:
                            continue
                        i, j = bisect_left(positions, left), bisect_left(positions, right)
                        if i < j:
                            coverage[cid].append([fragment['target_start'] + i - fragment['normalized_start'],
                                                  fragment['target_start'] + j - fragment['normalized_start']])
                total = _union_size(interval for intervals in coverage.values() for interval in intervals)
                label['alternatives'].append({'alternative': number, 'original_locator': candidate.get('locator'),
                    'required_non_whitespace_characters': required, 'source_fragments': fragments,
                    'chunk_coverage': dict(coverage), 'covered_non_whitespace_characters': total,
                    'fully_covered': total == required})
            label['mapped'] = bool(label['alternatives'])
            label['fully_covered'] = any(a['fully_covered'] for a in label['alternatives'])
            label['all_alternatives_covered'] = all(a['fully_covered'] for a in label['alternatives'])
        except (KeyError, TypeError, ValueError, IndexError) as error:
            label['reason'] = str(error)
    return {'total_links': len(labels), 'mapped_links': sum(r['mapped'] for r in labels),
            'unmapped_links': sum(not r['mapped'] for r in labels),
            'fully_covered_links': sum(r['fully_covered'] for r in labels),
            'covered_links': sum(r['fully_covered'] for r in labels),
            'uncovered_links': sum(not r['fully_covered'] for r in labels),
            'partially_or_not_covered_links': sum(not r['fully_covered'] for r in labels),
            'all_alternatives_covered_links': sum(r['all_alternatives_covered'] for r in labels),
            'source_ambiguous_links': sum(r['source_ambiguous'] for r in labels),
            'chunk_source_issue_count': len(issues), 'chunk_source_issues': issues,
            'valid': not issues and bool(labels) and all(r['fully_covered'] for r in labels),
            'document_aliases': {doc_id: doc.get('canonical_doc_id', doc_id) for doc_id, doc in source.documents.items()},
            'records': labels,
            'scope': 'Primary-source character-span preservation after chunking; not retrieval or answer accuracy.',
            'coverage_unit': 'non-whitespace Unicode source characters; exact raw character offsets retained'}


def _coverage(label, returned_chunk_ids):
    candidates = []
    for alternative in (label or {}).get('alternatives', []):
        required = alternative['required_non_whitespace_characters']
        intervals = [interval for cid, spans in alternative['chunk_coverage'].items()
                     if cid in returned_chunk_ids for interval in spans]
        covered = _union_size(intervals)
        candidates.append((covered / required if required else 0.0, covered == required and required > 0))
    return max((c[0] for c in candidates), default=0.0), any(c[1] for c in candidates)


def _case_links(case, labels):
    result = []
    for fact in case.get('required_facts', []):
        for index, _ in enumerate(fact.get('evidence', []), 1):
            link_id = _link_id(case['case_id'], fact['fact_id'], 'fact', index)
            result.append((link_id, labels.get(link_id)))
    return result


def _percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def evaluate_retrieval(index, cases, alignment, top_ks=(5, 10, 20), method='rrf', scope_mode='context', *, allow_holdout=False, search_config=None):
    """Evaluate fact/comparison evidence recovery with answer-free user queries.

    Context protocol filters only context.selected_doc_ids. Global protocol has
    no document filter. Gold targets, facts and values are never query inputs.
    Recall requires full quotation-character coverage across the returned set;
    MRR is separately labelled as first source overlap, not full-proof success.
    """
    if scope_mode not in {'context', 'global'}:
        raise ValueError('scope_mode must be context or global')
    top_ks = tuple(sorted(set(top_ks)))
    if not top_ks or any(not isinstance(k, int) or k < 1 for k in top_ks):
        raise ValueError('top_ks must contain positive integers')
    search_config = dict(search_config or {})
    fetch_k = search_config.get('fetch_k', 30)
    mmr_lambda = search_config.get('mmr_lambda', .75)
    if not isinstance(fetch_k, int) or isinstance(fetch_k, bool) or fetch_k < 1:
        raise ValueError('search_config.fetch_k must be a positive integer')
    if not isinstance(mmr_lambda, (int, float)) or isinstance(mmr_lambda, bool) or not 0 <= mmr_lambda <= 1:
        raise ValueError('search_config.mmr_lambda must be between zero and one')
    effective_search_config = {'fetch_k': max(fetch_k, max(top_ks)), 'mmr_lambda': mmr_lambda,
                               'method': method, 'top_k': max(top_ks)}
    if any(c.get('split') == 'holdout' for c in cases) and not allow_holdout:
        raise ValueError('Freeze development configuration before explicitly allowing holdout evaluation')
    labels = {r['link_id']: r for r in alignment['records']}
    alias = alignment.get('document_aliases', {})
    selected = [c for c in cases if c.get('feature') in {'fact', 'comparison'} and c.get('required_facts')]
    rows, latencies = [], []
    for case in selected:
        links = _case_links(case, labels)
        context = case.get('context', {}).get('selected_doc_ids')
        allowed = list(context) if scope_mode == 'context' and context else None
        started = time.perf_counter()
        hits = index.search(case['question'], top_k=max(top_ks), allowed_doc_ids=allowed,
                            method=method, fetch_k=effective_search_config['fetch_k'], mmr_lambda=mmr_lambda)
        elapsed = (time.perf_counter() - started) * 1000
        latencies.append(elapsed)
        row = {'case_id': case['case_id'], 'feature': case['feature'], 'query': case['question'],
               'scope_protocol': scope_mode, 'user_selected_doc_ids': allowed,
               'gold_scope_used_by_retriever': False, 'required_evidence_links': len(links),
               'unmapped_evidence_labels': sum(label is None or not label.get('mapped') for _, label in links),
               'latency_ms': elapsed, 'retrieved_chunk_ids': [h['chunk_id'] for h in hits],
               'metrics': {}}
        gold_docs = {alias.get(f['doc_id'], f['doc_id']) for f in case['required_facts']}
        first_overlap = 0.0
        for rank, hit in enumerate(hits, 1):
            if any(_coverage(label, {hit['chunk_id']})[0] > 0 for _, label in links):
                first_overlap = 1.0 / rank
                break
        row['reciprocal_rank_first_source_overlap'] = first_overlap
        for k in top_ks:
            chosen = hits[:k]
            ids = {h['chunk_id'] for h in chosen}
            coverage = [_coverage(label, ids) for _, label in links]
            complete = sum(full for _, full in coverage)
            retrieved_docs = {alias.get(h['doc_id'], h['doc_id']) for h in chosen}
            row['metrics'][str(k)] = {
                'fully_recovered_links': complete, 'evidence_links': len(links),
                'evidence_span_recall': complete / len(links) if links else None,
                'mean_evidence_character_coverage': statistics.mean(ratio for ratio, _ in coverage) if coverage else None,
                'document_recall': len(gold_docs & retrieved_docs) / len(gold_docs) if gold_docs else None,
                'all_required_evidence_recovered': bool(links) and complete == len(links)}
        rows.append(row)
    totals = {}
    for k in top_ks:
        metrics = [r['metrics'][str(k)] for r in rows]
        denominator = sum(m['evidence_links'] for m in metrics)
        numerator = sum(m['fully_recovered_links'] for m in metrics)
        totals[str(k)] = {'fully_recovered_links': numerator, 'evidence_links': denominator,
                          'micro_evidence_span_recall': numerator / denominator if denominator else None,
                          'macro_case_evidence_span_recall': statistics.mean(m['evidence_span_recall'] for m in metrics) if metrics else None,
                          'mean_document_recall': statistics.mean(m['document_recall'] for m in metrics) if metrics else None,
                          'mean_evidence_character_coverage': statistics.mean(m['mean_evidence_character_coverage'] for m in metrics) if metrics else None,
                          'complete_cases': sum(m['all_required_evidence_recovered'] for m in metrics)}
    return {'method': method, 'scope_protocol': scope_mode, 'case_count': len(rows),
            'effective_search_config': effective_search_config,
            'case_features': ['fact', 'comparison'], 'excluded_case_count': len(cases) - len(rows),
            'splits': sorted({c.get('split', 'unspecified') for c in selected}),
            'metrics_at_k': totals, 'mrr_first_source_overlap': statistics.mean(r['reciprocal_rank_first_source_overlap'] for r in rows) if rows else None,
            'latency_ms': {'p50': statistics.median(latencies) if latencies else None, 'p95': _percentile(latencies, .95)},
            'case_results': rows, 'api_calls': 0,
            'scope': 'Retrieval of positive fact/comparison evidence only; not generated-answer or five-feature task accuracy.',
            'query_policy': 'Exact case.question only. Context selected IDs may filter; target_doc_ids and gold values never do.',
            'recall_definition': 'A link succeeds only when all non-whitespace quotation source characters are covered by the union of top-k chunks; ambiguous source alternatives use any legitimate candidate.',
            'mrr_definition': 'First result overlapping a required source interval. This is not full-proof MRR.'}


def evaluate_summary_coverage(index, cases, alignment, *, allow_holdout=False):
    """Check the all-document summary route without claiming summary generation."""
    if any(c.get('split') == 'holdout' for c in cases) and not allow_holdout:
        raise ValueError('Holdout requires an explicitly frozen evaluation run')
    labels = {r['link_id']: r for r in alignment['records']}
    rows = []
    for case in cases:
        if case.get('feature') != 'summary':
            continue
        selected = list(case.get('context', {}).get('selected_doc_ids') or [])
        chunks = [chunk for doc_id in selected for chunk in index.get_document_chunks(doc_id)]
        ids = {c['chunk_id'] for c in chunks}
        links = _case_links(case, labels)
        fully_covered = sum(_coverage(label, ids)[1] for _, label in links)
        rows.append({'case_id': case['case_id'], 'user_selected_doc_ids': selected, 'chunks_in_full_route': len(ids),
                     'evidence_links': len(links), 'fully_covered_links': fully_covered,
                     'all_required_evidence_covered': bool(links) and fully_covered == len(links)})
    return {'cases': len(rows), 'all_evidence_covered_cases': sum(r['all_required_evidence_covered'] for r in rows),
            'case_results': rows, 'scope': 'Full-document input coverage only; no LLM summary was generated or graded.'}
