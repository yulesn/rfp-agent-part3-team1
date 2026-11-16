"""Byte-bounded local RFP chunks with exact source spans and explicit quality.

The byte budget is not a model token count.  Synthetic separators and metadata
are never attributed to the original document.  Table text comes exclusively
from primary paragraph blocks, not from a second rendering of table cells.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import hashlib
import json
import re

VERSION = 'rfp-source-chunker-v1'
DEFAULT_CONFIG = {'chunk_size_bytes': 1800, 'overlap_bytes': 240}


def _configuration(config):
    settings = dict(DEFAULT_CONFIG)
    if config:
        for key in DEFAULT_CONFIG:
            if key in config:
                settings[key] = config[key]
    maximum, overlap = settings['chunk_size_bytes'], settings['overlap_bytes']
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 4:
        raise ValueError('chunk_size_bytes must be an integer >= 4.')
    if not isinstance(overlap, int) or isinstance(overlap, bool) or not 0 <= overlap < maximum:
        raise ValueError('overlap_bytes must be an integer in [0, chunk_size_bytes).')
    return settings


def _prefix_bytes(text):
    values = [0]
    size = 0
    for character in text:
        size += len(character.encode('utf-8'))
        values.append(size)
    return values


def _boundary(block, document):
    if document.get('detected_format') == 'pdf' or block.get('page') is not None:
        return ('page', block.get('page', block['locator']))
    section = block.get('section')
    if section is None:
        match = re.match(r'(BodyText/Section\d+|Contents/section\d+\.xml)', block['locator'])
        section = match.group(1) if match else None
    # An unknown boundary is not permission to merge unrelated positions.
    return ('section', section) if section else ('block', block['block_id'])


def _eligible(block, collection):
    method = str(block.get('source_method', '')).lower()
    kind = str(block.get('kind', '')).lower()
    if 'ocr' in method or 'ocr' in kind:
        return False
    if collection == 'supplemental_blocks' or method == 'reviewed_visual_transcription' or kind == 'verified_transcription':
        return block.get('review_status') == 'visual_verified'
    return block.get('review_status') != 'unreviewed'


def _group_key(block, collection, document):
    boundary = _boundary(block, document)
    if collection == 'supplemental_blocks' or block.get('source_method') == 'reviewed_visual_transcription' or block.get('kind') == 'verified_transcription':
        # A reviewed transcription is never merged into automatic page text.
        return (boundary, 'verified_transcription', block['block_id'], collection)
    if document.get('detected_format') != 'pdf' and block.get('table_id'):
        return (boundary, 'table', block['table_id'], collection)
    return (boundary, 'prose', None, collection)


def _groups(document, audit):
    current, current_key = [], None
    seen = set()
    for collection in ['blocks', 'supplemental_blocks']:
        for block in document.get(collection, []) or []:
            if not isinstance(block, dict) or not isinstance(block.get('text'), str):
                raise ValueError('Every primary source block needs a string text.')
            if not block.get('block_id') or not block.get('locator'):
                raise ValueError('Every primary source block needs block_id and locator.')
            identity = (collection, block['block_id'])
            if identity in seen:
                raise ValueError('Duplicate source block ID in one document/collection.')
            seen.add(identity)
            if not _eligible(block, collection):
                audit['excluded_unreviewed_blocks'] += 1
                audit['excluded_unreviewed_characters'] += len(block['text'])
                continue
            audit['eligible_source_blocks'] += 1
            audit['eligible_source_characters'] += len(block['text'])
            key = _group_key(block, collection, document)
            if current and key != current_key:
                yield current_key, current
                current = []
            current_key = key
            current.append((collection, block))
        if current:
            yield current_key, current
            current, current_key = [], None


def _source_stream(items):
    pieces, ranges, units = [], [], []
    position = 0
    current_unit, unit_start = None, 0
    for collection, block in items:
        if pieces:
            pieces.append('\n')
            position += 1
        start = position
        pieces.append(block['text'])
        position += len(block['text'])
        ranges.append((start, position, collection, block))
        unit = ('row', block['row']) if block.get('table_id') and block.get('row') is not None else ('block', block['block_id'])
        if current_unit is not None and unit != current_unit:
            units.append((unit_start, start - 1))
            unit_start = start
        current_unit = unit
    if pieces:
        units.append((unit_start, position))
    return ''.join(pieces), ranges, units


def _spans(ranges, start, end, chunk_offset=0):
    result = []
    for begin, finish, collection, block in ranges:
        if finish <= start:
            continue
        if begin >= end:
            break
        left, right = max(begin, start), min(finish, end)
        if left >= right:
            continue
        span = {'block_id': block['block_id'], 'locator': block['locator'],
                'char_start': left - begin, 'char_end': right - begin,
                'chunk_start': chunk_offset + left - start,
                'chunk_end': chunk_offset + right - start, 'collection': collection}
        for name in ['page', 'section', 'paragraph', 'table_id', 'row', 'col']:
            if block.get(name) is not None:
                span[name] = block[name]
        result.append(span)
    return result


def _select_end(text, prefix, start, hard_end, unit_ends, native_table=False):
    if hard_end >= len(text):
        return len(text)
    capacity = prefix[hard_end] - prefix[start]
    if native_table:
        # A small complete row is better than a larger chunk that cuts the next
        # otherwise byte-bounded label/value pair. Large individual rows still
        # fall through to textual or hard splits with an explicit quality flag.
        candidates = [end for end in unit_ends if start < end <= hard_end]
        if candidates:
            return candidates[-1]
    # Keep complete native rows/paragraphs whenever they occupy a useful chunk.
    low = bisect_left(prefix, prefix[start] + max(1, int(capacity * 0.40)))
    candidates = [end for end in unit_ends if low <= end <= hard_end and end > start]
    if candidates:
        return candidates[-1]
    # A long paragraph or PDF page needs local textual boundaries.
    low = bisect_left(prefix, prefix[start] + max(1, int(capacity * 0.60)))
    for delimiter in ['\n', '. ', '。', '다. ', ' ']:
        position = text.rfind(delimiter, low, hard_end)
        if position >= low:
            return position + len(delimiter)
    return hard_end


def _next_start(prefix, previous_start, end, overlap, unit_starts, table):
    if overlap == 0:
        return end
    lower = bisect_left(prefix, prefix[end] - overlap)
    if table:
        # Repeating a complete short row is preferable to a dangling cell tail.
        candidates = [s for s in unit_starts if max(lower, previous_start + 1) <= s < end]
        return candidates[0] if candidates else end
    return max(previous_start + 1, min(lower, end))


def _quality(document, group, key, table_lookup):
    flags = set()
    locations = {b['locator'] for _, b in group}
    boundary, kind, table_id, collection = key
    table = table_lookup.get(table_id) if table_id else None
    if table:
        flags.update(str(w) for w in table.get('warnings', []))
        locations.add(table.get('locator'))
    for warning in document.get('warnings', []):
        if isinstance(warning, str):
            flags.add('document:' + warning)
            continue
        locator = warning.get('locator')
        if not locator:
            flags.add('document:' + warning.get('code', 'SOURCE_WARNING'))
        elif locator in locations:
            flags.add(warning.get('code', 'SOURCE_WARNING'))
    for _, block in group:
        quality = block.get('quality', {})
        if quality.get('replacement_characters') or quality.get('nul_characters') or '\ufffd' in block['text'] or '\x00' in block['text']:
            flags.add('SOURCE_CONTROL_OR_REPLACEMENT_CHARACTERS')
    if kind == 'table':
        status = table.get('structure_status', 'unresolved') if table else 'unresolved'
        if status != 'explicit':
            flags.add('TABLE_STRUCTURE_' + status.upper())
        return flags, status, [table_id]
    if document.get('detected_format') == 'pdf':
        candidates = [t for t in table_lookup.values() if t.get('page') == boundary[1]]
        if candidates:
            flags.add('PDF_TABLE_CANDIDATES_NOT_VISUALLY_VERIFIED')
            status = 'unresolved' if any(t.get('structure_status') == 'unresolved' for t in candidates) else 'heuristic'
            return flags, status, [t['table_id'] for t in candidates]
    return flags, None, []


def _make_chunks(document, aliases, config, audit):
    maximum, overlap = config['chunk_size_bytes'], config['overlap_bytes']
    table_lookup = {t['table_id']: t for t in document.get('tables', []) if t.get('table_id')}
    for key, group in _groups(document, audit):
        text, ranges, units = _source_stream(group)
        if not text:
            continue
        prefix = _prefix_bytes(text)
        boundary, kind, table_id, collection = key
        unit_ends = [end for _, end in units]
        unit_starts = [start for start, _ in units]
        synthetic_separators = {begin - 1 for begin, _, _, _ in ranges[1:]}
        flags, structure_status, table_ids = _quality(document, group, key, table_lookup)
        table_header_end = units[0][1] if kind == 'table' and units else 0
        # Native coordinates identify a row, not its semantic role as a header.
        # Repeating this first-row context shares (rather than adds to) overlap.
        header_usable = table_header_end > 0 and prefix[table_header_end] + 1 <= min(maximum // 3, overlap)
        repeated_context_bytes = prefix[table_header_end] + 1 if header_usable else 0
        if header_usable:
            flags.add('FIRST_ROW_CONTEXT_IS_NOT_A_VERIFIED_COLUMN_HEADER')
        if kind == 'table' and not header_usable and len(units) > 1:
            flags.add('TABLE_HEADER_TOO_LARGE_TO_REPEAT')
        oversized_units = sum(prefix[end] - prefix[start] > maximum for start, end in units)
        if kind == 'table' and oversized_units:
            flags.add('OVERSIZED_TABLE_ROW_SPLIT')
            audit['oversized_table_rows_split'] += oversized_units
        start, last_emitted_end = 0, 0
        while start < len(text):
            while start in synthetic_separators:
                start += 1
            if start >= len(text):
                break
            repeat_header = header_usable and start >= table_header_end and start > 0
            header = text[:table_header_end] if repeat_header else ''
            header_bytes = len(header.encode('utf-8')) + (1 if header else 0)
            budget = maximum - header_bytes
            if header:
                first_end = next((end for end in unit_ends if end > start), len(text))
                first_row_bytes = prefix[first_end] - prefix[start]
                if budget < first_row_bytes <= maximum:
                    # Optional first-row context must never split an intact row.
                    header, header_bytes, budget = '', 0, maximum
            hard_end = bisect_right(prefix, prefix[start] + budget) - 1
            if hard_end <= start:
                raise ValueError('Byte budget cannot hold a source character.')
            end = _select_end(text, prefix, start, hard_end, unit_ends, kind == 'table')
            if end <= last_emitted_end:
                # A cluster of tiny rows before a large row may all fit inside
                # the overlap budget. Do not emit an overlap-only chunk: move
                # directly to the first as-yet unseen source character instead.
                start = last_emitted_end
                continue
            body = text[start:end]
            chunk_text = header + '\n' + body if header else body
            spans = _spans(ranges, start, end, len(header) + 1 if header else 0)
            if header:
                spans = _spans(ranges, 0, table_header_end) + spans
                audit['chunks_with_repeated_native_header'] += 1
            if not spans:
                # This is possible only for a synthetic separator; retain progress.
                start = end
                continue
            identity = {'version': VERSION, 'sha256': document['sha256'], 'config': config,
                        'kind': kind, 'source_spans': [(s['collection'], s['block_id'], s['char_start'], s['char_end']) for s in spans]}
            identifier = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
            unique_locations = list(dict.fromkeys(s['locator'] for s in spans))
            locator = unique_locations[0] if len(unique_locations) == 1 else unique_locations[0] + ' … ' + unique_locations[-1]
            metadata = {'file_name': document.get('file_name', ''), 'agency': document.get('agency'),
                        'document_year': document.get('document_year'), 'source_method':
                        'reviewed_visual_transcription' if kind == 'verified_transcription' else 'raw_parser'}
            context = {'metadata_prefix_is_source_evidence': False,
                       'header_repeated_from_original': bool(header),
                       'repeated_first_row_role': 'unverified_context' if header else None,
                       'first_row_repeat_uses_overlap_budget': True,
                       'requested_overlap_is_maximum_bytes': overlap,
                       'row_ids': list(dict.fromkeys(s['row'] for s in spans if s.get('row') is not None))}
            chunk = {'chunk_id': 'rfp:' + document['sha256'][:16] + ':' + identifier[:24],
                     'doc_id': document['doc_id'], 'canonical_doc_id': document['doc_id'],
                     'alias_doc_ids': aliases, 'sha256': document['sha256'],
                     'file_name': metadata['file_name'], 'agency': metadata['agency'],
                     'document_year': metadata['document_year'], 'detected_format': document.get('detected_format'),
                     'kind': kind, 'text': chunk_text, 'locator': locator,
                     'table_ids': table_ids, 'source_method': metadata['source_method'],
                     'quality_flags': sorted(flags), 'length_unit': 'utf8_bytes',
                     'text_bytes': len(chunk_text.encode('utf-8')), 'source_spans': spans,
                     'metadata_context': context}
            if structure_status:
                chunk['structure_status'] = structure_status
            for name in ['page', 'section']:
                values = {b.get(name) for _, b in group}
                if len(values) == 1 and None not in values:
                    chunk[name] = next(iter(values))
            if kind == 'verified_transcription':
                chunk['review_status'] = 'visual_verified'
                context['reviewed_transcriptions'] = [{k: b.get(k) for k in ['block_id', 'review_status', 'review_date', 'bbox', 'bbox_coordinate_system', 'review_note']} for _, b in group]
            fields = [('문서', metadata['file_name']), ('기관', metadata['agency']),
                      ('작성연도', metadata['document_year']), ('위치', locator)]
            chunk['search_text'] = '\n'.join(f'{label}: {value}' for label, value in fields if value is not None and value != '') + '\n\n' + chunk_text
            if chunk['text_bytes'] > maximum:
                raise AssertionError('Chunk byte bound violated.')
            yield chunk
            last_emitted_end = end
            if end == len(text):
                break
            body_overlap = max(0, overlap - repeated_context_bytes) if kind == 'table' else overlap
            start = _next_start(prefix, start, end, body_overlap, unit_starts, kind == 'table')


def build_chunks(documents, config=None):
    """Build canonical chunks and an exact source-character coverage audit.

    Every eligible primary source character is retained. Native table rows and
    paragraphs are preferred split boundaries; overlap can be smaller than the
    requested maximum to preserve a complete row. Header repeats use real source
    spans, not fabricated column labels. Reference views and unreviewed OCR are
    deliberately excluded from the index and counted separately.
    """
    config = _configuration(config)
    by_hash = defaultdict(list)
    doc_ids = set()
    count = 0
    for document in documents:
        count += 1
        if document.get('status') == 'failed':
            raise ValueError('Cannot silently index a failed document: ' + str(document.get('doc_id')))
        if not document.get('doc_id') or document['doc_id'] in doc_ids:
            raise ValueError('Missing or duplicate document ID.')
        doc_ids.add(document['doc_id'])
        if not isinstance(document.get('sha256'), str) or not re.fullmatch(r'[0-9a-fA-F]{64}', document['sha256']):
            raise ValueError('Each document needs a valid raw SHA256.')
        by_hash[document['sha256'].lower()].append(document)
    audit = Counter()
    chunks = []
    missing = []
    covered_characters = 0
    emitted_source_characters = 0
    canonical_ids = []
    for digest, members in sorted(by_hash.items()):
        preferred = next((d for d in members if d['doc_id'] == d.get('canonical_doc_id') and d.get('blocks')), None)
        document = preferred or next((d for d in sorted(members, key=lambda d: d['doc_id']) if d.get('blocks')), None)
        if document is None:
            raise ValueError('No primary blocks for raw content ' + digest)
        canonical_ids.append(document['doc_id'])
        aliases = sorted(d['doc_id'] for d in members)
        audit['excluded_reference_blocks'] += len(document.get('reference_blocks', []))
        sources = {}
        for collection in ['blocks', 'supplemental_blocks']:
            for block in document.get(collection, []) or []:
                if not isinstance(block, dict) or not isinstance(block.get('text'), str) or not block.get('block_id') or not block.get('locator'):
                    raise ValueError('Every primary source block needs block_id, locator and string text.')
                if _eligible(block, collection):
                    sources[(collection, block['block_id'])] = block['text']
        coverage = defaultdict(list)
        for chunk in _make_chunks(document, aliases, config, audit):
            for span in chunk['source_spans']:
                identity = (span['collection'], span['block_id'])
                original = sources[identity][span['char_start']:span['char_end']]
                if original != chunk['text'][span['chunk_start']:span['chunk_end']]:
                    raise AssertionError('Source span does not reproduce chunk text.')
                coverage[identity].append((span['char_start'], span['char_end']))
                emitted_source_characters += len(original)
            chunks.append(chunk)
        for identity, original in sources.items():
            end, total = 0, 0
            for start, finish in sorted(coverage[identity]):
                if start > end:
                    missing.append({'doc_id': document['doc_id'], 'collection': identity[0], 'block_id': identity[1], 'char_start': end, 'char_end': start})
                if finish > end:
                    total += finish - max(start, end)
                    end = finish
            if end < len(original):
                missing.append({'doc_id': document['doc_id'], 'collection': identity[0], 'block_id': identity[1], 'char_start': end, 'char_end': len(original)})
            covered_characters += total
    identifiers = [c['chunk_id'] for c in chunks]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError('Deterministic chunk IDs collided.')
    result = {
        'version': VERSION, 'config': config, 'length_unit': 'utf8_bytes',
        'input_document_files': count, 'canonical_documents': len(by_hash),
        'duplicate_alias_files': count - len(by_hash), 'canonical_doc_ids': canonical_ids,
        'chunks': len(chunks), 'chunk_kind_counts': dict(Counter(c['kind'] for c in chunks)),
        **dict(audit), 'covered_source_characters': covered_characters,
        'emitted_source_characters_including_overlap': emitted_source_characters,
        'uncovered_source_characters': sum(x['char_end'] - x['char_start'] for x in missing),
        'uncovered_ranges': missing, 'source_spans_valid': True,
        'all_eligible_source_characters_covered': not missing,
        'max_chunk_text_bytes': max((c['text_bytes'] for c in chunks), default=0),
        'min_chunk_text_bytes': min((c['text_bytes'] for c in chunks), default=0),
        'limitations': ['Byte lengths are not model token counts.',
                        'Table/header ordering follows primary parser blocks; native cell coordinates do not certify visual semantics.',
                        'PDF pages remain page-text slices; table candidates are quality metadata, not a second duplicated rendering.',
                        'Overlap is a maximum byte budget and can be reduced to preserve complete native rows.',
                        'A repeated native first row is source context only; its role as a column header is not visually verified.',
                        'All eligible source characters are covered; this does not guarantee one chunk contains every long evidence passage.'],
    }
    return chunks, result
