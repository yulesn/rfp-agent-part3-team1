"""새 청크가 실제 원문을 보존하는지 검색기와 독립적으로 점검합니다."""
from collections import defaultdict
import re


def _indexable(block, collection):
    method = str(block.get('source_method', '')).lower()
    kind = str(block.get('kind', '')).lower()
    if block.get('reference_view') or 'ocr' in method or 'ocr' in kind or block.get('review_status') == 'unreviewed':
        return False
    if collection == 'supplemental_blocks' or method == 'reviewed_visual_transcription' or kind == 'verified_transcription':
        return method == 'reviewed_visual_transcription' and block.get('review_status') == 'visual_verified'
    return True


def _boundary(block):
    if block.get('page') is not None:
        return 'page', block['page']
    section = block.get('section')
    if section is None:
        match = re.match(r'(BodyText/Section\d+|Contents/section\d+\.xml)', block['locator'])
        section = match.group(1) if match else None
    return ('section', section) if section else ('unknown_block', block['block_id'])


def _merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(end, merged[-1][1])
    return merged


def validate_chunks(documents, chunks, max_text_bytes=1800):
    by_id = {d['doc_id']: d for d in documents}
    source = {}
    expected = {}
    for document in documents:
        if document.get('is_duplicate_alias'):
            continue
        for collection in ['blocks', 'supplemental_blocks']:
            for block in document.get(collection, []):
                if not _indexable(block, collection):
                    continue
                key = document['doc_id'], collection, block['block_id']
                source[key] = block
                expected[key] = block['text']
    seen_ids = set()
    covered = defaultdict(list)
    errors = []
    bytes_indexed = 0
    source_methods = defaultdict(int)

    def check(condition, message):
        if not condition:
            errors.append(message)

    for chunk in chunks:
        previous_error_count = len(errors)
        pending_coverage = []
        cid = chunk.get('chunk_id')
        check(bool(cid) and cid not in seen_ids, f'청크 ID 누락/중복: {cid}')
        seen_ids.add(cid)
        did = chunk.get('doc_id')
        document = by_id.get(did)
        if document is None:
            errors.append(f'{cid}: 원본 문서 ID가 없습니다.')
            continue
        check(not document.get('is_duplicate_alias'), f'{cid}: 동일 원문 사본을 중복 인덱싱했습니다.')
        check(chunk.get('sha256') == document['sha256'], f'{cid}: 원본 해시가 다릅니다.')
        text = chunk.get('text', '')
        count = len(text.encode('utf-8'))
        bytes_indexed += count
        check(0 < count <= max_text_bytes, f'{cid}: 본문 byte 예산을 벗어났습니다: {count}')
        check(chunk.get('text_bytes') == count, f'{cid}: 본문 byte 길이 기록이 다릅니다.')
        check(chunk.get('length_unit') == 'utf8_bytes', f'{cid}: 문자/byte/token 단위가 구분되지 않았습니다.')
        intervals_in_chunk = []
        pages, sections, collections, boundaries, provenances = set(), set(), set(), set(), set()
        for span in chunk.get('source_spans', []):
            collection = span.get('collection')
            check(collection in ['blocks', 'supplemental_blocks'], f'{cid}: 인용 전용 reference view 또는 미확인 보완 자료를 인덱싱했습니다.')
            key = did, collection, span.get('block_id')
            block = source.get(key)
            if block is None:
                errors.append(f'{cid}: 존재하지 않거나 인덱싱 대상이 아닌 원문 블록입니다.')
                continue
            a, b, x, y = (span.get(k) for k in ['char_start', 'char_end', 'chunk_start', 'chunk_end'])
            valid = all(type(v) is int for v in [a, b, x, y]) and 0 <= a < b <= len(block['text']) and 0 <= x < y <= len(text)
            if not valid:
                errors.append(f'{cid}: 원문 또는 청크 문자 범위가 잘못되었습니다.')
                continue
            check(block['text'][a:b] == text[x:y], f'{cid}: 청크 내용이 연결된 원문과 다릅니다.')
            check(span.get('locator') == block['locator'], f'{cid}: 원문 위치가 다릅니다.')
            for field in ['page', 'section']:
                check(span.get(field) == block.get(field), f'{cid}: 원문 {field} 위치 메타데이터가 다릅니다.')
            pending_coverage.append((key, (a, b)))
            intervals_in_chunk.append((x, y))
            boundaries.add(_boundary(block))
            if block.get('page') is not None:
                pages.add(block['page'])
            if block.get('section') is not None:
                sections.add(block['section'])
            collections.add(collection)
            provenances.add('reviewed' if block.get('source_method') == 'reviewed_visual_transcription' else 'automatic')
        check(bool(intervals_in_chunk), f'{cid}: 원문 근거 범위가 없습니다.')
        check(len(pages) <= 1, f'{cid}: PDF 물리 페이지를 넘었습니다.')
        check(len(sections) <= 1, f'{cid}: 원본 내부 섹션을 넘었습니다.')
        check(len(collections) <= 1, f'{cid}: 자동 추출과 시각 검수 전사를 혼합했습니다.')
        check(len(boundaries) <= 1, f'{cid}: 출처가 다른 페이지/섹션 또는 위치 미상 블록을 합쳤습니다.')
        check(len(provenances) <= 1, f'{cid}: 자동 추출과 시각 검수 전사 출처를 혼합했습니다.')
        for field, values in [('page', pages), ('section', sections)]:
            if field in chunk:
                check(len(values) == 1 and chunk[field] in values, f'{cid}: 청크 {field} 위치 메타데이터가 다릅니다.')
        if 'reviewed' in provenances:
            check(chunk.get('source_method') == 'reviewed_visual_transcription' and chunk.get('kind') == 'verified_transcription', f'{cid}: 시각 검수 전사 표시가 사라졌습니다.')
        last = 0
        for a, b in _merge(intervals_in_chunk):
            check(not text[last:a].strip(), f'{cid}: 원문 범위로 설명되지 않은 텍스트를 삽입했습니다.')
            last = b
        check(not text[last:].strip(), f'{cid}: 원문 범위로 설명되지 않은 끝 텍스트가 있습니다.')
        if len(errors) == previous_error_count:
            for key, interval in pending_coverage:
                covered[key].append(interval)
        source_methods[chunk.get('source_method', 'unknown')] += 1
    uncovered = []
    total_characters = sum(sum(not ch.isspace() for ch in text) for text in expected.values())
    missing_characters = 0
    for key, text in expected.items():
        last = 0
        gaps = []
        for a, b in _merge(covered[key]):
            if text[last:a].strip():
                gaps.append((last, a))
            last = b
        if text[last:].strip():
            gaps.append((last, len(text)))
        if gaps:
            count = sum(sum(not ch.isspace() for ch in text[a:b]) for a, b in gaps)
            missing_characters += count
            uncovered.append({'doc_id': key[0], 'collection': key[1], 'block_id': key[2],
                              'locator': source[key]['locator'], 'missing_non_whitespace_characters': count,
                              'missing_ranges': gaps})
    check(not uncovered, f'청킹 중 원문의 비공백 문자 누락: {len(uncovered)}개 블록')
    return {'valid': not errors, 'errors': errors, 'source_files': len(documents),
            'unique_documents': len({d['sha256'] for d in documents}), 'chunks': len(chunks),
            'source_non_whitespace_characters': total_characters,
            'uncovered_non_whitespace_characters': missing_characters,
            'source_character_coverage': (total_characters - missing_characters) / total_characters if total_characters else None,
            'uncovered_blocks': uncovered, 'indexable_bytes_with_overlap': bytes_indexed,
            'source_methods': dict(source_methods),
            'scope': '이미 파싱한 원문 문자·버전·위치·길이·분리 보존. PDF 그림/표의 시각적 의미 전체에 대한 검증은 아님.'}
