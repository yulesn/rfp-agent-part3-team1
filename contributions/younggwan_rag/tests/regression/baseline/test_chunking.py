"""Source integrity and boundary behavior, independent of corpus gold answers."""
import copy
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rfp_rag.retrieval.chunking import build_chunks


def block(identifier, text, section='BodyText/Section0', **metadata):
    result = {'block_id': identifier, 'text': text,
              'locator': section + ' / ' + identifier, 'section': section}
    result.update(metadata)
    return result


def document(identifier='d1', blocks=None, **metadata):
    result = {'doc_id': identifier, 'canonical_doc_id': identifier,
              'sha256': hashlib.sha256(identifier.encode()).hexdigest(),
              'file_name': identifier + '.hwp', 'detected_format': 'hwp',
              'status': 'parsed', 'blocks': blocks or [], 'tables': [],
              'reference_blocks': [], 'supplemental_blocks': [], 'warnings': []}
    result.update(metadata)
    return result


class SourceAssertions(unittest.TestCase):
    def assert_source_integrity(self, documents, chunks, audit, maximum):
        sources = {(d['doc_id'], collection, b['block_id']): b['text']
                   for d in documents for collection in ['blocks', 'supplemental_blocks']
                   for b in d.get(collection, [])}
        for chunk in chunks:
            self.assertLessEqual(len(chunk['text'].encode('utf-8')), maximum)
            self.assertEqual(chunk['text_bytes'], len(chunk['text'].encode('utf-8')))
            self.assertEqual(chunk['length_unit'], 'utf8_bytes')
            self.assertFalse(chunk['metadata_context']['metadata_prefix_is_source_evidence'])
            for span in chunk['source_spans']:
                key = (chunk['doc_id'], span['collection'], span['block_id'])
                self.assertEqual(sources[key][span['char_start']:span['char_end']],
                                 chunk['text'][span['chunk_start']:span['chunk_end']])
        self.assertTrue(audit['source_spans_valid'])
        self.assertTrue(audit['all_eligible_source_characters_covered'])
        self.assertEqual(audit['uncovered_source_characters'], 0)
        self.assertEqual(audit['covered_source_characters'], audit['eligible_source_characters'])

    def test_large_korean_unicode_is_byte_bounded_and_source_complete(self):
        text = ('가나다🙂한글 비용 99억원.\n입찰 자료\x00\ufffd ' * 120) + '끝'
        docs = [document(blocks=[block('large', text)])]
        chunks, audit = build_chunks(docs, {'chunk_size_bytes': 77, 'overlap_bytes': 13})
        self.assertGreater(len(chunks), 50)
        self.assert_source_integrity(docs, chunks, audit, 77)
        self.assertGreater(audit['emitted_source_characters_including_overlap'], len(text))
        self.assertTrue(any('SOURCE_CONTROL_OR_REPLACEMENT_CHARACTERS' in c['quality_flags'] for c in chunks))
        # Union of spans recovers every character, including raw control values.
        coverage = set()
        for chunk in chunks:
            for s in chunk['source_spans']:
                coverage.update(range(s['char_start'], s['char_end']))
        self.assertEqual(coverage, set(range(len(text))))

    def test_smallest_valid_budget_does_not_cut_utf8_codepoints(self):
        docs = [document(blocks=[block('unicode', '🙂가A🙂나B' * 9)])]
        chunks, audit = build_chunks(docs, {'chunk_size_bytes': 4, 'overlap_bytes': 3})
        self.assert_source_integrity(docs, chunks, audit, 4)

    def test_document_page_section_and_unknown_boundaries(self):
        pdf = document('pdf', [
            {'block_id': 'p1', 'text': '페이지 하나', 'page': 1, 'locator': 'PDF p.1'},
            {'block_id': 'p2', 'text': '페이지 둘', 'page': 2, 'locator': 'PDF p.2'}], detected_format='pdf')
        hwp = document('hwp', [block('s0', '섹션 하나'), block('s1', '섹션 둘', section='BodyText/Section1')])
        unknown = document('unknown', [
            {'block_id': 'x1', 'text': '위치 불명 1', 'locator': 'unknown 1'},
            {'block_id': 'x2', 'text': '위치 불명 2', 'locator': 'unknown 2'}])
        docs = [pdf, hwp, unknown]
        chunks, audit = build_chunks(docs)
        self.assertEqual(len(chunks), 6)
        self.assert_source_integrity(docs, chunks, audit, 1800)
        for chunk in chunks:
            pages = {s['page'] for s in chunk['source_spans'] if 'page' in s}
            sections = {s['section'] for s in chunk['source_spans'] if 'section' in s}
            self.assertLessEqual(len(pages), 1)
            self.assertLessEqual(len(sections), 1)
            self.assertEqual(len(chunk['source_spans']), 1)

    def test_short_native_table_cells_keep_label_value_pairs(self):
        rows = [('항목', '요구 내용'), ('예산', '총 1,000,000원'),
                ('제출 방식', '나라장터 온라인 제출'), ('제출 기한', '2026년 11월 20일'),
                ('평가 방식', '기술 90점 가격 10점')]
        blocks = [block(f'r{r}c{col}', text, table_id='t1', row=r, col=col)
                  for r, values in enumerate(rows) for col, text in enumerate(values)]
        docs = [document(blocks=blocks, tables=[{'table_id': 't1', 'structure_status': 'explicit',
                 'cells': [{'text': 'STRUCTURED_TABLE_COPY_MUST_NOT_BE_INDEXED'}]}],
                 reference_blocks=[{'text': 'REFERENCE_VIEW_MUST_NOT_BE_INDEXED'}])]
        chunks, audit = build_chunks(docs, {'chunk_size_bytes': 120, 'overlap_bytes': 40})
        self.assert_source_integrity(docs, chunks, audit, 120)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(audit['excluded_reference_blocks'], 1)
        for r, values in enumerate(rows):
            self.assertTrue(any(all(f'r{r}c{col}' in {s['block_id'] for s in c['source_spans']}
                                    for col in [0, 1]) for c in chunks), 'A short label/value row split')
        self.assertTrue(any(c['metadata_context']['header_repeated_from_original'] for c in chunks))
        for c in chunks:
            self.assertEqual(c['kind'], 'table')
            self.assertEqual(c['structure_status'], 'explicit')
            self.assertNotIn('STRUCTURED_TABLE_COPY', c['text'])
            self.assertNotIn('REFERENCE_VIEW', c['text'])
            if c['metadata_context']['header_repeated_from_original']:
                self.assertEqual(c['metadata_context']['repeated_first_row_role'], 'unverified_context')

    def test_oversized_native_row_is_split_and_never_lost(self):
        docs = [document(blocks=[block('hugecell', '긴표내용' * 100, table_id='t1', row=0, col=0)],
                         tables=[{'table_id': 't1', 'structure_status': 'unresolved'}])]
        chunks, audit = build_chunks(docs, {'chunk_size_bytes': 85, 'overlap_bytes': 15})
        self.assert_source_integrity(docs, chunks, audit, 85)
        self.assertEqual(audit['oversized_table_rows_split'], 1)
        self.assertTrue(all('OVERSIZED_TABLE_ROW_SPLIT' in c['quality_flags'] for c in chunks))
        self.assertTrue(all('TABLE_STRUCTURE_UNRESOLVED' in c['quality_flags'] for c in chunks))

    def test_short_row_is_kept_intact_even_after_a_tiny_row(self):
        # A source row that fits the cap must not be cut to fill a small prior
        # row's chunk, nor to accommodate optional repeated first-row context.
        docs = [document(blocks=[
            block('header', 'X', table_id='t1', row=0, col=0),
            block('label', 'L', table_id='t1', row=1, col=0),
            block('value', 'V' * 98, table_id='t1', row=1, col=1),
            block('tail', '종료', table_id='t1', row=2, col=0)])]
        chunks, audit = build_chunks(docs, {'chunk_size_bytes': 100, 'overlap_bytes': 10})
        self.assert_source_integrity(docs, chunks, audit, 100)
        full_row = [c for c in chunks if {s['block_id'] for s in c['source_spans']} >= {'label', 'value'}]
        self.assertEqual(len(full_row), 1)
        self.assertEqual([s['char_end'] for s in full_row[0]['source_spans'] if s['block_id'] == 'value'], [98])

    def test_tiny_rows_before_oversized_row_do_not_emit_overlap_only_duplicates(self):
        blocks = [block(f'small{r}', '작은 행', table_id='t1', row=r, col=0) for r in range(6)]
        blocks.append(block('oversized', '큰행' * 1000, table_id='t1', row=6, col=0))
        docs = [document(blocks=blocks)]
        chunks, audit = build_chunks(docs)
        self.assert_source_integrity(docs, chunks, audit, 1800)
        self.assertEqual(len(chunks), len({c['chunk_id'] for c in chunks}))
        large_chunks = [c for c in chunks if any(s['block_id'] == 'oversized' for s in c['source_spans'])]
        self.assertGreater(len(large_chunks), 1)
        self.assertEqual(len(chunks) - len(large_chunks), 1)

    def test_zero_overlap_does_not_repeat_table_context(self):
        blocks = [block(f'b{r}', '예산 수치와 요구 조건 ' * 2, table_id='t1', row=r, col=0) for r in range(5)]
        docs = [document(blocks=blocks)]
        chunks, audit = build_chunks(docs, {'chunk_size_bytes': 90, 'overlap_bytes': 0})
        self.assert_source_integrity(docs, chunks, audit, 90)
        self.assertFalse(any(c['metadata_context']['header_repeated_from_original'] for c in chunks))
        self.assertEqual(audit['emitted_source_characters_including_overlap'], audit['covered_source_characters'])

    def test_varied_table_rows_within_budget_are_recoverable_together(self):
        blocks = []
        for row in range(40):
            # Rows use 4..100 bytes, exposing unlucky packing boundaries.
            blocks.extend([block(f'label{row}', 'L', table_id='t1', row=row, col=0),
                           block(f'value{row}', 'V' * (2 + (row * 17) % 97), table_id='t1', row=row, col=1)])
        docs = [document(blocks=blocks)]
        chunks, audit = build_chunks(docs, {'chunk_size_bytes': 100, 'overlap_bytes': 20})
        self.assert_source_integrity(docs, chunks, audit, 100)
        lengths = {b['block_id']: len(b['text']) for b in blocks}
        for row in range(40):
            expected = {f'label{row}', f'value{row}'}
            self.assertTrue(any(expected <= {s['block_id'] for s in c['source_spans']
                                             if s['char_start'] == 0 and s['char_end'] == lengths[s['block_id']]}
                                for c in chunks), f'Row {row} was unnecessarily split')

    def test_aliases_are_canonicalized_once_and_ids_are_reproducible(self):
        canonical = document(blocks=[block('b1', '원문 자료 ' * 30)])
        alias = copy.deepcopy(canonical)
        alias.update(doc_id='alias', canonical_doc_id='d1', is_duplicate_alias=True, file_name='별칭.hwp')
        first, audit = build_chunks([alias, canonical])
        second, audit2 = build_chunks([canonical, alias])
        self.assertEqual(first, second)
        self.assertEqual(audit['input_document_files'], 2)
        self.assertEqual(audit['canonical_documents'], 1)
        self.assertEqual(audit['duplicate_alias_files'], 1)
        self.assertTrue(all(c['doc_id'] == 'd1' and c['alias_doc_ids'] == ['alias', 'd1'] for c in first))
        altered = copy.deepcopy(canonical)
        altered['file_name'] = '이름변경.hwp'
        third, _ = build_chunks([altered])
        self.assertEqual([c['chunk_id'] for c in first], [c['chunk_id'] for c in third])
        changed_budget, _ = build_chunks([canonical], {'chunk_size_bytes': 900})
        self.assertNotEqual(first[0]['chunk_id'], changed_budget[0]['chunk_id'])

    def test_verified_transcription_provenance_and_ocr_exclusion(self):
        d = document('pdf', [{'block_id': 'p1', 'text': '자동 추출 본문', 'page': 1, 'locator': 'PDF p.1'}],
                     detected_format='pdf', warnings=[{'code': 'VECTOR_GLYPH_TEXT_GAP', 'locator': 'PDF p.1'}],
                     tables=[{'table_id': 'pdf_table', 'page': 1, 'structure_status': 'heuristic'}])
        d['supplemental_blocks'] = [
            {'block_id': 'verified', 'text': '예산 10억원', 'page': 1, 'locator': 'PDF p.1 / reviewed',
             'kind': 'verified_transcription', 'source_method': 'reviewed_visual_transcription',
             'review_status': 'visual_verified', 'review_date': '2026-11-11', 'bbox': [1, 2, 3, 4]},
            {'block_id': 'ocr', 'text': '추측 금액 88억원', 'page': 1, 'locator': 'PDF p.1 / OCR',
             'kind': 'ocr', 'source_method': 'unreviewed_ocr', 'review_status': 'unreviewed'}]
        chunks, audit = build_chunks([d])
        self.assert_source_integrity([d], chunks, audit, 1800)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(audit['excluded_unreviewed_blocks'], 1)
        self.assertFalse(any('88억원' in c['text'] for c in chunks))
        reviewed = next(c for c in chunks if c['kind'] == 'verified_transcription')
        self.assertEqual(reviewed['source_method'], 'reviewed_visual_transcription')
        self.assertEqual(reviewed['review_status'], 'visual_verified')
        self.assertEqual(reviewed['metadata_context']['reviewed_transcriptions'][0]['bbox'], [1, 2, 3, 4])
        automatic = next(c for c in chunks if c['kind'] == 'prose')
        self.assertIn('VECTOR_GLYPH_TEXT_GAP', automatic['quality_flags'])
        self.assertIn('PDF_TABLE_CANDIDATES_NOT_VISUALLY_VERIFIED', automatic['quality_flags'])
        self.assertEqual(automatic['structure_status'], 'heuristic')

    def test_empty_blocks_whitespace_and_invalid_inputs(self):
        docs = [document(blocks=[block('empty', ''), block('spaces', ' \n\t  '), block('text', '유효 본문')])]
        chunks, audit = build_chunks(docs)
        self.assert_source_integrity(docs, chunks, audit, 1800)
        for settings in [{'chunk_size_bytes': 3}, {'chunk_size_bytes': True},
                         {'chunk_size_bytes': 30, 'overlap_bytes': 30}, {'overlap_bytes': -1}]:
            with self.assertRaises(ValueError):
                build_chunks(docs, settings)
        invalid = document(blocks=[{'text': 'broken'}])
        with self.assertRaises(ValueError):
            build_chunks([invalid])
        with self.assertRaises(ValueError):
            build_chunks([document(blocks=[block('same', 'x'), block('same', 'y')])])


if __name__ == '__main__':
    unittest.main()
