"""Independent source audits with deliberately false provenance and omissions."""
import copy
import unittest

from rfp_rag.retrieval.quality import validate_chunks


def block(identifier='b1', text='원문 예산 100원', **metadata):
    result = {'block_id': identifier, 'text': text, 'locator': 'PDF p.1 / ' + identifier,
              'page': 1}
    result.update(metadata)
    return result


def document(blocks=None, **metadata):
    result = {'doc_id': 'd1', 'canonical_doc_id': 'd1', 'sha256': 'a' * 64,
              'blocks': [block()] if blocks is None else blocks,
              'supplemental_blocks': [], 'reference_blocks': []}
    result.update(metadata)
    return result


def chunk(source_block=None, *, cid='c1', collection='blocks', start=0, end=None,
          **metadata):
    source_block = source_block or block()
    end = len(source_block['text']) if end is None else end
    text = source_block['text'][start:end]
    result = {'chunk_id': cid, 'doc_id': 'd1', 'sha256': 'a' * 64, 'text': text,
              'text_bytes': len(text.encode()), 'length_unit': 'utf8_bytes',
              'source_method': 'automatic_pdf_text', 'kind': 'prose',
              'source_spans': [{'collection': collection, 'block_id': source_block['block_id'],
                  'locator': source_block['locator'], 'page': source_block.get('page'),
                  'char_start': start, 'char_end': end, 'chunk_start': 0, 'chunk_end': len(text)}]}
    result.update(metadata)
    return result


def join_chunks(first, second):
    result = copy.deepcopy(first)
    offset = len(result['text']) + 1
    result['text'] += '\n' + second['text']
    result['text_bytes'] = len(result['text'].encode())
    for span in copy.deepcopy(second['source_spans']):
        span['chunk_start'] += offset
        span['chunk_end'] += offset
        result['source_spans'].append(span)
    return result


class QualityAuditTests(unittest.TestCase):
    def assert_invalid(self, documents, chunks):
        result = validate_chunks(documents, chunks)
        self.assertFalse(result['valid'])
        self.assertTrue(result['errors'])
        return result

    def test_exact_source_is_valid(self):
        result = validate_chunks([document()], [chunk()])
        self.assertTrue(result['valid'])
        self.assertEqual(result['source_character_coverage'], 1)

    def test_source_gap_counts_only_missing_substantive_characters(self):
        b = block(text='가 나\n다라')
        result = self.assert_invalid([document([b])], [chunk(b, end=1)])
        self.assertEqual(result['uncovered_non_whitespace_characters'], 3)
        self.assertEqual(result['source_character_coverage'], .25)
        self.assertEqual(result['uncovered_blocks'][0]['missing_ranges'], [(1, len(b['text']))])

    def test_whitespace_gap_does_not_invent_missing_evidence(self):
        b = block(text='가 \n\t 나')
        result = validate_chunks([document([b])], [chunk(b, end=1), chunk(b, cid='c2', start=5)])
        self.assertTrue(result['valid'])
        self.assertEqual(result['uncovered_non_whitespace_characters'], 0)

    def test_overlapping_spans_do_not_double_count_source_coverage(self):
        b = block(text='가나다라')
        result = validate_chunks([document([b])], [chunk(b, end=3), chunk(b, cid='c2', start=1)])
        self.assertTrue(result['valid'])
        self.assertEqual(result['source_non_whitespace_characters'], 4)
        self.assertEqual(result['source_character_coverage'], 1)

    def test_wrong_source_hash_is_rejected(self):
        self.assert_invalid([document()], [chunk(sha256='b' * 64)])

    def test_unknown_document_id_is_rejected(self):
        self.assert_invalid([document()], [chunk(doc_id='other-document')])

    def test_missing_and_duplicate_chunk_ids_are_rejected(self):
        for chunks in [[chunk(chunk_id=None)], [chunk(), chunk()]]:
            self.assert_invalid([document()], chunks)

    def test_unknown_block_and_wrong_locator_are_rejected(self):
        for key, value in [('block_id', 'invented-block'), ('locator', 'invented location')]:
            c = chunk()
            c['source_spans'][0][key] = value
            self.assert_invalid([document()], [c])

    def test_negative_reversed_and_out_of_range_source_spans_are_rejected(self):
        for start, end in [(-1, 1), (2, 1), (0, 1000), (0, 0)]:
            c = chunk()
            c['source_spans'][0].update(char_start=start, char_end=end)
            self.assert_invalid([document()], [c])

    def test_out_of_range_and_misaligned_chunk_spans_are_rejected(self):
        for start, end in [(-1, 1), (0, 1000), (1, 2)]:
            c = chunk()
            c['source_spans'][0].update(chunk_start=start, chunk_end=end)
            self.assert_invalid([document()], [c])

    def test_substituting_a_different_budget_in_exact_source_slice_is_rejected(self):
        c = chunk(text='원문 예산 999원')
        self.assert_invalid([document()], [c])

    def test_synthetic_prefix_is_not_treated_as_original_source(self):
        c = chunk()
        prefix = '가짜 발주기관\n'
        c['text'] = prefix + c['text']
        c['text_bytes'] = len(c['text'].encode())
        for span in c['source_spans']:
            span['chunk_start'] += len(prefix)
            span['chunk_end'] += len(prefix)
        self.assert_invalid([document()], [c])

    def test_bytes_and_units_and_budget_are_verified(self):
        for metadata in [{'text_bytes': 1}, {'length_unit': 'characters'}]:
            self.assert_invalid([document()], [chunk(**metadata)])
        self.assertFalse(validate_chunks([document()], [chunk()], max_text_bytes=1)['valid'])

    def test_reference_view_is_excluded_without_becoming_a_source_gap(self):
        reference = block('reference', '인용 전용 통합문')
        d = document(reference_blocks=[reference])
        self.assertTrue(validate_chunks([d], [chunk()])['valid'])
        self.assert_invalid([d], [chunk(), chunk(reference, cid='ref', collection='reference_blocks')])

    def test_unreviewed_supplemental_ocr_is_excluded(self):
        ocr = block('ocr', '추측 금액', source_method='local_ocr', review_status='unreviewed')
        d = document(supplemental_blocks=[ocr])
        self.assertTrue(validate_chunks([d], [chunk()])['valid'])
        self.assert_invalid([d], [chunk(), chunk(ocr, cid='ocr', collection='supplemental_blocks')])

    def test_ocr_method_cannot_be_relabelled_by_only_changing_review_flags(self):
        ocr = block('ocr', '추측 금액', source_method='local_ocr', review_status='visual_verified')
        d = document(supplemental_blocks=[ocr])
        c = chunk(ocr, cid='ocr', collection='supplemental_blocks',
                  source_method='reviewed_visual_transcription', kind='verified_transcription')
        self.assert_invalid([d], [chunk(), c])

    def test_unreviewed_primary_ocr_is_excluded_from_expected_source(self):
        ocr = block('ocr', '미검수 OCR', source_method='unreviewed_ocr', review_status='unreviewed')
        d = document([block(), ocr])
        self.assertTrue(validate_chunks([d], [chunk()])['valid'])
        self.assert_invalid([d], [chunk(), chunk(ocr, cid='ocr')])

    def test_verified_transcription_has_to_keep_its_source_method(self):
        verified = block('verified', '검수 완료 예산', source_method='reviewed_visual_transcription',
                         review_status='visual_verified')
        d = document(supplemental_blocks=[verified])
        c = chunk(verified, cid='verified', collection='supplemental_blocks',
                  source_method='reviewed_visual_transcription', kind='verified_transcription')
        self.assertTrue(validate_chunks([d], [chunk(), c])['valid'])
        c['source_method'] = 'automatic_pdf_text'
        self.assert_invalid([d], [chunk(), c])

    def test_pdf_pages_cannot_be_merged_even_with_matching_text(self):
        first, second = block('p1', '첫페이지', page=1), block('p2', '둘째페이지', page=2)
        c = join_chunks(chunk(first), chunk(second, cid='c2'))
        result = self.assert_invalid([document([first, second])], [c])
        self.assertTrue(any('페이지' in message for message in result['errors']))

    def test_internal_sections_cannot_be_merged(self):
        first = block('s0', '첫 섹션', page=None, section='section0')
        second = block('s1', '다음 섹션', page=None, section='section1')
        c = join_chunks(chunk(first), chunk(second, cid='c2'))
        self.assert_invalid([document([first, second])], [c])

    def test_forged_page_metadata_is_rejected(self):
        c = chunk(page=99)
        c['source_spans'][0]['page'] = 99
        self.assert_invalid([document()], [c])

    def test_unknown_boundaries_are_not_permission_to_merge_blocks(self):
        first, second = block('a', '위치 불명 하나', page=None), block('b', '위치 불명 둘', page=None)
        c = join_chunks(chunk(first), chunk(second, cid='c2'))
        self.assert_invalid([document([first, second])], [c])

    def test_automatic_and_verified_source_collections_cannot_be_mixed(self):
        verified = block('verified', '검수 전사', source_method='reviewed_visual_transcription',
                         review_status='visual_verified')
        d = document(supplemental_blocks=[verified])
        c = join_chunks(chunk(), chunk(verified, cid='verified', collection='supplemental_blocks'))
        self.assert_invalid([d], [c])

    def test_duplicate_source_alias_is_not_indexed_twice(self):
        alias = document(doc_id='alias', canonical_doc_id='d1', is_duplicate_alias=True)
        result = validate_chunks([document(), alias], [chunk()])
        self.assertTrue(result['valid'])
        self.assertEqual(result['unique_documents'], 1)
        self.assert_invalid([document(), alias], [chunk(), chunk(cid='alias', doc_id='alias')])


if __name__ == '__main__':
    unittest.main()
