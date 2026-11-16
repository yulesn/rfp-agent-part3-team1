"""Citation provenance tests use invented sources; no API or evaluation labels."""
import copy
import unittest

from rfp_rag.generation.evidence import ANSWER_SCHEMA, EvidenceError, EvidenceRegistry


def block(identifier='b1', text='사업예산: 100원. 사업기간: 5개월.', page=1, **extra):
    value = {'block_id': identifier, 'text': text, 'locator': f'PDF p.{page} / {identifier}', 'page': page}
    value.update(extra)
    return value


def document(did='d1', source=None, sha=None, **extra):
    value = {'doc_id': did, 'canonical_doc_id': did, 'sha256': sha or ('a' if did == 'd1' else 'b') * 64,
             'file_name': did + '.pdf', 'blocks': [source or block()], 'supplemental_blocks': []}
    value.update(extra)
    return value


def chunk(source=None, did='d1', cid='c1', collection='blocks', start=0, end=None, sha=None, **extra):
    source = source or block()
    end = len(source['text']) if end is None else end
    text = source['text'][start:end]
    span = {'collection': collection, 'block_id': source['block_id'], 'locator': source['locator'],
            'char_start': start, 'char_end': end, 'chunk_start': 0, 'chunk_end': len(text)}
    for field in ('page', 'section'):
        if field in source:
            span[field] = source[field]
    value = {'chunk_id': cid, 'doc_id': did, 'canonical_doc_id': did,
             'sha256': sha or ('a' if did == 'd1' else 'b') * 64, 'file_name': did + '.pdf',
             'kind': 'prose', 'source_method': 'raw_parser', 'text': text,
             'locator': source['locator'], 'length_unit': 'utf8_bytes',
             'text_bytes': len(text.encode('utf-8')), 'source_spans': [span]}
    for field in ('page', 'section'):
        if field in source:
            value[field] = source[field]
    value.update(extra)
    return value


def answer(quote='사업예산: 100원.', text='예산은 100원입니다.', did='d1', cid='c1', status='answered', missing=None):
    return {'status': status, 'claims': [{'doc_id': did, 'category': 'budget', 'text': text,
            'citations': [{'chunk_id': cid, 'quote': quote}]}], 'missing_information': missing or []}


def join(first, second):
    value = copy.deepcopy(first)
    offset = len(value['text']) + 1
    value['text'] += '\n' + second['text']
    value['text_bytes'] = len(value['text'].encode('utf-8'))
    for span in copy.deepcopy(second['source_spans']):
        span['chunk_start'] += offset; span['chunk_end'] += offset
        value['source_spans'].append(span)
    return value


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.doc, self.chunk = document(), chunk()
        self.registry = EvidenceRegistry([self.doc], [self.chunk])

    def check(self, payload=None, context=None, allowed=None):
        return self.registry.validate_answer(payload or answer(), [self.chunk] if context is None else context, allowed)

    def invalid(self, payload=None, context=None, allowed=None):
        result = self.check(payload, context, allowed)
        self.assertFalse(result['valid'])
        self.assertEqual(result['claims'], [])
        self.assertEqual(result['status'], 'insufficient_evidence')
        self.assertTrue(result['errors'])
        return result

    def test_schema_has_strict_objects_and_all_required_fields(self):
        def inspect(value):
            if isinstance(value, dict):
                if value.get('type') == 'object':
                    self.assertFalse(value['additionalProperties'])
                    self.assertEqual(set(value['required']), set(value['properties']))
                for child in value.values(): inspect(child)
            elif isinstance(value, list):
                for child in value: inspect(child)
        inspect(ANSWER_SCHEMA)

    def test_valid_quote_has_precise_raw_offsets_not_whole_chunk_offsets(self):
        result = self.check()
        self.assertTrue(result['valid'])
        citation = result['claims'][0]['citations'][0]
        self.assertEqual(citation['quote_start'], 0)
        self.assertEqual(citation['quote_end'], len('사업예산: 100원.'))
        span = citation['source_ranges'][0]
        self.assertEqual(self.doc['blocks'][0]['text'][span['char_start']:span['char_end']], citation['quote'])
        self.assertLess(span['char_end'], len(self.chunk['text']))
        self.assertEqual(citation['sha256'], self.doc['sha256'])

    def test_nonzero_source_and_chunk_offsets_resolve_exactly(self):
        source = block(text='머리말\n사업예산: 100원. 끝말')
        c = chunk(source, start=4)
        reg = EvidenceRegistry([document(source=source)], [c])
        result = reg.validate_answer(answer(), [c])
        self.assertTrue(result['valid'])
        span = result['claims'][0]['citations'][0]['source_ranges'][0]
        self.assertEqual(source['text'][span['char_start']:span['char_end']], '사업예산: 100원.')
        self.assertEqual(span['char_start'], source['text'].index('사업예산'))

    def test_retrieval_annotations_allowed_but_removed_from_context(self):
        hit = copy.deepcopy(self.chunk)
        hit.update(score=.99, rank=1, method='bm25', component_scores={'bm25': 1.0})
        checked = self.registry.validate_chunks([hit])
        self.assertNotIn('score', checked[0])
        self.assertTrue(self.check(context=[hit])['valid'])

    def test_duplicate_hits_are_indexed_once(self):
        self.assertEqual(len(self.registry.validate_chunks([self.chunk, self.chunk])), 1)

    def test_fabricated_quote_rejected(self):
        self.invalid(answer(quote='사업예산: 999원.'))

    def test_whitespace_normalization_does_not_fake_exact_quote(self):
        self.invalid(answer(quote='사업예산:100원.'))

    def test_blank_quote_or_uncited_claim_rejected(self):
        self.invalid(answer(quote=' \n'))
        p = answer(); p['claims'][0]['citations'] = []
        self.invalid(p)

    def test_unknown_and_out_of_context_chunk_rejected(self):
        self.invalid(answer(cid='not-a-chunk'))
        second = chunk(cid='c2')
        reg = EvidenceRegistry([self.doc], [self.chunk, second])
        self.assertFalse(reg.validate_answer(answer(cid='c2'), [self.chunk])['valid'])

    def test_metadata_file_name_is_not_source_quote(self):
        self.invalid(answer(quote='d1.pdf'))

    def test_tampered_hit_text_span_and_metadata_rejected(self):
        for key, value in [('text', '위조된 텍스트'), ('sha256', 'f' * 64),
                           ('file_name', '다른 문서.pdf'), ('locator', 'PDF p.99')]:
            hit = copy.deepcopy(self.chunk); hit[key] = value
            self.invalid(context=[hit])
        hit = copy.deepcopy(self.chunk); hit['source_spans'][0]['char_end'] -= 1
        self.invalid(context=[hit])

    def test_canonical_chunk_mutation_after_registration_rejected(self):
        self.chunk['text'] += ' 위조'
        self.chunk['text_bytes'] = len(self.chunk['text'].encode())
        self.invalid()

    def test_source_mutation_after_registration_rejected(self):
        self.doc['blocks'][0]['text'] = '사업예산: 999원. 사업기간: 5개월.'
        self.invalid()

    def test_wrong_raw_span_offset_rejected_even_if_registered(self):
        for field, value in [('char_start', -1), ('char_end', 999), ('chunk_start', True)]:
            c = chunk(); c['source_spans'][0][field] = value
            reg = EvidenceRegistry([document()], [c])
            self.assertFalse(reg.validate_answer(answer(), [c])['valid'])

    def test_wrong_raw_locator_and_page_rejected(self):
        for field, value in [('locator', '다른 위치'), ('page', 2)]:
            c = chunk(); c['source_spans'][0][field] = value
            reg = EvidenceRegistry([document()], [c])
            self.assertFalse(reg.validate_answer(answer(), [c])['valid'])

    def test_nonwhitespace_synthetic_prefix_rejected(self):
        c = chunk(); c['text'] = '합성예산 999원 ' + c['text']
        extra = len('합성예산 999원 ')
        c['text_bytes'] = len(c['text'].encode())
        c['source_spans'][0]['chunk_start'] += extra; c['source_spans'][0]['chunk_end'] += extra
        reg = EvidenceRegistry([document()], [c])
        self.assertFalse(reg.validate_answer(answer(), [c])['valid'])

    def test_quote_can_cross_raw_blocks_with_whitespace_only_separator(self):
        first, second = block('b1', '사업예산:'), block('b2', '100원.')
        c = join(chunk(first), chunk(second, cid='c2'))
        reg = EvidenceRegistry([document(blocks=[first, second])], [c])
        result = reg.validate_answer(answer(quote='사업예산:\n100원.'), [c])
        self.assertTrue(result['valid'])
        citation = result['claims'][0]['citations'][0]
        self.assertEqual(len(citation['source_ranges']), 2)
        self.assertEqual(citation['source_ranges'][1]['char_start'], 0)

    def test_cross_page_chunk_rejected(self):
        first, second = block('b1', '사업예산:', page=1), block('b2', '100원.', page=2)
        c = join(chunk(first), chunk(second, cid='c2'))
        reg = EvidenceRegistry([document(blocks=[first, second])], [c])
        self.assertFalse(reg.validate_answer(answer(quote='사업예산:\n100원.'), [c])['valid'])

    def test_unknown_boundary_blocks_cannot_be_merged(self):
        first, second = block('b1', '사업예산:'), block('b2', '100원.')
        for b in [first, second]: b.pop('page'); b['locator'] = 'unknown/' + b['block_id']
        c = join(chunk(first), chunk(second, cid='c2'))
        reg = EvidenceRegistry([document(blocks=[first, second])], [c])
        self.assertFalse(reg.validate_answer(answer(quote='사업예산:\n100원.'), [c])['valid'])

    def test_reference_view_and_unreviewed_ocr_rejected(self):
        for collection, change in [('reference_blocks', {}), ('supplemental_blocks', {'review_status': 'unreviewed'}),
                                   ('blocks', {'source_method': 'tesseract_ocr'})]:
            b = block(**change); d = document(blocks=[]); d[collection] = [b]
            c = chunk(b, collection=collection)
            reg = EvidenceRegistry([d], [c])
            self.assertFalse(reg.validate_answer(answer(), [c])['valid'])

    def test_verified_transcription_remains_distinct(self):
        b = block(source_method='reviewed_visual_transcription', review_status='visual_verified')
        d = document(blocks=[], supplemental_blocks=[b])
        c = chunk(b, collection='supplemental_blocks', source_method='reviewed_visual_transcription',
                  kind='verified_transcription', review_status='visual_verified')
        reg = EvidenceRegistry([d], [c]); result = reg.validate_answer(answer(), [c])
        self.assertTrue(result['valid'])
        citation = result['claims'][0]['citations'][0]
        self.assertEqual(citation['source_method'], 'reviewed_visual_transcription')
        self.assertEqual(citation['source_ranges'][0]['collection'], 'supplemental_blocks')

    def test_transcription_flag_cannot_relabel_ocr_as_verified(self):
        b = block(source_method='reviewed_ocr', review_status='visual_verified')
        d = document(blocks=[], supplemental_blocks=[b])
        c = chunk(b, collection='supplemental_blocks', source_method='reviewed_visual_transcription',
                  kind='verified_transcription', review_status='visual_verified')
        reg = EvidenceRegistry([d], [c]); self.assertFalse(reg.validate_answer(answer(), [c])['valid'])

    def test_alias_scope_canonicalizes_but_model_doc_id_is_canonical(self):
        alias = document('alias', sha='a' * 64, canonical_doc_id='d1', is_duplicate_alias=True)
        reg = EvidenceRegistry([self.doc, alias], [self.chunk])
        self.assertEqual(reg.resolve_doc_ids(['alias']), ['d1'])
        self.assertTrue(reg.validate_answer(answer(), [self.chunk], ['alias'])['valid'])
        self.assertFalse(reg.validate_answer(answer(did='alias'), [self.chunk], ['alias'])['valid'])

    def test_scope_escape_empty_scope_and_unknown_scope_never_widen(self):
        self.invalid(allowed=[])
        self.invalid(allowed=['missing-doc'])
        d2, c2 = document('d2'), chunk(did='d2', cid='c2')
        reg = EvidenceRegistry([self.doc, d2], [self.chunk, c2])
        self.assertFalse(reg.validate_answer(answer(did='d2', cid='c2'), [c2], ['d1'])['valid'])

    def test_claim_cannot_relabel_other_documents_quote(self):
        d2, c2 = document('d2'), chunk(did='d2', cid='c2')
        reg = EvidenceRegistry([self.doc, d2], [self.chunk, c2])
        self.assertFalse(reg.validate_answer(answer(did='d2'), [self.chunk, c2])['valid'])

    def test_schema_extra_fields_wrong_types_and_statuses_fail_closed(self):
        variants = []
        p = answer(); p['answer'] = '인용 없는 추가사실'; variants.append(p)
        p = answer(); p['claims'][0]['confidence'] = .99; variants.append(p)
        p = answer(); p['claims'][0]['citations'][0]['url'] = 'https://bad.invalid'; variants.append(p)
        p = answer(); p['status'] = 'retrieved'; variants.append(p)
        p = answer(); p['claims'][0]['category'] = 'arbitrary'; variants.append(p)
        p = answer(); p['claims'][0]['text'] = ''; variants.append(p)
        p = answer(); p['missing_information'] = 'not a list'; variants.append(p)
        for p in variants: self.invalid(p)

    def test_status_claim_and_missing_information_consistency(self):
        self.invalid(answer(status='insufficient_evidence'))
        self.invalid(answer(missing=['제출기한']))
        self.invalid(answer(status='partial'))
        p = {'status': 'answered', 'claims': [], 'missing_information': []}
        self.invalid(p)
        self.assertTrue(self.check(answer(status='partial', missing=['제출기한']))['valid'])
        p = {'status': 'insufficient_evidence', 'claims': [], 'missing_information': ['제출기한']}
        self.assertTrue(self.check(p)['valid'])

    def test_late_invalid_claim_removes_all_model_claims(self):
        p = answer(); p['claims'].append(copy.deepcopy(p['claims'][0]))
        p['claims'][1]['citations'][0]['quote'] = '조작된 근거'
        result = self.invalid(p)
        self.assertNotIn('예산은 100원', self.registry.render(result))

    def test_real_quote_does_not_establish_claim_entailment(self):
        # Deliberately wrong claim with a real quote passes provenance only.
        # This test prevents renaming citation validity as answer correctness.
        result = self.check(answer(text='예산은 999원입니다.'))
        self.assertTrue(result['valid'])
        self.assertFalse(result['semantic_entailment_verified'])
        self.assertFalse(result['answer_correctness_verified'])
        self.assertIn('별도 검토', self.registry.render(result))

    def test_render_escapes_untrusted_links_and_html(self):
        result = self.check(answer(text='<script>alert(1)</script> [링크](https://bad.invalid)'))
        rendered = self.registry.render(result)
        self.assertNotIn('<script>', rendered)
        self.assertIn('&lt;script&gt;', rendered)
        self.assertIn('\\[링크\\]', rendered)

    def test_duplicate_registry_ids_and_wrong_alias_sha_rejected(self):
        with self.assertRaises(EvidenceError): EvidenceRegistry([self.doc, self.doc], [self.chunk])
        with self.assertRaises(EvidenceError): EvidenceRegistry([self.doc], [self.chunk, self.chunk])
        with self.assertRaises(EvidenceError):
            EvidenceRegistry([self.doc, document('alias', canonical_doc_id='d1')], [self.chunk])


if __name__ == '__main__':
    unittest.main()
