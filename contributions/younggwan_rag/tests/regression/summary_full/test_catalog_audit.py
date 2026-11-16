"""Document selection and complete corpus identity checks; no model service."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from rfp_rag.summary_full.catalog_audit import (CatalogError, _chunks_fingerprint, audit_catalog,
                           build_catalog, get_validated_document_chunks,
                           select_documents)
from rfp_rag.generation.evidence import EvidenceRegistry


def corpus():
    documents, chunks = [], []
    for did, marker, n in [('doc-a', 'a', 2), ('doc-b', 'b', 1)]:
        blocks = []
        for i in range(n):
            text = f'{did} 원문 {i}: 사업비와 계약기간을 확인합니다.'
            block = {'block_id': f'b{i}', 'text': text, 'locator': f'PDF p.{i+1}', 'page': i+1}
            blocks.append(block)
            chunks.append({'chunk_id': f'{did}:c{i}', 'doc_id': did, 'canonical_doc_id': did,
                'sha256': marker*64, 'file_name': did+'.pdf', 'text': text,
                'text_bytes': len(text.encode()), 'length_unit': 'utf8_bytes', 'kind': 'prose',
                'source_method': 'raw_parser', 'locator': block['locator'], 'page': i+1,
                'source_spans': [{'collection': 'blocks', 'block_id': block['block_id'],
                    'char_start': 0, 'char_end': len(text), 'chunk_start': 0, 'chunk_end': len(text),
                    'page': i+1, 'locator': block['locator']}]})
        documents.append({'doc_id': did, 'canonical_doc_id': did, 'sha256': marker*64,
            'file_name': did+'.pdf', 'blocks': blocks, 'supplemental_blocks': [], 'is_duplicate_alias': False})
    documents.append({'doc_id': 'copy-a', 'canonical_doc_id': 'doc-a', 'sha256': 'a'*64,
        'file_name': 'copy-a.pdf', 'blocks': [], 'supplemental_blocks': [], 'is_duplicate_alias': True})
    return documents, chunks


class Index:
    def __init__(self, documents, chunks):
        self.chunks = copy.deepcopy(chunks)
        self.aliases = {d['doc_id']: d['canonical_doc_id'] for d in documents}
        self.catalog = [{'doc_id': d['doc_id'], 'canonical_doc_id': d['doc_id'],
            'file_name': d['file_name'], 'sha256': d['sha256'],
            'alias_doc_ids': [k for k, v in self.aliases.items() if v == d['doc_id']]}
            for d in documents if d['doc_id'] == d['canonical_doc_id']]
        self.metadata = {'canonical_document_count': len(self.catalog), 'chunk_count': len(chunks),
            'chunks_fingerprint': _chunks_fingerprint(chunks), 'index_fingerprint': 'test-only',
            'alias_map': self.aliases}
        self.reverse = False

    def get_metadata(self): return copy.deepcopy(self.metadata)
    def get_catalog(self): return copy.deepcopy(self.catalog)

    def resolve_doc_ids(self, values):
        if any(k not in self.aliases for k in values): raise ValueError('unknown')
        return list(dict.fromkeys(self.aliases[k] for k in values))

    def get_document_chunks(self, value):
        did = self.resolve_doc_ids([value])[0]
        chunks = [copy.deepcopy(c) for c in self.chunks if c['doc_id'] == did]
        return list(reversed(chunks)) if self.reverse else chunks


def env():
    documents, chunks = corpus()
    return {'documents': documents, 'chunks': chunks, 'index': Index(documents, chunks),
            'registry': EvidenceRegistry(documents, chunks)}


class CatalogAuditTests(unittest.TestCase):
    def test_every_original_and_alias_are_listed(self):
        value = env(); rows = build_catalog(value)
        self.assertEqual(len(rows), 3)
        row = next(r for r in rows if r['doc_id'] == 'copy-a')
        self.assertTrue(row['is_alias'])
        self.assertEqual(row['canonical_doc_id'], 'doc-a')
        self.assertEqual(row['chunk_count'], 2)

    def test_all_selection_deduplicates_aliases_but_preserves_canonical_order(self):
        value = env()
        self.assertEqual(select_documents(value, 'all'), ['doc-a', 'doc-b'])
        self.assertEqual(select_documents(value, ['doc-b', 'copy-a', 'doc-a']), ['doc-b', 'doc-a'])
        self.assertEqual(select_documents(value, 'copy-a'), ['doc-a'])

    def test_unknown_empty_or_ill_typed_selection_does_not_select_all(self):
        value = env()
        for selected in ['missing', None, [], {}, {'doc-a'}, [None], ['doc-a', 'missing'], 1]:
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                select_documents(value, selected)

    def test_alias_retrieval_gets_all_the_same_registered_chunks(self):
        value = env()
        self.assertEqual(get_validated_document_chunks(value, 'copy-a'),
                         get_validated_document_chunks(value, 'doc-a'))
        self.assertEqual(len(get_validated_document_chunks(value, 'doc-a')), 2)

    def test_full_audit_is_zero_network_and_never_semantic_quality(self):
        value = env(); before = copy.deepcopy(value['documents'])
        with patch('socket.socket', side_effect=AssertionError('network prohibited')):
            report = audit_catalog(value)
        self.assertTrue(report['passed'], report['errors'])
        self.assertEqual((report['original_document_count'], report['canonical_document_count'], report['chunk_count']), (3, 2, 3))
        self.assertEqual(report['api_calls'], 0)
        self.assertFalse(report['semantic_quality_verified'])
        self.assertEqual(before, value['documents'])

    def test_changed_catalog_sha_and_missing_alias_map_are_failures(self):
        for mode in ['sha', 'aliases', 'catalog_aliases']:
            value = env()
            if mode == 'sha': value['index'].catalog[0]['sha256'] = 'c'*64
            elif mode == 'aliases': value['index'].metadata['alias_map'] = {'doc-a':'doc-a'}
            else: value['index'].catalog[0]['alias_doc_ids'] = ['doc-a']
            with self.subTest(mode=mode):
                self.assertFalse(audit_catalog(value)['passed'])
                with self.assertRaises(CatalogError): build_catalog(value)

    def test_missing_or_reordered_chunks_cannot_report_full_document(self):
        for mode in ['missing', 'reverse', 'duplicate']:
            value = env()
            if mode == 'missing': value['index'].chunks.pop(0)
            elif mode == 'reverse': value['index'].reverse = True
            else: value['index'].chunks.insert(0, copy.deepcopy(value['index'].chunks[0]))
            with self.subTest(mode=mode):
                self.assertFalse(audit_catalog(value)['passed'])
                with self.assertRaises(ValueError): get_validated_document_chunks(value, 'doc-a')

    def test_changed_index_source_text_is_rejected(self):
        value = env(); value['index'].chunks[0]['text'] = '조작된 본문'
        self.assertFalse(audit_catalog(value)['passed'])

    def test_duplicate_canonical_same_sha_is_not_double_counted_as_valid(self):
        value = env()
        value['index'].catalog[1]['sha256'] = value['index'].catalog[0]['sha256']
        self.assertFalse(audit_catalog(value)['passed'])

    def test_saved_content_fingerprint_mismatch_is_failure(self):
        value = env(); value['index'].metadata['chunks_fingerprint'] = '0'*64
        report = audit_catalog(value)
        self.assertFalse(report['passed'])
        self.assertIn('saved_chunks_fingerprint_matches_index', [e['code'] for e in report['errors']])


if __name__ == '__main__': unittest.main()
