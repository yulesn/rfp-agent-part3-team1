"""Derived summaries cannot overwrite generation evidence or borrow old review."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rfp_rag.summary_content.storage import (assess_content_review, digest, load_saved_result,
    repair_saved_results, save_content_result, saved_document_ids)
from rfp_rag.summary_full.review import answer_fingerprint


def raw_result(did='doc-a',sha='a'*64):
    cite={'doc_id':did,'sha256':sha,'file_name':did+'.hwp','chunk_id':did+':c0',
          'quote':'사업예산100원','locator':'paragraph1','provenance_verified':True}
    claim={'doc_id':did,'category':'budget','text':'[budget_vat|fact] 사업예산100원',
           'citations':[cite]}
    return {'doc_id':did,'source_sha256':sha,'run_fingerprint':'test-run',
        'status':'partial','execution_status':'completed','answer_generated':True,
        'claims':[claim],'fields':[{'field_id':'budget_vat','label':'예산·VAT','state':'evidence_found',
                                  'claims':[{'text':'사업예산100원','mode':'fact','citations':[cite]}]}],
        'missing_information':[],'rejected_claims':[],'batch_outcomes':[],
        'source_chunk_ids':[did+':c0'],'input_received_chunk_ids':[did+':c0'],
        'checked_chunk_ids':[did+':c0'],'input_chunk_coverage':1.0,'checked_chunk_coverage':1.0,
        'model_origin':{'all_live_or_live_cache':True},'semantic_quality_verified':False}


def save_raw(root,raw):
    from rfp_rag.summary_full.engine import write_json
    fingerprint=answer_fingerprint(raw)
    folder=root/'documents'/raw['doc_id']/fingerprint[:20]
    write_json(folder/'result.json',raw)
    (folder/'comparison.csv').write_text('original human input\n',encoding='utf-8')
    write_json(folder.parent/'latest.json',{'folder':folder.relative_to(root).as_posix(),
        'answer_fingerprint':fingerprint})
    return folder


class ContentStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.raw=raw_result();self.raw_folder=save_raw(self.root,self.raw)

    def test_correction_is_a_separate_revision_and_keeps_original_bytes(self):
        before={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        derived=copy.deepcopy(self.raw);derived['content_audit']={'local_extraction_count':1}
        value=save_content_result(self.root,self.raw,derived)
        self.assertNotIn('content_storage',self.raw)
        self.assertNotIn('content_storage',derived)
        for path,data in before.items():self.assertEqual(path.read_bytes(),data)
        loaded,folder=load_saved_result(self.root,'doc-a',content=True)
        self.assertEqual(loaded,value)
        self.assertIn('content_documents',folder.parts)
        self.assertTrue((folder/'answer.md').is_file())

    def test_second_save_preserves_human_cells_and_creates_new_revision_for_changed_answer(self):
        first=save_content_result(self.root,self.raw,self.raw)
        _,folder=load_saved_result(self.root,'doc-a',content=True)
        review=folder/'review.csv';original=review.read_bytes()
        with review.open(encoding='utf-8-sig',newline='') as handle:
            reader=csv.DictReader(handle);rows=list(reader);columns=reader.fieldnames
        rows[0].update({'정확성':'통과','누락없음':'실패','조건보존':'통과','메모':'VAT 확인 필요'})
        with review.open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(rows)
        reviewed=review.read_bytes()
        second=save_content_result(self.root,self.raw,self.raw)
        self.assertEqual(first['content_storage'],second['content_storage'])
        self.assertNotEqual(original,reviewed);self.assertEqual(review.read_bytes(),reviewed)
        changed=copy.deepcopy(self.raw);changed['fields'][0]['claims'][0]['text']+=' 부가세 미확인'
        third=save_content_result(self.root,self.raw,changed)
        self.assertNotEqual(first['content_storage']['content_fingerprint'],third['content_storage']['content_fingerprint'])
        self.assertEqual(review.read_bytes(),reviewed)
        _,newfolder=load_saved_result(self.root,'doc-a',content=True)
        self.assertIn('review_pending',assess_content_review(newfolder,third,'tester')['status'])

    def test_blank_and_failed_review_never_passes_automatically(self):
        value=save_content_result(self.root,self.raw,self.raw)
        _,folder=load_saved_result(self.root,'doc-a',content=True)
        report=assess_content_review(folder,value,reviewer='named reviewer')
        self.assertFalse(report['case_accepted'])
        self.assertFalse(report['human_review_completed'])
        self.assertFalse(report['semantic_quality_automatically_verified'])

    def test_tampering_with_raw_or_corrected_result_is_rejected(self):
        raw_path=self.raw_folder/'result.json';value=json.loads(raw_path.read_text())
        value['claims'][0]['text']='[budget_vat|fact] changed'
        raw_path.write_text(json.dumps(value),encoding='utf-8')
        with self.assertRaises(ValueError):load_saved_result(self.root,'doc-a')
        save_content_result(self.root,self.raw,self.raw)
        _,folder=load_saved_result(self.root,'doc-a',content=True)
        path=folder/'result.json';value=json.loads(path.read_text());value['claims'][0]['text']='changed'
        path.write_text(json.dumps(value),encoding='utf-8')
        with self.assertRaises(ValueError):load_saved_result(self.root,'doc-a',content=True)

    def test_latest_cannot_escape_document_folder_or_swap_another_doc(self):
        pointer=self.raw_folder.parent/'latest.json'
        pointer.write_text(json.dumps({'folder':'../elsewhere','answer_fingerprint':answer_fingerprint(self.raw)}))
        with self.assertRaises(ValueError):load_saved_result(self.root,'doc-a')
        pointer.write_text(json.dumps({'folder':str(self.raw_folder),'answer_fingerprint':answer_fingerprint(self.raw)}))
        with self.assertRaises(ValueError):load_saved_result(self.root,'doc-a')
        for did in ['../doc-a','a/b','a\\b','..']:
            with self.subTest(did=did),self.assertRaises(ValueError):load_saved_result(self.root,did)

    def test_readonly_review_values_and_stale_revisions_are_rejected(self):
        value=save_content_result(self.root,self.raw,self.raw)
        _,folder=load_saved_result(self.root,'doc-a',content=True)
        path=folder/'review.csv'
        path.write_text(path.read_text(encoding='utf-8-sig').replace('사업예산100원','invented'),encoding='utf-8-sig')
        report=assess_content_review(folder,value,'tester')
        self.assertEqual(report['status'],'invalid_review')
        self.assertFalse(report['case_accepted'])

    def test_source_identity_cannot_change_during_save(self):
        derived=copy.deepcopy(self.raw);derived['source_sha256']='b'*64
        with self.assertRaises(ValueError):save_content_result(self.root,self.raw,derived)

    def test_existing_only_selection_does_not_schedule_any_document(self):
        self.assertEqual(saved_document_ids(self.root),['doc-a'])
        empty=self.root/'empty';empty.mkdir()
        self.assertEqual(saved_document_ids(empty),[])

    def test_repair_continues_past_corrupt_saved_document_and_calls_no_service(self):
        raw_b=raw_result('doc-b','b'*64);bad_folder=save_raw(self.root,raw_b)
        path=bad_folder/'result.json';payload=json.loads(path.read_text())
        payload['claims'][0]['text']='changed without matching pointer'
        path.write_text(json.dumps(payload),encoding='utf-8')
        env={'documents':[{'doc_id':'doc-a','sha256':'a'*64},{'doc_id':'doc-b','sha256':'b'*64}],
             'registry':SimpleNamespace(resolve_doc_ids=lambda ids:ids)}
        original={p:p.read_bytes() for p in (self.root/'documents').rglob('*') if p.is_file()}
        with patch('rfp_rag.summary_content.storage.get_validated_document_chunks',return_value=[]),\
             patch('rfp_rag.summary_content.pipeline.rebuild_document_summary',side_effect=lambda r,*args:copy.deepcopy(r)) as rebuild,\
             patch('socket.socket.connect',side_effect=AssertionError('network prohibited')):
            report=repair_saved_results(env,self.root)
        self.assertEqual(report['selected_saved_documents'],2)
        self.assertEqual(report['repaired_documents'],1)
        self.assertEqual(report['failed_documents'],1)
        self.assertEqual(rebuild.call_count,1)
        self.assertEqual(report['api_calls'],0)
        self.assertFalse(report['generation_reexecuted'])
        self.assertFalse(report['human_review_completed'])
        for path,expected in original.items():self.assertEqual(path.read_bytes(),expected)

    def test_changed_source_version_is_rejected_before_rebuild(self):
        env={'documents':[{'doc_id':'doc-a','sha256':'b'*64}],
             'registry':SimpleNamespace(resolve_doc_ids=lambda ids:ids)}
        with patch('rfp_rag.summary_content.pipeline.rebuild_document_summary') as rebuild:
            report=repair_saved_results(env,self.root)
        self.assertEqual(report['failed_documents'],1)
        rebuild.assert_not_called()

    def test_frozen_run_binding_is_checked_for_raw_results(self):
        (self.root/'frozen.json').write_text(json.dumps({'other':'run'}),encoding='utf-8')
        with self.assertRaises(ValueError):load_saved_result(self.root,'doc-a')

    def test_saved_symlink_does_not_escape_run(self):
        with tempfile.TemporaryDirectory() as outside:
            out=Path(outside);(out/'latest.json').write_text('{}')
            (self.root/'documents'/'escape').symlink_to(out,target_is_directory=True)
            with self.assertRaises(ValueError):load_saved_result(self.root,'escape')


if __name__=='__main__':unittest.main()
