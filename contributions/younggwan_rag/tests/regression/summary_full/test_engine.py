"""Runner regression tests use synthetic schemas/responses, never real API calls."""
import copy
from dataclasses import replace
import importlib
import io
import contextlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rfp_rag.summary_full.engine import FinalRun, FinalConfig, TASK, QUESTION, FIELD_SCHEMA, read_json
from rfp_rag.summary_full.transport import ShortCitationLLM
from regression.summary_full.test_catalog_audit import env as small_env


class FakeClient:
    def __init__(self):
        self.responses=self;self.requests=[];self.interrupt=False;self.fail=False;self.bad_field=False;self.bad_alias=False

    def create(self,**body):
        self.requests.append(copy.deepcopy(body))
        if self.interrupt:raise KeyboardInterrupt()
        if self.fail:raise RuntimeError('test-only connection problem')
        data=json.loads(body['input'][1]['content']);unit=next(e for e in data['evidence'] if e['citable'])
        claims=[{'doc_id':'D001','category':'budget','text':'[budget_vat|fact] 합성 테스트의 원문 연결',
            'citations':[{'chunk_id':'E99999' if self.bad_alias else unit['chunk_id'],'quote':'SOURCE'}]}]
        if self.bad_field:
            claims.append({'doc_id':'D001','category':'budget','text':'[duration|fact] REJECTED_TEXT_MUST_NOT_BE_RENDERED',
                'citations':[{'chunk_id':unit['chunk_id'],'quote':'SOURCE'}]})
        return SimpleNamespace(status='completed',output=[],output_text=json.dumps({
            'status':'answered','claims':claims,'missing_information':[]}),
            usage=SimpleNamespace(input_tokens=12,output_tokens=8,total_tokens=20,
                input_tokens_details=SimpleNamespace(cached_tokens=0)))


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.env=small_env();self.env['root']=self.root
        for name in ['rfp_rag.generation.llm','rfp_rag.generation.evidence','rfp_rag.summary_fields.fields']:
            module=importlib.import_module(name);target=self.root/(name.replace('.','/')+'.py')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(module.__file__,target)
        self.client=FakeClient()
        self.keys=patch.dict('os.environ',{'OPENAI_API_KEY':'offline-test-key'});self.keys.start();self.addCleanup(self.keys.stop)
        self.net=patch('socket.socket.connect',side_effect=AssertionError('network prohibited'));self.net.start();self.addCleanup(self.net.stop)

    def run_instance(self,config=None,run_id='test'):
        return FinalRun(self.env,run_id,config=config,client=self.client)

    def execute(self,runner,*args,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()):return runner.run(*args,**kwargs)

    def test_all_original_alias_ids_plan_and_roundtrip_without_api(self):
        runner=self.run_instance();audit=runner.audit_all_plans()
        self.assertTrue(audit['all_document_id_transport_passed'])
        self.assertEqual(audit['document_count'],2)
        self.assertEqual(runner.plan('copy-a')['doc_id'],'doc-a')
        self.assertEqual(self.client.requests,[])

    def test_disabled_run_never_generates_and_later_output_gets_new_review_revision(self):
        runner=self.run_instance()
        self.execute(runner,['doc-b'],allow_api=False)
        pending=runner.result('doc-b');old=read_json(runner.folder/'documents/doc-b/latest.json')['folder']
        self.assertEqual(pending['execution_status'],'awaiting_api')
        self.assertEqual(self.client.requests,[])
        self.execute(runner,['doc-b'],allow_api=True)
        complete=runner.result('doc-b');new=read_json(runner.folder/'documents/doc-b/latest.json')['folder']
        self.assertNotEqual(old,new)
        self.assertTrue((runner.folder/old/'comparison.csv').is_file())
        self.assertEqual(complete['execution_status'],'completed')
        self.assertFalse(complete['model_origin']['all_live_or_live_cache'])

    def test_completed_batch_is_not_called_again_and_alias_is_deduplicated(self):
        runner=self.run_instance();self.execute(runner,['doc-a','copy-a'],allow_api=True)
        calls=len(self.client.requests)
        self.execute(runner,['copy-a'],allow_api=True)
        self.assertEqual(len(self.client.requests),calls)
        result=runner.result('copy-a')
        self.assertEqual(result['checked_chunk_coverage'],1)
        self.assertEqual(result['doc_id'],'doc-a')

    def test_global_budget_persists_and_raising_budget_reuses_saved_work(self):
        runner=self.run_instance(FinalConfig(max_total_calls=1))
        r=self.execute(runner,'all',allow_api=True)
        self.assertTrue(r['budget_halted']);self.assertEqual(len(self.client.requests),1)
        self.execute(runner,'all',allow_api=True);self.assertEqual(len(self.client.requests),1)
        upgraded=self.run_instance(FinalConfig(max_total_calls=2))
        self.assertEqual(upgraded.fingerprint,runner.fingerprint)
        self.execute(upgraded,'all',allow_api=True);self.assertEqual(len(self.client.requests),2)

    def test_interrupted_api_attempt_is_reserved_and_not_automatically_retried(self):
        runner=self.run_instance();self.client.interrupt=True
        with self.assertRaises(KeyboardInterrupt):self.execute(runner,['doc-b'],allow_api=True)
        self.assertEqual(runner.budget_usage()['calls_used_or_reserved'],1)
        self.client.interrupt=False
        self.execute(runner,['doc-b'],allow_api=True)
        self.assertEqual(len(self.client.requests),1)
        self.assertEqual(runner.result('doc-b')['execution_status'],'incomplete')

    def test_interrupted_after_cache_write_recovers_without_another_request(self):
        runner=self.run_instance()
        with patch.object(runner,'_accepted_outcome',side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):self.execute(runner,['doc-b'],allow_api=True)
        self.execute(runner,['doc-b'],allow_api=True)
        self.assertEqual(len(self.client.requests),1)
        self.assertEqual(runner.result('doc-b')['execution_status'],'completed')

    def test_bad_field_claim_is_quarantined_and_valid_claims_continue(self):
        runner=self.run_instance();self.client.bad_field=True
        self.execute(runner,'all',allow_api=True)
        for did in ['doc-a','doc-b']:
            r=runner.result(did)
            self.assertEqual(r['execution_status'],'completed_with_rejections')
            self.assertEqual(len(r['claims']),1);self.assertEqual(len(r['rejected_claims']),1)
            self.assertEqual(r['checked_chunk_coverage'],1)
            self.assertEqual(next(f for f in r['fields'] if f['field_id']=='duration')['state'],'rejected_evidence')
            path=runner.folder/read_json(runner.folder/'documents'/did/'latest.json')['folder']
            self.assertNotIn('REJECTED_TEXT_MUST_NOT_BE_RENDERED',(path/'answer.md').read_text())
            self.assertIn('REJECTED_TEXT_MUST_NOT_BE_RENDERED',(path/'rejection_review.csv').read_text(encoding='utf-8-sig'))

    def test_unknown_wire_alias_is_rejected_and_other_documents_continue(self):
        runner=self.run_instance();self.client.bad_alias=True
        self.execute(runner,'all',allow_api=True)
        self.assertEqual(len(self.client.requests),2)
        for did in ['doc-a','doc-b']:
            r=runner.result(did);self.assertEqual(r['execution_status'],'incomplete')
            self.assertEqual(r['claims'],[])
        calls=len(self.client.requests)
        self.execute(runner,'all',allow_api=True);self.assertEqual(len(self.client.requests),calls)

    def test_service_failure_stops_new_calls_until_explicit_resume(self):
        runner=self.run_instance();self.client.fail=True
        self.execute(runner,'all',allow_api=True);self.assertEqual(len(self.client.requests),1)
        self.client.fail=False
        self.execute(runner,'all',allow_api=True);self.assertEqual(len(self.client.requests),1)
        self.execute(runner,'all',allow_api=True,resume_after_service_error=True)
        self.assertEqual(len(self.client.requests),2)

    def test_retry_token_is_consumed_once_and_success_is_never_regenerated(self):
        runner=self.run_instance();self.client.bad_alias=True
        self.execute(runner,['doc-b'],allow_api=True)
        key=runner.plan('doc-b')['batches'][0]['job_key']
        self.execute(runner,['doc-b'],allow_api=True,retry_jobs=[key],retry_tag='retry1')
        self.assertEqual(len(self.client.requests),2)
        self.execute(runner,['doc-b'],allow_api=True,retry_jobs=[key],retry_tag='retry1')
        self.assertEqual(len(self.client.requests),2)
        self.client.bad_alias=False
        self.execute(runner,['doc-b'],allow_api=True,retry_jobs=[key],retry_tag='retry2')
        self.assertEqual(len(self.client.requests),3)
        self.assertEqual(runner.result('doc-b')['execution_status'],'completed')

    def test_journal_result_tampering_and_stale_model_settings_fail(self):
        runner=self.run_instance();self.execute(runner,['doc-b'],allow_api=True)
        with self.assertRaises(ValueError):self.run_instance(FinalConfig(model='different-model'))
        journal=runner._journal();job=next(iter(journal['jobs'].values()))
        (runner.folder/job['outcome_file']).write_text('{}')
        with self.assertRaises(ValueError):runner.result('doc-b')

    def test_swapped_valid_batch_files_cannot_be_rendered_as_another_document(self):
        runner=self.run_instance();self.execute(runner,'all',allow_api=True)
        journal=runner._journal();jobs=list(journal['jobs'].values())
        for key in ['outcome_file','outcome_sha256']:
            jobs[0][key],jobs[1][key]=jobs[1][key],jobs[0][key]
        runner._save_journal(journal)
        with self.assertRaises(ValueError):runner.result('doc-a')

    def test_known_blocked_document_never_spends_api_on_its_prefix(self):
        runner=self.run_instance();plan=runner.plan('doc-a')
        plan['can_execute']=False;plan['block_reasons']=['test_late_chunk_block']
        result=self.execute(runner,['doc-a'],allow_api=True)
        self.assertEqual(self.client.requests,[])
        self.assertEqual(runner.result('doc-a')['execution_status'],'plan_blocked')

    def test_repeated_text_binding_uses_selected_offset_not_first_find(self):
        from rfp_rag.summary_full.transport import decode_payload
        from rfp_rag.summary_full.engine import validate_claims
        from regression.summary_full.test_catalog_audit import Index
        from rfp_rag.generation.evidence import EvidenceRegistry
        # One real registered block containing identical lines twice.
        doc=self.env['documents'][1]
        text='반복되는 예산 문장입니다.\n다른 줄\n세 번째\n네 번째\n반복되는 예산 문장입니다.\n'
        b=doc['blocks'][0];b['text']=text
        c=next(c for c in self.env['chunks'] if c['doc_id']=='doc-b')
        c.update(text=text,text_bytes=len(text.encode()));c['source_spans'][0].update(char_end=len(text),chunk_end=len(text))
        self.env['registry']=EvidenceRegistry(self.env['documents'],self.env['chunks'])
        self.env['index']=Index(self.env['documents'],self.env['chunks'])
        runner=self.run_instance();client=runner.plan_client
        d=client.canonical_request_descriptor(TASK,QUESTION,[c],schema=FIELD_SCHEMA)
        unit=next(u for u in d['mapping']['units'] if u['char_start']>0)
        raw={'status':'answered','claims':[{'doc_id':'D001','category':'budget',
            'text':'[budget_vat|fact] 합성 문장 연결','citations':[{'chunk_id':unit['alias'],'quote':'SOURCE'}]}],
            'missing_information':[]}
        payload,bindings=decode_payload(raw,d)
        accepted,rejected=validate_claims(self.env,{'payload':payload,'request_fingerprint':d['request_fingerprint'],
            'citation_transport':{**d,'source_unit_bindings':bindings}},[c])
        self.assertEqual(rejected,[])
        cite=accepted[0]['citations'][0]
        self.assertEqual(cite['quote_start'],unit['char_start'])
        self.assertEqual(cite['source_ranges'][0]['char_start'],unit['char_start'])


if __name__=='__main__':unittest.main()
