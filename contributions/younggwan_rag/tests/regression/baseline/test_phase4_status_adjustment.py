"""Scoped comparison status handling preserves content, guards and budgets."""
from copy import deepcopy
from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from rfp_rag.generation.llm import LLMConfig,LLMError,OpenAIGroundedLLM,COMPARISON_DOCUMENT_TASK
from rfp_rag.generation.evidence import EvidenceRegistry
from rfp_rag.generation.rag import AnswerSession
from regression.baseline.test_phase4_llm import FakeClient, PAYLOAD, SHA, evidence
from regression.baseline.test_phase4_rag import fixture, Index


class ConservativeStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.key=patch.dict(os.environ,{'OPENAI_API_KEY':'offline-status-test-only'})
        self.key.start();self.addCleanup(self.key.stop)
        self.net=patch('socket.socket.connect',side_effect=AssertionError('network prohibited'))
        self.net.start();self.addCleanup(self.net.stop)

    def make(self,payload=None,**options):
        if payload is None:
            payload=deepcopy(PAYLOAD);payload['missing_information']=['안정화 조건']
        fake=FakeClient(payload=payload,**options)
        llm=OpenAIGroundedLLM(Path(self.tmp.name)/str(len(list(Path(self.tmp.name).iterdir()))),
            LLMConfig(enabled=True),{'A':SHA},fake)
        return llm,fake

    def generate(self,llm,task=COMPARISON_DOCUMENT_TASK):
        return llm.generate(task,'현재 문서의 사업기간과 조건은?',evidence())

    def test_answered_with_missing_is_only_downgraded_and_content_unchanged(self):
        llm,fake=self.make();original=deepcopy(fake.payload);r=self.generate(llm)
        self.assertEqual(r['payload']['status'],'partial')
        self.assertEqual(r['payload']['claims'],original['claims'])
        self.assertEqual(r['payload']['missing_information'],original['missing_information'])
        self.assertEqual(fake.payload,original)
        self.assertEqual(r['response_status_adjustment']['original_status'],'answered')
        self.assertFalse(r['response_status_adjustment']['claims_changed'])
        self.assertEqual(len(fake.calls),1);self.assertEqual(llm.usage_report()['failed_attempts'],0)

    def test_normalized_response_caches_and_replays_without_extra_calls(self):
        llm,fake=self.make();first=self.generate(llm);second=self.generate(llm)
        self.assertTrue(second['cache_hit']);self.assertEqual(len(fake.calls),1)
        self.assertEqual(second['payload'],first['payload'])
        self.assertEqual(second['response_status_adjustment'],first['response_status_adjustment'])
        self.assertEqual(second['usage']['api_calls'],0)

    def test_regular_qa_and_summary_contracts_remain_strict(self):
        for task in ['qa','comparison','summary_map']:
            with self.subTest(task=task):
                llm,_=self.make()
                with self.assertRaises(LLMError):self.generate(llm,task)

    def test_no_missing_information_is_ever_deleted_or_promoted_to_answered(self):
        payload=deepcopy(PAYLOAD);payload.update(status='partial',missing_information=['VAT'])
        llm,_=self.make(payload);r=self.generate(llm)
        self.assertEqual(r['payload'],payload)
        self.assertNotIn('response_status_adjustment',r)

    def test_no_repair_for_empty_claims_partial_without_missing_or_bad_shape(self):
        cases=[]
        p=deepcopy(PAYLOAD);p.update(claims=[],missing_information=['기간']);cases.append(p)
        p=deepcopy(PAYLOAD);p['status']='partial';cases.append(p)
        p=deepcopy(PAYLOAD);p['missing_information']=[''];cases.append(p)
        p=deepcopy(PAYLOAD);p.update(missing_information=['기간'],extra='untrusted');cases.append(p)
        p=deepcopy(PAYLOAD);p['claims'][0]['citations']=[];p['missing_information']=['기간'];cases.append(p)
        for p in cases:
            with self.subTest(payload=p):
                llm,_=self.make(p)
                with self.assertRaises(LLMError):self.generate(llm)
                self.assertFalse(llm._cache_path().is_file())

    def test_refusal_incomplete_and_malformed_json_remain_blocked(self):
        for options in [{'status':'incomplete'}, {'raw':'not-json'},
                        {'output':[{'content':[{'type':'refusal','refusal':'refused'}]}]}]:
            llm,_=self.make(**options)
            with self.assertRaises(LLMError):self.generate(llm)
            self.assertFalse(llm._cache_path().is_file())

    def test_new_task_has_a_separate_cache_key_and_regular_requests_stay_exact(self):
        llm,_=self.make();a=llm._prepare('qa','기간?',evidence());b=llm._prepare(COMPARISON_DOCUMENT_TASK,'기간?',evidence())
        self.assertNotEqual(a['fingerprint'],b['fingerprint'])

    def integration(self,*,bad_quote=False):
        docs,chunks=fixture();allowed={d['doc_id']:d['sha256'] for d in docs}
        class PerDocument(FakeClient):
            def create(inner,**body):
                inner.calls.append(body);c=json.loads(body['input'][1]['content'])['evidence'][0]
                p={'status':'answered','claims':[{'doc_id':c['doc_id'],'category':'budget','text':c['text'],
                    'citations':[{'chunk_id':c['chunk_id'],'quote':'INVENTED QUOTE' if bad_quote and c['doc_id']=='B' else c['text']}]}],
                    'missing_information':['현재 문서의 안정화 조건'] if c['doc_id']=='B' else []}
                return {'status':'completed','output_text':json.dumps(p),'output':[],
                    'usage':{'input_tokens':20,'output_tokens':10,'total_tokens':30}}
        fake=PerDocument();llm=OpenAIGroundedLLM(Path(self.tmp.name)/'integration',LLMConfig(enabled=True),allowed,fake)
        session=AnswerSession(Index(docs,chunks),llm,EvidenceRegistry(docs,chunks),active_doc_ids=['A','B'])
        return session.ask('두 사업의 예산과 안정화 조건 비교',feature='comparison'),llm,fake

    def test_second_document_pattern_does_not_abort_comparison(self):
        r,llm,fake=self.integration()
        self.assertEqual(r['execution_status'],'completed');self.assertEqual(r['status'],'partial')
        self.assertTrue(r['validation']['valid']);self.assertEqual(len(fake.calls),2)
        self.assertEqual(llm.usage_report()['failed_attempts'],0)
        self.assertEqual(r['document_validation'][1]['response_status_adjustment']['effective_status'],'partial')
        self.assertTrue(r['missing_information'])

    def test_status_downgrade_does_not_bypass_exact_source_validation(self):
        r,_,_=self.integration(bad_quote=True)
        self.assertEqual(r['execution_status'],'validation_failed');self.assertEqual(r['claims'],[])
        self.assertNotIn('INVENTED QUOTE',r['markdown'])
        self.assertEqual(r['document_validation'][1]['response_status_adjustment']['effective_status'],'partial')


if __name__=='__main__':unittest.main()
