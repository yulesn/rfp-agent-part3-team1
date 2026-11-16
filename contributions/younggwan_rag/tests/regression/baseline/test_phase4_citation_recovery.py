"""Formatting recovery must still end in exact source validation, without API."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from regression.baseline.test_phase4_evidence import document, block, chunk, answer
from rfp_rag.generation.evidence import EvidenceRegistry
from rfp_rag.generation.citation_recovery import validate_with_quote_recovery, replay_saved_result
from rfp_rag.generation.llm import LLMConfig, OpenAIGroundedLLM


class RecoveryTests(unittest.TestCase):
    def fixture(self, text='사업예산 :  100원.\n사업기간 : 5개월.'):
        b=block(text=text); d=document(source=b); c=chunk(b)
        return d,c,EvidenceRegistry([d],[c])

    def test_original_validator_remains_strict(self):
        d,c,reg=self.fixture()
        payload=answer(quote='사업예산:100원.')
        self.assertFalse(reg.validate_answer(payload,[c])['valid'])

    def test_unique_whitespace_difference_is_replaced_by_exact_original(self):
        d,c,reg=self.fixture()
        payload=answer(quote='사업예산:100원.')
        original=copy.deepcopy(payload)
        r=validate_with_quote_recovery(reg,payload,[c],['d1'])
        self.assertTrue(r['valid']); self.assertEqual(payload,original)
        quote=r['claims'][0]['citations'][0]
        self.assertEqual(quote['quote'],'사업예산 :  100원.')
        span=quote['source_ranges'][0]
        self.assertEqual(d['blocks'][0]['text'][span['char_start']:span['char_end']],quote['quote'])
        self.assertEqual(r['citation_recovery']['corrected_citations'],1)
        self.assertEqual(r['citation_recovery']['additional_api_calls'],0)
        self.assertFalse(r['semantic_entailment_verified'])

    def test_line_wrap_tab_and_unicode_whitespace_are_source_restored(self):
        d,c,reg=self.fixture('사업예산\n:\t100원.\u00a0부가세 포함')
        r=validate_with_quote_recovery(reg,answer(quote='사업예산:100원. 부가세포함'),[c])
        self.assertTrue(r['valid'])
        self.assertEqual(r['claims'][0]['citations'][0]['quote'],c['text'])

    def test_digits_currency_vat_punctuation_negation_are_never_repaired(self):
        d,c,reg=self.fixture('사업예산 : 100원(부가세 포함), 면제 아님.')
        for quote in ['사업예산:999원(부가세포함)','사업예산:100달러(부가세포함)',
                      '사업예산:100원(부가세별도)','사업예산:100원[부가세포함]', '면제임.']:
            with self.subTest(quote=quote):
                r=validate_with_quote_recovery(reg,answer(quote=quote),[c])
                self.assertFalse(r['valid']); self.assertEqual(r['claims'],[])

    def test_ambiguous_nonexact_recovery_is_rejected(self):
        d,c,reg=self.fixture('예산 : 100원 / 예산: 100원')
        r=validate_with_quote_recovery(reg,answer(quote='예산:100원'),[c])
        self.assertFalse(r['valid'])
        self.assertEqual(r['citation_diagnostics'][0]['outcome'],'ambiguous_whitespace_match')

    def test_other_chunks_and_documents_are_not_searched(self):
        d,c,reg=self.fixture()
        second=block('b2',text='예산 : 999원')
        d['blocks'].append(second)
        c2=chunk(second,cid='c2')
        reg=EvidenceRegistry([d],[c,c2])
        self.assertEqual(len(reg.validate_chunks([c,c2])),2)
        r=validate_with_quote_recovery(reg,answer(quote='예산:999원'),[c,c2])
        self.assertFalse(r['valid'])
        r=validate_with_quote_recovery(reg,answer(quote='사업예산:100원.',cid='not-provided'),[c])
        self.assertFalse(r['valid'])

    def test_scope_escape_wrong_doc_and_tampered_source_remain_rejected(self):
        d,c,reg=self.fixture(); p=answer(quote='사업예산:100원.')
        self.assertFalse(validate_with_quote_recovery(reg,p,[c],[])['valid'])
        self.assertFalse(validate_with_quote_recovery(reg,answer(quote=p['claims'][0]['citations'][0]['quote'],did='other'),[c])['valid'])
        c['text']='가짜 원문'; c['text_bytes']=len(c['text'].encode())
        self.assertFalse(validate_with_quote_recovery(reg,p,[c])['valid'])

    def test_one_unrepairable_claim_blocks_other_recovered_claims(self):
        d,c,reg=self.fixture()
        p=answer(quote='사업예산:100원.')
        p['claims']+=answer(quote='기간99개월')['claims']
        r=validate_with_quote_recovery(reg,p,[c])
        self.assertFalse(r['valid']);self.assertEqual(r['claims'],[])
        self.assertEqual(r['citation_recovery']['corrected_citations'],1)
        self.assertEqual(len(r['citation_diagnostics']),2)

    def test_exact_quotes_are_left_unchanged(self):
        d,c,reg=self.fixture()
        r=validate_with_quote_recovery(reg,answer(quote='사업예산 :  100원.'),[c])
        self.assertTrue(r['valid']);self.assertEqual(r['citation_recovery']['corrected_citations'],0)

    def test_schema_errors_still_fail(self):
        d,c,reg=self.fixture()
        for p in [None,[],{},dict(answer(),unexpected='field')]:
            self.assertFalse(validate_with_quote_recovery(reg,p,[c])['valid'])


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        b=block(text='사업예산 :  100원.')
        self.d,self.c=document(source=b),chunk(b)
        self.reg=EvidenceRegistry([self.d],[self.c])
        self.payload=answer(quote='사업예산:100원.')
        class Responses:
            def __init__(inner):inner.calls=0
            def create(inner,**kwargs):
                inner.calls+=1
                return SimpleNamespace(status='completed',output=[],output_text=json.dumps(self.payload),
                    usage=SimpleNamespace(input_tokens=10,output_tokens=20,total_tokens=30))
        self.responses=Responses()
        with patch.dict(os.environ,{'OPENAI_API_KEY':'offline-test-only'}):
            client=OpenAIGroundedLLM(self.root/'phase4_outputs/generation_cache',
                config=LLMConfig(enabled=True),allowed_sources={'d1':'a'*64},client=SimpleNamespace(responses=self.responses))
            generated=client.generate('qa','예산은?',[self.c])
        self.saved={'status':'not_generated','execution_status':'validation_failed','claims':[],
                    'source_chunks':[self.c],'active_doc_ids':['d1'],
                    'retrieval':{'comparison':False,'standalone_query':'예산은?'},
                    'generation':{k:v for k,v in generated.items() if k!='payload'}}
        self.path=self.root/'result_003.json';self.path.write_text(json.dumps(self.saved))
        self.cache=self.root/'phase4_outputs/generation_cache/grounded_responses.sqlite3'

    def test_replay_works_without_api_key_and_does_not_call_sdk(self):
        before=self.path.read_bytes()
        with patch.dict(os.environ,{},clear=True):
            r=replay_saved_result(self.root,self.path,registry=self.reg)
        self.assertEqual(r['status'],'answered');self.assertTrue(r['validation']['valid'])
        self.assertEqual(r['replay']['additional_api_calls'],0)
        self.assertEqual(r['generation']['usage']['api_calls'],0)
        self.assertEqual(r['generation']['original_usage']['api_calls'],1)
        self.assertEqual(self.responses.calls,1)
        self.assertEqual(self.path.read_bytes(),before)
        self.assertTrue(Path(r['revalidated_result_path']).is_file())

    def test_tampered_checksum_is_rejected_without_network(self):
        with sqlite3.connect(self.cache) as db:db.execute("UPDATE responses SET checksum='bad'")
        with self.assertRaisesRegex(ValueError,'무결성'):
            replay_saved_result(self.root,self.path,registry=self.reg,persist=False)

    def test_changed_saved_context_and_query_are_rejected(self):
        for key in ['query','text','doc']:
            s=copy.deepcopy(self.saved)
            if key=='query':s['retrieval']['standalone_query']='다른 질문'
            elif key=='text':s['source_chunks'][0]['text']='다른 원문'
            else:s['source_chunks'][0]['doc_id']='other'
            self.path.write_text(json.dumps(s))
            with self.assertRaises(ValueError):replay_saved_result(self.root,self.path,registry=self.reg,persist=False)

    def test_missing_cache_does_not_fallback_to_api(self):
        self.cache.unlink()
        with self.assertRaises(FileNotFoundError):replay_saved_result(self.root,self.path,registry=self.reg)
        self.assertEqual(self.responses.calls,1)

    def test_repeated_replay_keeps_original_result_and_cache_bytes(self):
        before=self.cache.read_bytes()
        a=replay_saved_result(self.root,self.path,registry=self.reg)
        b=replay_saved_result(self.root,self.path,registry=self.reg)
        self.assertNotEqual(a['revalidated_result_path'],b['revalidated_result_path'])
        self.assertEqual(self.cache.read_bytes(),before)


if __name__=='__main__':unittest.main()
