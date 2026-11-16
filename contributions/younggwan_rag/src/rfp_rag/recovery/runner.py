"""Audit and explicitly retry failed jobs while preserving the original run."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re

from rfp_rag.generation.llm import LLMError
from rfp_rag.summary_fields.fields import FIELD_BY_ID, parse_field_claim
from rfp_rag.summary_full.engine import FinalRun, FinalConfig, read_json, write_json, file_hash
from .client import RecoveryLLM, REVISION


FAILED_STATES=frozenset({'validation_failed','cache_invalid','api_failed','interrupted'})


def rejection_reason(claim):
    if not isinstance(claim,dict) or not isinstance(claim.get('text'),str):return 'invalid_claim_shape'
    marker=re.match(r'^\[([a-z_]+)\|([^\]]+)\]\s*(.*)',claim['text'],re.S)
    if marker is None:return 'invalid_field_marker'
    fid,mode,text=marker.groups()
    if fid not in FIELD_BY_ID:return 'unknown_field'
    if mode not in {'fact','notice_reference','conflict_candidate'}:return 'unknown_field_mode'
    if not text.strip():return 'empty_claim_text'
    if claim.get('category')!=FIELD_BY_ID[fid]['category']:return 'field_category_mismatch'
    try:parse_field_claim(claim)
    except (ValueError,TypeError,KeyError):return 'invalid_field_contract'
    return 'source_or_binding_validation_failed'


class RecoveryRun(FinalRun):
    """Use the unchanged frozen implementation for all acceptance and accounting."""

    def _client(self,allowed,cache_dir=None):
        client=RecoveryLLM(cache_dir or self.root/'phase5_outputs/source_unit_cache',
            config=self.config.llm_config(),allowed_sources=allowed,client=self.client_injection)
        client._allowed_request_fingerprints=getattr(self,'_retry_fingerprints',None)
        return client

    def _failure(self,b,status,error):
        value=super()._failure(b,status,error)
        extra=getattr(error,'recovery_diagnostic',None)
        if isinstance(error,LLMError) and isinstance(extra,dict):
            value['recovery_diagnostic']=dict(extra)
        return value

    def audit(self):
        self.assert_frozen()
        journal=self._journal();completed=[];failed=[];rejected=[]
        for key,job in journal['jobs'].items():
            if job.get('status') in {'completed','completed_with_rejections'}:completed.append(key)
            elif job.get('status') in FAILED_STATES:
                outcome=self._load_outcome(job)
                failed.append({'job_key':key,'status':job['status'],
                    'request_fingerprint':job['request_fingerprint'],
                    'error_diagnostic':(outcome or {}).get('error_diagnostic'),
                    'recovery_diagnostic':(outcome or {}).get('recovery_diagnostic'),
                    'attempt_count':len(job.get('attempts',[]))})
            outcome=self._load_outcome(job)
            for row in (outcome or {}).get('rejected_claims',[]):
                raw=row.get('raw_claim',{})
                rejected.append({'job_key':key,'rejection_id':row.get('rejection_id'),
                    'field_id':row.get('field_id'),'reason':rejection_reason(raw),
                    'category':raw.get('category') if isinstance(raw,dict) else None,
                    'requires_human_review':True,'automatically_accepted':False,
                    'classification_is_first_failure_only':True,'source_validation_rechecked':False})
        counts={}
        for row in rejected:counts[row['reason']]=counts.get(row['reason'],0)+1
        policy={p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
        report={'revision':REVISION,'run_fingerprint':self.fingerprint,
            'recovery_code_sha256':policy,'completed_jobs':len(completed),'failed_jobs':failed,
            'rejected_claim_count':len(rejected),'rejection_reason_counts':counts,'rejected_claims':rejected,
            'budget':self.budget_usage(),'api_calls_by_audit':0,
            'past_failed_response_reconstructed':False,'past_failure_root_cause_confirmed':False,
            'semantic_quality_verified':False,'human_reviews_modified':False}
        write_json(self.folder/'recovery_audit.json',report)
        return report

    def retry_failed(self,job_keys=None,*,retry_tag='phase5_recovery_v1',max_new_calls=1,allow_api=False,
                     resume_after_service_error=False):
        if type(max_new_calls)is not int or max_new_calls<1:raise ValueError('추가 호출 상한은 양의 정수여야 합니다.')
        if type(allow_api)is not bool:raise TypeError('allow_api는 bool이어야 합니다.')
        if type(resume_after_service_error)is not bool:raise TypeError('서비스 재개 옵션은 bool이어야 합니다.')
        self.assert_frozen();journal=self._journal()
        keys=list(dict.fromkeys(job_keys if job_keys is not None else [
            k for k,j in journal['jobs'].items() if j.get('status') in FAILED_STATES]))
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',retry_tag):raise ValueError('retry_tag는 영문·숫자·밑줄·하이픈으로 지정하세요.')
        ready=[];already_completed=[];tag_already_used=[]
        for key in keys:
            job=journal['jobs'].get(key)
            if job is None:raise ValueError('기존 실행에 없는 작업 키입니다: '+key)
            if job['status'] in {'completed','completed_with_rejections'}:already_completed.append(key);continue
            if job['status'] not in FAILED_STATES:raise ValueError('명시적인 실패 상태가 아닌 작업은 재시도하지 않습니다: '+key)
            if any(a.get('retry_tag')==retry_tag for a in job.get('attempts',[])):
                tag_already_used.append(key);continue
            ready.append(key)
        if len(ready)>max_new_calls:raise ValueError('선택한 실패 작업 수가 추가 호출 상한을 넘습니다. --job으로 범위를 줄이세요.')
        selected=[];fingerprints=set()
        for did in dict.fromkeys(k.split(':',1)[0] for k in ready):
            plan=self.plan(did)
            if not plan['can_execute']:raise ValueError('해당 문서 입력 계획이 실행 불가 상태입니다.')
            actual={b['job_key']:b for b in plan['batches']}
            for key in [k for k in ready if k.startswith(did+':')]:
                if key not in actual:raise ValueError('작업 키가 현재 원문·요청 계획과 다릅니다.')
                self._load_outcome(journal['jobs'][key],actual[key])
                fingerprints.add(actual[key]['request_fingerprint'])
            # Parent runner cannot generate any unrelated pending batch.
            for b in plan['batches']:
                state=journal['jobs'].get(b['job_key'],{}).get('status')
                if state not in {'completed','completed_with_rejections'}|FAILED_STATES:
                    raise ValueError('선택 문서에 미실행·중단 중 배치가 있어 복구 범위를 보장할 수 없습니다.')
            selected.append(did)
        before=self.budget_usage()
        if before['calls_used_or_reserved']+len(ready)>self.config.max_total_calls:
            raise ValueError('기존 실행의 전체 호출 상한을 초과합니다.')
        summary={'selected_failed_jobs':ready,'already_completed':already_completed,
            'retry_tag_already_used':tag_already_used,'max_new_calls':max_new_calls,
            'retry_tag':retry_tag,'allow_api':allow_api,'budget_before':before}
        if not ready or not allow_api:
            summary.update(api_calls_this_recovery=0,retry_executed=False)
            return summary
        self._retry_fingerprints=fingerprints
        old_config=self.config
        self.config=replace(self.config,max_total_calls=min(self.config.max_total_calls,
            before['calls_used_or_reserved']+max_new_calls))
        try:
            execution=super().run(selected,allow_api=True,retry_jobs=ready,retry_tag=retry_tag,
                resume_after_service_error=resume_after_service_error)
        finally:
            self.config=old_config
            self._retry_fingerprints=None
        after=self.budget_usage()
        summary.update(execution=execution,budget_after=after,retry_executed=True,
            api_calls_this_recovery=after['calls_used_or_reserved']-before['calls_used_or_reserved'])
        write_json(self.folder/'recovery_last_attempt.json',summary)
        return summary


def open_existing_run(env,run_id='phase5_final_source_units',*,client=None):
    folder=Path(env['root'])/'phase5_outputs/final_source_units'/run_id
    frozen_path=folder/'frozen.json'
    if not frozen_path.is_file():raise FileNotFoundError('기존 frozen.json이 없습니다. 새 유료 실행은 시작하지 않습니다.')
    frozen=read_json(frozen_path)
    report_path=folder/'final_report.json'
    budget=read_json(report_path).get('budget',{}) if report_path.exists() else {}
    config=FinalConfig(**frozen['model_settings'],max_total_calls=budget.get('max_total_calls',100),
        max_total_input_bytes=budget.get('max_total_input_bytes',6000000))
    refs=read_json(Path(env['root'])/'rfp_phase5_final/references.json')
    return RecoveryRun(env,run_id,config,references=refs,client=client)


def load_api_key(root,env_path=None):
    root=Path(root).resolve()
    if env_path is not None:
        path=Path(env_path).expanduser()
        if not path.is_absolute():path=root/path
        if not path.is_file():raise ValueError('지정한 .env 파일을 찾지 못했습니다.')
    else:path=next((p for p in (root/'.env',root.parent/'.env') if p.is_file()),None)
    if path is not None:
        try:from dotenv import dotenv_values
        except ImportError:raise RuntimeError('현재 커널/가상환경에 python-dotenv를 설치하세요.') from None
        values=dotenv_values(path,encoding='utf-8-sig',interpolate=False)
        if 'OPENAI_API_KEY' in values:
            key=(values['OPENAI_API_KEY'] or '').strip()
            if not key:raise ValueError('.env의 OPENAI_API_KEY가 비어 있습니다.')
            os.environ['OPENAI_API_KEY']=key
    if not os.environ.get('OPENAI_API_KEY','').strip():raise ValueError('OPENAI_API_KEY를 .env에 설정하세요. 키 값은 출력하지 않습니다.')
    return {'key_present':True,'env_file':str(path) if path else None}
