"""All-document Phase 5 runner with source-bound short refs and durable budgets.

Planning, corpus checks, cache reads and review never call a model service.
Only run(..., allow_api=True) may invoke generate, after durable reservation.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import uuid
import zipfile

from rfp_rag.generation.llm import LLMConfig
from rfp_rag.summary_fields.fields import FIELDS, FIELD_SCHEMA, FIELD_QUESTION, parse_field_claim, parse_missing_label
from .transport import ShortCitationLLM, TRANSPORT_VERSION
from .catalog_audit import build_catalog, get_validated_document_chunks, select_documents

VERSION = 'rfp-phase5-final-source-units-v1'
TASK = 'phase5_final_fields'
QUESTION = FIELD_QUESTION + '''
마지막 확인: 단가계약·실제 인원/물량 정산, 과업기간 이후 지원, 안정화 포함,
익명 평가용 원본/사본 구분, USB 수록 형식, 낙찰자 선정방식과 평가배점을 함께 살피세요.
명시된 조건은 해당 항목에 남기고 별도 공고의 실제 내용을 추정하지 마세요.
근거 단위 번호를 선택하고 quote는 SOURCE로 고정합니다. 원문은 프로그램이 연결합니다.
'''
_TERMINAL_FAILURES = {'api_failed', 'validation_failed', 'cache_invalid', 'interrupted'}
_STATE_LABELS = {'evidence_found':'근거 추출·검토 필요', 'notice_reference':'별도 자료 참조',
    'conflict_candidates':'상충 후보·검토 필요', 'not_found_in_processed_input':'처리한 입력에서 검증된 근거 미확보',
    'rejected_evidence':'생성 주장을 거절함·원문 재검토 필요', 'not_processed':'입력 일부 미처리'}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temporary.replace(path)


def _utc(): return datetime.now(timezone.utc).isoformat()


@contextmanager
def _run_lock(folder):
    """OS locks release after a kernel/process exits; no silent simultaneous jobs."""
    path = Path(folder)/'execution.lock';path.parent.mkdir(parents=True,exist_ok=True)
    f = path.open('a+')
    try:
        try:
            import fcntl
            try: fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: raise RuntimeError('같은 실행 ID를 다른 커널에서 실행 중입니다.') from None
        except ImportError:
            import msvcrt
            f.seek(0);f.write('1');f.flush();f.seek(0)
            try: msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            except OSError: raise RuntimeError('같은 실행 ID를 다른 커널에서 실행 중입니다.') from None
        yield
    finally:
        f.close()


@dataclass(frozen=True)
class FinalConfig:
    model: str = 'gpt-5.4-mini'
    max_output_tokens: int = 6000
    reasoning_effort: str = 'none'
    max_request_bytes: int = 60000
    batch_max_bytes: int = 35000
    max_batches_per_document: int = 200
    max_total_calls: int = 100
    max_total_input_bytes: int = 6000000
    reuse_verified_legacy: bool = False

    def __post_init__(self):
        for key in ('max_output_tokens','max_request_bytes','batch_max_bytes','max_batches_per_document',
                    'max_total_calls','max_total_input_bytes'):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:raise ValueError(key+'는 양의 정수여야 합니다.')
        if self.batch_max_bytes>self.max_request_bytes:raise ValueError('배치 상한은 요청 상한 이하여야 합니다.')
        if type(self.reuse_verified_legacy) is not bool:raise TypeError('기존 캐시 사용 옵션은 bool이어야 합니다.')

    def model_settings(self):
        return {k:v for k,v in asdict(self).items() if k not in {'max_total_calls','max_total_input_bytes'}}

    def llm_config(self):
        # Stable transport cache identity. Actual spending permission is the
        # durable global journal below, checked before every generate call.
        return LLMConfig(enabled=True,model=self.model,max_output_tokens=self.max_output_tokens,
            reasoning_effort=self.reasoning_effort,max_request_bytes=self.max_request_bytes,
            max_calls=1000000,max_total_input_bytes=1000000000,cache_enabled=True)


def _binding_citation(registry, chunk, quote, binding, base):
    left,right=binding['char_start'],binding['char_end']
    if (binding['chunk_id']!=chunk['chunk_id'] or type(left)is not int or type(right)is not int
        or left<0 or right>len(chunk['text']) or right<=left or chunk['text'][left:right]!=quote
        or hashlib.sha256(quote.encode()).hexdigest()!=binding['quote_sha256']):
        raise ValueError('source_unit_binding_mismatch')
    ranges=[];coverage=[]
    for span in chunk['source_spans']:
        a,b=max(left,span['chunk_start']),min(right,span['chunk_end'])
        if a>=b:continue
        item={'collection':span['collection'],'block_id':span['block_id'],'locator':span['locator'],
            'char_start':span['char_start']+a-span['chunk_start'],
            'char_end':span['char_start']+b-span['chunk_start'],'chunk_start':a,'chunk_end':b}
        for k in ('page','section','table_id','row','col'):
            if k in span:item[k]=span[k]
        ranges.append(item);coverage.append((a-left,b-left))
    cursor=0
    for a,b in sorted(coverage):
        if a>cursor and quote[cursor:a].strip():raise ValueError('source_unit_range_uncovered')
        cursor=max(cursor,b)
    if not ranges or quote[cursor:].strip():raise ValueError('source_unit_range_uncovered')
    result={**base,'quote_start':left,'quote_end':right,'source_ranges':ranges,
        'locator':' / '.join(dict.fromkeys(r['locator'] for r in ranges)),
        'source_unit_alias':binding['alias'],'source_position_verified':True}
    for key in ('page','section'):
        values={r[key] for r in ranges if key in r}
        if len(values)==1:result[key]=next(iter(values))
        else:result.pop(key,None)
    return result


def validate_claims(env, response, chunks, *, legacy=False):
    """Reject an entire unsupported claim, keep valid claims, never invent quotes."""
    canonical=list(env['registry'].validate_chunks(chunks,allowed_doc_ids=[chunks[0]['doc_id']]))
    by_id={c['chunk_id']:c for c in canonical};payload=response['payload']
    transport=response.get('citation_transport') or {}
    bindings={(b['claim_index'],b['citation_index']):b for b in transport.get('source_unit_bindings',[])}
    accepted=[];rejected=[]
    for i,claim in enumerate(payload.get('claims',[]),1):
        field=None
        marker=re.match(r'^\[([a-z_]+)\|',str(claim.get('text',''))) if isinstance(claim,dict) else None
        if marker and marker.group(1) in {f['id'] for f in FIELDS}:field=marker.group(1)
        try:
            field=parse_field_claim(claim)['field_id']
            validation=env['registry'].validate_answer({'status':'answered','claims':[claim],
                'missing_information':[]},canonical,allowed_doc_ids=[canonical[0]['doc_id']])
            if not validation['valid']:raise ValueError('literal_source_validation_failed')
            fresh=validation['claims'][0]
            if not legacy:
                for j,c in enumerate(fresh['citations'],1):
                    binding=bindings.get((i,j))
                    if binding is None:raise ValueError('source_unit_binding_missing')
                    fresh['citations'][j-1]=_binding_citation(env['registry'],by_id[c['chunk_id']],c['quote'],binding,c)
                    fresh['citations'][j-1]['source_unit_mapping_fingerprint']=transport.get('mapping_fingerprint')
            for citation in fresh['citations']:
                citation['source_request_fingerprint']=response.get('request_fingerprint')
                citation['source_origin']='legacy_literal_cache' if legacy else 'source_units'
            accepted.append(fresh)
        except (ValueError,TypeError,KeyError,AttributeError):
            rejected.append({'claim_index':i,'field_id':field,'reason':'field_or_source_validation_failed',
                'raw_claim':copy.deepcopy(claim),'raw_claim_is_unvalidated_diagnostic':True})
    return accepted,rejected


def plan_document(env,llm,doc_id,config,legacy_candidates=()):
    chunks=get_validated_document_chunks(env,doc_id)
    did=chunks[0]['doc_id'] if chunks else env['index'].resolve_doc_ids([doc_id])[0]
    ids=[c['chunk_id'] for c in chunks];batches=[];blocks=[]
    legacy_by_start={}
    for c in legacy_candidates:
        if c['doc_id']==did and c['source_chunk_ids']==ids[c['source_start']:c['source_end']]:
            legacy_by_start.setdefault(c['source_start'],[]).append(c)
    offset=0
    def descriptor(items):return llm.canonical_request_descriptor(TASK,QUESTION,items,schema=FIELD_SCHEMA)
    while offset<len(chunks):
        if offset in legacy_by_start:
            candidate=sorted(legacy_by_start[offset],key=lambda c:(-c['source_end'],c['legacy_request_fingerprint']))[0]
            end=candidate['source_end']
            batches.append({'kind':'legacy_literal_cache','chunks':chunks[offset:end],
                'request_fingerprint':candidate['legacy_request_fingerprint'],'request_input_bytes':0,
                'legacy_candidate':candidate});offset=end;continue
        boundary=min([n for n in legacy_by_start if n>offset]+[len(chunks)])
        try:
            first=descriptor(chunks[offset:offset+1])
            if first['request_input_bytes']>config.batch_max_bytes or not first.get('schema_limits',{}).get('within_limits',True):
                blocks.append('single_chunk_exceeds_wire_or_schema_limit');break
            best=1;probe=2
            while offset+probe<=boundary:
                d=descriptor(chunks[offset:offset+probe])
                if d['request_input_bytes']>config.batch_max_bytes or not d.get('schema_limits',{}).get('within_limits',True):break
                best=probe;probe*=2
            lo,hi=best+1,min(probe,boundary-offset)
            while lo<=hi:
                mid=(lo+hi)//2;d=descriptor(chunks[offset:offset+mid])
                if d['request_input_bytes']<=config.batch_max_bytes and d.get('schema_limits',{}).get('within_limits',True):best=mid;lo=mid+1
                else:hi=mid-1
            items=chunks[offset:offset+best];d=descriptor(items)
            batches.append({'kind':'source_units','chunks':items,'request_fingerprint':d['request_fingerprint'],
                'request_input_bytes':d['request_input_bytes'],'mapping_fingerprint':d['mapping_fingerprint'],
                'unit_count':len(d['mapping']['units']),'descriptor':d});offset+=best
        except (ValueError,TypeError,RuntimeError) as error:
            blocks.append('wire_planning_failed:'+type(error).__name__);break
    if len(batches)>config.max_batches_per_document:blocks.append('document_batch_limit_exceeded')
    if not chunks:blocks.append('document_has_no_chunks')
    if not blocks and [c['chunk_id'] for b in batches for c in b['chunks']]!=ids:
        blocks.append('source_chunk_order_or_coverage_mismatch')
    for i,b in enumerate(batches,1):
        b['batch_id']=f'B{i:04d}';b['source_chunk_ids']=[c['chunk_id'] for c in b['chunks']]
        b['job_key']=did+':'+b['batch_id']+':'+b['request_fingerprint'][:12]
    return {'doc_id':did,'source_sha256':chunks[0]['sha256'] if chunks else None,
        'source_chunk_ids':ids,'chunk_count':len(ids),'batches':batches,'batch_count':len(batches),
        'request_input_bytes':sum(b['request_input_bytes'] for b in batches),
        'new_request_upper_bound':sum(b['kind']=='source_units' for b in batches),
        'can_execute':not blocks,'block_reasons':blocks,
        'plan_fingerprint':digest([(b['kind'],b['source_chunk_ids'],b['request_fingerprint']) for b in batches])}


def public_plan(plan):
    return {k:plan[k] for k in ('doc_id','source_sha256','chunk_count','batch_count','request_input_bytes',
        'new_request_upper_bound','can_execute','block_reasons','plan_fingerprint')}


def _safe_error(error):
    try:
        from rfp_rag.generation.llm import safe_llm_diagnostic
        return safe_llm_diagnostic(error)
    except ImportError:
        return {'code':'generation_failed','exception_type':type(error).__name__}


class FinalRun:
    def __init__(self,env,run_id='phase5_final_source_units',config=None,*,references=None,client=None):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',run_id):raise ValueError('실행 ID는 영문·숫자·밑줄·하이픈으로 지정하세요.')
        self.env=env;self.root=Path(env['root']);self.config=config or FinalConfig();self.references=references
        self.folder=self.root/'phase5_outputs/final_source_units'/run_id
        self.folder.mkdir(parents=True,exist_ok=True);self.client_injection=client
        self.catalog=build_catalog(env)
        self.allowed={r['canonical_doc_id']:r['sha256'] for r in self.catalog}
        self.plan_client=self._client(self.allowed)
        self.legacy_candidates=[];self.legacy_report={'status':'disabled','candidate_count':0,'api_calls':0}
        if self.config.reuse_verified_legacy:
            from .legacy import scan_legacy_batches
            self.legacy_report=scan_legacy_batches(env,self.config.model)
            self.legacy_candidates=self.legacy_report['candidates']
        package=Path(__file__).parent
        code={p.name:file_hash(p) for p in sorted(package.glob('*.py'))}
        baseline={p:file_hash(package.parent/p) for p in ('generation/llm.py','generation/evidence.py','summary_fields/fields.py')}
        self.freeze={'version':VERSION,'transport_version':TRANSPORT_VERSION,'model_settings':self.config.model_settings(),
            'catalog_fingerprint':digest(self.catalog),'index_fingerprint':env['index'].get_metadata()['index_fingerprint'],
            'code_sha256':code,'baseline_code_sha256':baseline,'question_fingerprint':digest(QUESTION),
            'references_fingerprint':digest(references),'execution_mode':'test_double' if client is not None else 'live_openai'}
        self.freeze['legacy_scan_fingerprint']=self.legacy_report.get('scan_fingerprint')
        self.freeze['legacy_candidate_fingerprints']=[c['candidate_fingerprint'] for c in self.legacy_candidates]
        manifest=self.root/'reports/run_manifest.json'
        self.source_files={}
        if manifest.exists():
            for rel in read_json(manifest).get('files_sha256',{}):
                p=self.root/rel
                stat=p.stat();self.source_files[rel]={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        self.fingerprint=digest(self.freeze)
        path=self.folder/'frozen.json'
        if path.exists() and read_json(path)!=self.freeze:raise ValueError('실행 코드·설정·원문이 바뀌었습니다. 새 RUN_ID를 사용하세요.')
        if not path.exists():write_json(path,self.freeze)
        self._plans={}

    def _client(self,allowed,cache_dir=None):
        return ShortCitationLLM(cache_dir or self.root/'phase5_outputs/source_unit_cache',
            config=self.config.llm_config(),allowed_sources=allowed,client=self.client_injection)

    def assert_frozen(self):
        if read_json(self.folder/'frozen.json')!=self.freeze:raise ValueError('고정 실행 기록이 변경됐습니다.')
        package=Path(__file__).parent
        if {p.name:file_hash(p) for p in sorted(package.glob('*.py'))}!=self.freeze['code_sha256']:
            raise ValueError('실행 중 코드가 변경됐습니다. 커널을 재시작하고 새 RUN_ID를 사용하세요.')
        if any(file_hash(package.parent/p)!=h for p,h in self.freeze['baseline_code_sha256'].items()):
            raise ValueError('기반 코드가 변경됐습니다. 새 RUN_ID로 실행하세요.')
        if digest(self.references)!=self.freeze['references_fingerprint']:
            raise ValueError('검토 기준표가 변경됐습니다.')
        for rel,expected in self.source_files.items():
            stat=(self.root/rel).stat()
            if {'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}!=expected:
                raise ValueError('실행 중 원문·인덱스 파일이 변경됐습니다. 새 환경을 로드하세요.')

    def plan(self,doc_id):
        did=select_documents(self.env,[doc_id])[0]
        if did not in self._plans:
            self._plans[did]=plan_document(self.env,self.plan_client,did,self.config,self.legacy_candidates)
        return self._plans[did]

    def audit_all_plans(self):
        """Audit every doc and every alias/span in every planned wire batch; no model."""
        from .transport import decode_payload
        rows=[];total_units=0
        for did in select_documents(self.env,'all'):
            plan=plan_document(self.env,self.plan_client,did,self.config,self.legacy_candidates)
            checked_units=0;issues=[]
            if plan['can_execute']:
                for b in plan['batches']:
                    if b['kind']!='source_units':continue
                    d=b['descriptor'];units=[u for u in d['mapping']['units'] if u['citable']]
                    # Echo every allowed ID in an explicitly synthetic structure.
                    raw={'status':'answered','claims':[{'doc_id':'D001','category':'overview',
                        'text':'[project_name|fact] 출처 번호 왕복 검사용 합성 주장',
                        'citations':[{'chunk_id':u['alias'],'quote':'SOURCE'}]} for u in units],
                        'missing_information':[]}
                    restored,bindings=decode_payload(raw,d)
                    byid={c['chunk_id']:c for c in b['chunks']}
                    for claim,binding in zip(restored['claims'],bindings):
                        c=claim['citations'][0]
                        if byid[c['chunk_id']]['text'][binding['char_start']:binding['char_end']]!=c['quote']:
                            issues.append('source_unit_roundtrip_mismatch')
                    if not d['unit_coverage']['passed']:issues.append('source_unit_coverage_failed')
                    checked_units+=len(units)
            total_units+=checked_units
            rows.append({**public_plan(plan),'source_units_checked':checked_units,'id_transport_passed':not issues and plan['can_execute'],
                'audit_errors':issues})
        report={'version':VERSION,'document_count':len(rows),'source_units_checked':total_units,
            'run_fingerprint':self.fingerprint,
            'all_document_id_transport_passed':bool(rows) and all(r['id_transport_passed'] for r in rows),
            'blocked_documents':[r['doc_id'] for r in rows if not r['can_execute']],
            'rows':rows,'api_calls':0,'semantic_quality_verified':False,
            'scope':'모든 문서의 입력 계획·근거 번호·원문 범위 왕복 검사. 모델 요약 품질 평가는 아닙니다.'}
        write_json(self.folder/'all_document_plans.json',report)
        return report

    def _journal(self):
        path=self.folder/'journal.json'
        value=read_json(path) if path.exists() else {'run_fingerprint':self.fingerprint,'jobs':{}}
        if value['run_fingerprint']!=self.fingerprint:raise ValueError('실행 기록의 버전이 다릅니다.')
        return value

    def _save_journal(self,value):write_json(self.folder/'journal.json',value)

    def budget_usage(self):
        calls=size=live=mock=unknown=0
        for job in self._journal()['jobs'].values():
            for a in job.get('attempts',[]):
                usage=a.get('usage')
                if usage is None:
                    calls+=a.get('reserved_calls',0);size+=a.get('reserved_input_bytes',0)
                    unknown+=a.get('reserved_calls',0)
                else:
                    calls+=usage.get('api_calls',0);size+=usage.get('attempted_input_bytes',0)
                    live+=usage.get('live_api_calls',0);mock+=usage.get('mock_calls',0)
                    unknown+=usage.get('usage_unknown_attempts',0)
        return {'calls_used_or_reserved':calls,'input_bytes_used_or_reserved':size,
            'live_api_calls':live,'mock_calls':mock,'unknown_or_interrupted_calls':unknown,
            'max_total_calls':self.config.max_total_calls,'max_total_input_bytes':self.config.max_total_input_bytes}

    def _load_outcome(self,job,expected_batch=None):
        if not job.get('outcome_file'):return None
        path=self.folder/job['outcome_file']
        if not path.resolve().is_relative_to(self.folder.resolve()) or file_hash(path)!=job['outcome_sha256']:
            raise ValueError('배치 결과 파일 또는 지문이 변경됐습니다.')
        result=read_json(path)
        if expected_batch is not None:
            if (any(result.get(k)!=expected_batch.get(k) for k in ('job_key','request_fingerprint','source_chunk_ids','kind'))
                or job.get('request_fingerprint')!=expected_batch['request_fingerprint']):
                raise ValueError('배치 결과가 현재 문서·요청·입력 범위와 다릅니다.')
        return result

    def _store_outcome(self,journal,b,outcome,*,attempt=None):
        key=b['job_key'];job=journal['jobs'].setdefault(key,{'attempts':[]})
        path=self.folder/'batches'/b['request_fingerprint']/(uuid.uuid4().hex+'.json')
        write_json(path,outcome)
        job.update(status=outcome['status'],outcome_file=path.relative_to(self.folder).as_posix(),
            outcome_sha256=file_hash(path),request_fingerprint=b['request_fingerprint'])
        if attempt is not None:job['attempts'][attempt]['status']=outcome['status']
        self._save_journal(journal)

    def _accepted_outcome(self,b,response,*,legacy=False):
        claims,rejected=validate_claims(self.env,response,b['chunks'],legacy=legacy)
        for r in rejected:
            r.update(batch_id=b['batch_id'],rejection_id=b['request_fingerprint'][:16]+f'-{r["claim_index"]}')
        return {'batch_id':b['batch_id'],'job_key':b['job_key'],'source_chunk_ids':b['source_chunk_ids'],
            'request_fingerprint':b['request_fingerprint'],'kind':b['kind'],
            'status':'completed_with_rejections' if rejected else 'completed',
            'claims':claims,'rejected_claims':rejected,'missing_information':response['payload'].get('missing_information',[]),
            'response_received':True,'claims_checked':True,'cache_hit':bool(response.get('cache_hit')),
            'execution_mode':response.get('execution_mode'),'live_api':bool(response.get('live_api')),
            'mapping_fingerprint':(response.get('citation_transport') or {}).get('mapping_fingerprint'),
            'original_usage':response.get('original_usage',response.get('usage',{})),
            'semantic_quality_verified':False}

    def preflight(self,doc_ids):
        selected=select_documents(self.env,doc_ids);journal=self._journal();calls=size=0;rows=[]
        for did in selected:
            p=self.plan(did);pending=cache=done=fail=0
            for b in p['batches'] if p['can_execute'] else []:
                job=journal['jobs'].get(b['job_key'],{})
                if job.get('status') in {'completed','completed_with_rejections'}:done+=1;continue
                if job.get('status') in _TERMINAL_FAILURES or job.get('status')=='running':fail+=1;continue
                if b['kind']=='legacy_literal_cache':cache+=1;continue
                try:response=self.plan_client.offline_replay(TASK,QUESTION,b['chunks'],schema=FIELD_SCHEMA)
                except Exception:fail+=1;continue
                if response is None:pending+=1;size+=b['request_input_bytes']
                else:cache+=1
            calls+=pending
            rows.append({**public_plan(p),'uncached_requests':pending,'available_cached_batches':cache,
                'saved_batches':done,'failed_or_interrupted_batches':fail})
        used=self.budget_usage()
        return {'selected_doc_ids':selected,'rows':rows,'uncached_calls':calls,'uncached_input_bytes':size,
            'budget':used,'within_remaining_budget':used['calls_used_or_reserved']+calls<=self.config.max_total_calls
                and used['input_bytes_used_or_reserved']+size<=self.config.max_total_input_bytes,
            'api_calls_by_preflight':0,'scope':'입력 byte와 호출 횟수 상한. 토큰·달러 금액 상한이 아닙니다.'}

    def run(self,doc_ids,*,allow_api=False,retry_jobs=(),retry_tag='',resume_after_service_error=False):
        if type(allow_api)is not bool:raise TypeError('API 실행 옵션은 bool이어야 합니다.')
        if retry_jobs and (not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',retry_tag)):
            raise ValueError('재시도할 때는 새 RETRY_TAG를 지정하세요.')
        selected=select_documents(self.env,doc_ids);self.assert_frozen()
        allowed={did:self.allowed[did] for did in selected}
        with _run_lock(self.folder):
            journal=self._journal();budget_halted=False
            if resume_after_service_error:
                journal.pop('service_blocker',None);self._save_journal(journal)
            service_halted=bool(journal.get('service_blocker'))
            for did in selected:
                plan=self.plan(did)
                print('문서 처리:',did,flush=True)
                if not plan['can_execute']:
                    self.result(did);continue
                for b in plan['batches']:
                    key=b['job_key'];job=journal['jobs'].get(key,{})
                    previous=self._load_outcome(job,b)
                    if previous and previous['status'] in {'completed','completed_with_rejections'}:continue
                    retry=key in retry_jobs and retry_tag not in [a.get('retry_tag') for a in job.get('attempts',[])]
                    if previous and previous['status'] in _TERMINAL_FAILURES and not retry:continue
                    client=self._client(allowed)
                    if retry:
                        client=self._client(allowed,self.root/'phase5_outputs/source_unit_retry_cache'/retry_tag)
                    elif job.get('status')=='running' and job.get('attempts'):
                        previous_cache=job['attempts'][-1].get('cache_relative')
                        if previous_cache:
                            path=(self.root/previous_cache).resolve()
                            if not path.is_relative_to((self.root/'phase5_outputs').resolve()):
                                raise ValueError('재실행 캐시 경로가 프로젝트 출력 폴더를 벗어납니다.')
                            client=self._client(allowed,path)
                    if b['kind']=='legacy_literal_cache':
                        c=b['legacy_candidate'];response={'payload':c['payload'],'cache_hit':True,
                            'execution_mode':'live_openai','live_api':True,'original_usage':c['original_usage'],
                            'request_fingerprint':c['legacy_request_fingerprint']}
                        self._store_outcome(journal,b,self._accepted_outcome(b,response,legacy=True));continue
                    try:cached=client.offline_replay(TASK,QUESTION,b['chunks'],schema=FIELD_SCHEMA)
                    except Exception as error:
                        outcome=self._failure(b,'cache_invalid',error)
                        self._store_outcome(journal,b,outcome);continue
                    if cached is not None:
                        self._store_outcome(journal,b,self._accepted_outcome(b,cached));continue
                    if job.get('status')=='running' and not retry:
                        self._store_outcome(journal,b,self._failure(b,'interrupted',None));continue
                    if not allow_api or budget_halted or service_halted:continue
                    used=self.budget_usage()
                    if (used['calls_used_or_reserved']+1>self.config.max_total_calls or
                        used['input_bytes_used_or_reserved']+b['request_input_bytes']>self.config.max_total_input_bytes):
                        budget_halted=True;continue
                    self.assert_frozen()
                    job=journal['jobs'].setdefault(key,{'attempts':[]})
                    attempt={'status':'running','reserved_calls':1,'reserved_input_bytes':b['request_input_bytes'],
                        'request_fingerprint':b['request_fingerprint'],'retry_tag':retry_tag if retry else '',
                        'cache_relative':client.cache_dir.relative_to(self.root).as_posix(),
                        'usage':None,'started_at_utc':_utc()}
                    job['attempts'].append(attempt);number=len(job['attempts'])-1;job['status']='running'
                    job.pop('outcome_file',None);job.pop('outcome_sha256',None)
                    self._save_journal(journal)  # Durable before SDK/network invocation.
                    try:
                        response=client.generate(TASK,QUESTION,b['chunks'],schema=FIELD_SCHEMA)
                        outcome=self._accepted_outcome(b,response)
                    except Exception as error:
                        received=bool((getattr(error,'diagnostic',None) or {}).get('response_received'))
                        outcome=self._failure(b,'validation_failed' if received else 'api_failed',error)
                        outcome['response_received']=received
                        if not received:
                            journal['service_blocker']={'job_key':key,'error_diagnostic':outcome['error_diagnostic']}
                            service_halted=True
                    attempt['usage']=client.usage_report();attempt['finished_at_utc']=_utc()
                    self._store_outcome(journal,b,outcome,attempt=number)
                    print(' ',b['batch_id'],outcome['status'],flush=True)
                self.result(did)
        return {'selected_doc_ids':selected,'budget':self.budget_usage(),'allow_api':allow_api,
            'budget_halted':budget_halted,'service_halted':service_halted,
            'automatic_retry':False,'output_folder':str(self.folder)}

    def _failure(self,b,status,error):
        return {'batch_id':b['batch_id'],'job_key':b['job_key'],'source_chunk_ids':b['source_chunk_ids'],
            'request_fingerprint':b['request_fingerprint'],'kind':b['kind'],'status':status,
            'claims':[],'rejected_claims':[],'missing_information':[],
            'response_received':False,'claims_checked':False,'live_api':False,
            'error_diagnostic':_safe_error(error) if error is not None else {'code':'interrupted_no_usable_response'},
            'automatic_retry':False}

    def result(self,doc_id):
        did=select_documents(self.env,[doc_id])[0];plan=self.plan(did);journal=self._journal()
        outcomes=[];received=set();checked=set();claims=[];rejected=[];missing=[]
        for b in plan['batches']:
            job=journal['jobs'].get(b['job_key'],{});outcome=self._load_outcome(job,b)
            if outcome is None:outcome=self._failure(b,'interrupted' if job.get('status')=='running' else 'awaiting_api',None)
            outcomes.append(outcome)
            if outcome['response_received']:received.update(b['source_chunk_ids'])
            if outcome['claims_checked']:checked.update(b['source_chunk_ids'])
            claims.extend(outcome['claims']);rejected.extend(outcome['rejected_claims'])
            for label in outcome['missing_information']:
                parsed=parse_missing_label(label)
                if parsed:missing.append(f'[{parsed["field_id"]}] 입력 {b["batch_id"]}에서 미확인: {parsed["text"]}')
                else:missing.append(f'입력 {b["batch_id"]}에서 미확인: '+str(label))
        source=plan['source_chunk_ids'];complete=bool(source) and len(checked)==len(source) and plan['can_execute']
        raw_accepted_count=len(claims)
        merged=[];by_text={}
        for claim in claims:
            key=(claim['doc_id'],claim['category'],claim['text'])
            if key not in by_text:
                fresh=copy.deepcopy(claim);merged.append(fresh);by_text[key]=fresh
            else:
                target=by_text[key]['citations']
                known={(c['chunk_id'],c['quote'],c.get('quote_start'),c.get('quote_end')) for c in target}
                for cite in claim['citations']:
                    identity=(cite['chunk_id'],cite['quote'],cite.get('quote_start'),cite.get('quote_end'))
                    if identity not in known:target.append(copy.deepcopy(cite));known.add(identity)
        claims=merged;fields=[]
        for spec in FIELDS:
            accepted=[];count=0;notes=[]
            for c in claims:
                parsed=parse_field_claim(c)
                if parsed['field_id']==spec['id']:accepted.append({'text':parsed['text'],'mode':parsed['mode'],'citations':c['citations']})
            count=sum(r.get('field_id')==spec['id'] for r in rejected)
            modes={c['mode'] for c in accepted}
            state=('conflict_candidates' if 'conflict_candidate'in modes else 'evidence_found' if 'fact'in modes else
                'notice_reference' if 'notice_reference'in modes else 'rejected_evidence' if count else
                'not_found_in_processed_input' if complete else 'not_processed')
            for m in missing:
                parsed=parse_missing_label(m)
                if parsed and parsed['field_id']==spec['id']:notes.append(parsed['text'])
            fields.append({'field_id':spec['id'],'label':spec['label'],'state':state,'claims':accepted,
                'rejected_claim_count':count,'notes':list(dict.fromkeys(notes))})
        all_live=self.client_injection is None and bool(outcomes) and all(o.get('claims_checked') and o.get('execution_mode')=='live_openai'
            and o.get('live_api') is True for o in outcomes)
        execution=('completed_with_rejections' if rejected else 'completed') if complete else (
            'plan_blocked' if not plan['can_execute'] else
            'incomplete' if any(o['status'] in _TERMINAL_FAILURES for o in outcomes) else 'awaiting_api')
        status=('answered' if complete and not rejected and not missing and all(f['state']=='evidence_found' for f in fields) else
            'partial' if claims else 'insufficient_evidence' if complete else 'not_generated')
        result={'version':VERSION,'run_fingerprint':self.fingerprint,'doc_id':did,'source_sha256':plan['source_sha256'],
            'status':status,'execution_status':execution,'answer_generated':bool(claims),'claims':claims,'fields':fields,
            'rejected_claims':rejected,'batch_outcomes':outcomes,'source_chunk_ids':source,
            'input_received_chunk_ids':[c for c in source if c in received],
            'checked_chunk_ids':[c for c in source if c in checked],
            'input_chunk_coverage':len(received)/len(source) if source else 0,
            'checked_chunk_coverage':len(checked)/len(source) if source else 0,
            'raw_accepted_claim_count':raw_accepted_count,
            'raw_claim_acceptance_rate':raw_accepted_count/(raw_accepted_count+len(rejected)) if raw_accepted_count or rejected else None,
            'raw_claim_acceptance_is_semantic_accuracy':False,'model_origin':{'all_live_or_live_cache':all_live},
            'missing_information':list(dict.fromkeys(missing)), 'plan_block_reasons':plan['block_reasons'],
            'semantic_quality_verified':False,'all_documents_quality_verified':False}
        from .review import create_review_files, answer_fingerprint
        folder=self.folder/'documents'/did/answer_fingerprint(result)[:20]
        folder.mkdir(parents=True,exist_ok=True)
        write_json(folder/'result.json',result)
        (folder/'answer.md').write_text(render_result(result),encoding='utf-8')
        create_review_files(folder,result,self.references)
        write_json(self.folder/'documents'/did/'latest.json',{'folder':folder.relative_to(self.folder).as_posix(),
            'answer_fingerprint':answer_fingerprint(result)})
        return result

    def report(self,doc_ids,reviewer=''):
        from .review import assess_review_files
        selected=select_documents(self.env,doc_ids);cases=[]
        for did in selected:
            result=self.result(did)
            latest=read_json(self.folder/'documents'/did/'latest.json')
            folder=self.folder/latest['folder']
            review=assess_review_files(folder,result,reviewer)
            cases.append({'doc_id':did,'execution_status':result['execution_status'],'status':result['status'],
                'input_chunk_coverage':result['input_chunk_coverage'],'checked_chunk_coverage':result['checked_chunk_coverage'],
                'rejected_claim_count':len(result['rejected_claims']),'review':review,
                'result_path':str(folder/'result.json'),'review_path':str(folder/'comparison.csv')})
        samples={s['doc_id'] for s in (self.references or {}).get('samples',[])}
        evaluated={c['doc_id']:c for c in cases}
        sample_passed=bool(samples) and samples.issubset(evaluated) and all(
            evaluated[d]['review'].get('case_accepted') is True for d in samples)
        technical_path=self.folder/'all_document_plans.json';catalog_path=self.folder/'catalog_audit.json'
        technical=read_json(technical_path) if technical_path.exists() else {}
        catalog=read_json(catalog_path) if catalog_path.exists() else {}
        tests_path=self.folder/'offline_tests.json';tests=read_json(tests_path) if tests_path.exists() else {}
        reports_bound=all(r.get('run_fingerprint')==self.fingerprint for r in (technical,catalog,tests))
        passed=sample_passed and reports_bound and technical.get('all_document_id_transport_passed') is True and catalog.get('passed') is True and tests.get('passed') is True
        report={'version':VERSION,'run_fingerprint':self.fingerprint,'phase5_sample_validation_passed':passed,
            'all_document_id_transport_passed':technical.get('all_document_id_transport_passed',False),
            'sample_review_passed':sample_passed,'cases':cases,'budget':self.budget_usage(),
            'all_documents_semantic_quality_verified':False,'automatic_retry':False,
            'scope':'전 문서 ID·전송 검사와 고정 7건 요약 내용 평가를 구분합니다.'}
        write_json(self.folder/'final_report.json',report);return report

    def export(self,doc_ids,reviewer=''):
        self.report(doc_ids,reviewer)
        target=self.folder.parent/(self.folder.name+'_results.zip')
        with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(self.folder.rglob('*')):
                if p.is_file() and p.suffix in {'.json','.csv','.md','.log'}:
                    z.write(p,self.folder.name+'/'+p.relative_to(self.folder).as_posix())
        return target


def render_result(result):
    def esc(s):
        text=html.escape(str(s),quote=False)
        text=re.sub(r'([\\`*_{}\[\]()#!+\-.])',r'\\\1',text)
        return text.replace('|','&#124;').replace('\n',' / ')
    lines=['| 항목 | 검증된 원문에 연결된 내용 | 상태 |','| --- | --- | --- |'];refs=[]
    for f in result['fields']:
        text=[]
        for c in f['claims']:
            nums=[]
            for cite in c['citations']:
                refs.append(cite);nums.append('['+str(len(refs))+']')
            text.append(esc(c['text'])+' '+''.join(nums))
        lines.append('| '+esc(f['label'])+' | '+(' / '.join(text) or '검증된 근거 미확보')+' | '+esc(_STATE_LABELS[f['state']])+' |')
    if refs:
        lines+=['','원문 근거:','']
        for i,c in enumerate(refs,1):lines.append(f'{i}. '+esc(c['file_name'])+' · '+esc(c['locator'])+' — “'+esc(c['quote'])+'”')
    if result.get('missing_information'):
        lines+=['','추가 확인(부분 입력에 대해 모델이 표시한 미확인 사항):','']
        lines.extend('- '+esc(label) for label in result['missing_information'])
        lines+=['','다른 입력 부분에서 해결됐는지는 자동 확정하지 않습니다. 문서 전체의 정보 부재를 뜻하지 않습니다.']
    lines+=['',f'입력 응답 수신율: {result["input_chunk_coverage"]:.1%} / 주장 검사 완료율: {result["checked_chunk_coverage"]:.1%}',
        f'검증에서 제외한 생성 주장: {len(result["rejected_claims"])}개. 제외 내용은 별도 진단·검토 파일에 있습니다.',
        '원문 인용의 연결을 검사했습니다. 주장 정확성·요약 완전성·조건 보존은 사람 검토가 필요합니다.']
    return '\n'.join(lines)+'\n'
