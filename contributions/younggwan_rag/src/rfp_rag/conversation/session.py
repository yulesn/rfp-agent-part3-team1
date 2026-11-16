"""Resolve conversational references without treating assistant text as evidence.

Planning is deterministic and offline. Source retrieval and generation are an
injected backend; the selected scope can never silently become a global search.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from rfp_rag.retrieval.session import _project_title
from . import VERSION
from .query import plan_content_queries
from .resolution import MetadataResolver
from .focus import WORK_SCOPE_PATTERN, ACCEPTANCE_PATTERN


_UNSET=object()
_SINGULAR=re.compile(r'그건|그것|그럼|그\s*(?:사업|안|기간|금액)|해당\s*(?:사업|문서)|이거|이것|이\s*(?:사업|문서)|거기|여기')
_PLURAL=re.compile(r'둘\s*다|두\s*(?:사업|문서|건|RFP|ISP)|각각|비교|차이|모두|전체', re.I)
_SWITCH=re.compile(r'바꿔\s*(?:줘|주|줄|보자)|바꿀게|바꾸(?:자|겠|고\s*싶)|'
    r'(?:으로|로)\s*(?:전환|변경)|(?:문서|대상|사업)\s*(?:을|를)?\s*(?:바꿔|변경|전환)|다른\s*(?:문서|사업)')
_FIRST=re.compile(r'앞(?:의|쪽)?\s*(?:사업|문서)|첫\s*번째|'
                  r'(?<![가-힣a-z0-9])전자(?:는|의|가|를|도|만|부터|와|과)?(?=$|[\s,.?!:])', re.I)
_SECOND=re.compile(r'뒤(?:의|쪽)?\s*(?:사업|문서)|두\s*번째|'
                   r'(?<![가-힣a-z0-9])후자(?:는|의|가|를|도|만|부터|와|과)?(?=$|[\s,.?!:])', re.I)
_FACETS=(('duration',r'기간|개월|며칠|몇\s*(?:달|일|주)|안정화|언제까지'),
    ('budget',r'예산|금액|용역비|사업비|부가세|부가가치세|VAT|세금|얼마'),
    ('eligibility',r'자격|공동수급|공동도급|하도급|참가조건'),
    ('submission',r'제출|마감|접수|서류|방문|우편'),
    ('evaluation',r'평가|배점|낙찰|협상|점수'),
    ('work_scope',WORK_SCOPE_PATTERN),
    ('acceptance',ACCEPTANCE_PATTERN),
    ('requirements',r'요구사항|필수|과업(?!기간)|지원|기능|GPU'))
_FACET_LABELS={'duration':'사업기간', 'budget':'사업예산', 'eligibility':'참가자격',
               'submission':'제출 조건', 'evaluation':'평가기준', 'requirements':'요구사항',
               'work_scope':'구체적인 사업·과업 범위', 'acceptance':'검수·완료 조건'}


def _norm(text):
    return ''.join(c for c in unicodedata.normalize('NFC',str(text)).casefold() if c.isalnum())


def _digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ConversationConfig:
    max_history_turns: int = 8
    max_question_chars: int = 2000
    max_rewrite_chars: int = 3500
    previous_question_chars: int = 800

    def __post_init__(self):
        for name,value in asdict(self).items():
            if type(value)is not int or value<1:raise ValueError(name+'는 양의 정수여야 합니다.')
        if self.max_rewrite_chars<self.max_question_chars:
            raise ValueError('검색 질문 상한은 사용자 질문 상한 이상이어야 합니다.')


class ConversationSession:
    def __init__(self, documents, *, backend=None, config=None):
        self.config=config or ConversationConfig();self.backend=backend
        self._documents={};self._aliases={}
        for d in documents:
            did=d.get('doc_id')
            if not isinstance(did,str) or not did or did in self._documents:
                raise ValueError('문서 ID가 누락되었거나 중복됐습니다.')
            self._documents[did]=copy.deepcopy(d)
            self._aliases[did]=d.get('canonical_doc_id',did)
        self._catalog={}
        for did,d in self._documents.items():
            target=self._aliases[did]
            if target not in self._documents or self._aliases[target]!=target:
                raise ValueError('대표 문서 연결이 잘못됐습니다.')
            if not d.get('sha256') or d.get('sha256')!=self._documents[target].get('sha256'):
                raise ValueError('원본/대표 문서 버전이 다릅니다.')
            if target==did:
                self._catalog[did]={k:copy.deepcopy(d.get(k)) for k in
                    ('doc_id','sha256','file_name','agency','document_year','detected_format')}
                self._catalog[did]['project_title']=_project_title(d)
        self.catalog_fingerprint=_digest({'catalog':self._catalog,'aliases':self._aliases})
        self._resolver=MetadataResolver(self._catalog)
        self._active=[];self._focus=[];self._filters={};self._history=[];self._recent=[]
        self._epoch=0;self._pending=None;self._turn=0

    @property
    def active_doc_ids(self):return list(self._active)

    @property
    def filters(self):return copy.deepcopy(self._filters)

    @property
    def state(self):
        return copy.deepcopy({'version':VERSION,'catalog_fingerprint':self.catalog_fingerprint,
            'active_doc_ids':self._active,'focus_doc_ids':self._focus,'filters':self._filters,
            'history':self._history,'recent_doc_ids':self._recent,
            'epoch':self._epoch,'turn':self._turn,'pending':self._pending})

    def list_documents(self,keyword='',*,filtered=False,include_aliases=False):
        if not isinstance(keyword,str):raise TypeError('검색어는 문자열이어야 합니다.')
        rows=[];needle=_norm(keyword)
        for did,target in self._aliases.items():
            if not include_aliases and did!=target:continue
            row=copy.deepcopy(self._catalog[target])
            row.update(doc_id=did,canonical_doc_id=target,is_alias=did!=target)
            if filtered and not self._matches_filter(target):continue
            if needle and needle not in _norm(' '.join(str(row.get(k) or '') for k in
                ('doc_id','canonical_doc_id','project_title','file_name','agency','document_year'))):continue
            rows.append(row)
        return rows

    def _resolve(self,ids):
        if ids=='all':return list(self._catalog)
        if isinstance(ids,str):ids=[ids]
        if not isinstance(ids,(list,tuple)) or not ids:
            raise ValueError('문서 ID를 한 개 이상 지정하세요.')
        if any(not isinstance(x,str) or x not in self._aliases for x in ids):
            raise ValueError('목록에 없는 문서 ID입니다. 전체 문서로 범위를 넓히지 않습니다.')
        return list(dict.fromkeys(self._aliases[x] for x in ids))

    def _matches_filter(self,did,filters=None):
        row=self._catalog[did]
        return all(_norm(row.get(k,''))==_norm(v) for k,v in (self._filters if filters is None else filters).items())

    @staticmethod
    def _validate_filters(filters):
        if not isinstance(filters,dict) or set(filters)-{'agency','document_year','detected_format'}:
            raise ValueError('필터는 agency, document_year, detected_format만 지원합니다.')
        result={}
        for key,value in filters.items():
            if key=='document_year':
                if type(value)is not int:raise ValueError('document_year는 정수여야 합니다.')
            elif not isinstance(value,str) or not value.strip():raise ValueError('문자 필터를 입력하세요.')
            if key=='detected_format' and value not in {'hwp','hwpx','pdf'}:raise ValueError('지원하지 않는 문서 형식입니다.')
            result[key]=value.strip() if isinstance(value,str) else value
        return result

    def _boundary(self):
        self._epoch+=1;self._history=[];self._focus=[];self._pending=None

    def select_documents(self,ids):
        chosen=self._resolve(ids)
        if any(not self._matches_filter(d) for d in chosen):raise ValueError('선택한 문서와 유지 중인 필터가 일치하지 않습니다.')
        if chosen!=self._active:self._boundary();self._active=chosen
        self._focus=list(chosen)
        self._remember(chosen)
        return self.active_doc_ids

    def _remember(self,ids):
        # Only identities survive a scope boundary. No old question or answer
        # is appended to a new business's retrieval input.
        self._recent=[d for d in self._recent if d not in ids]+list(ids)
        self._recent=self._recent[-max(2,self.config.max_history_turns):]

    def set_filters(self,filters):
        values=self._validate_filters(filters)
        if values!=self._filters:
            self._boundary();self._filters=values
            self._recent=[]
            # A filter conflict is visible; it never silently becomes all docs.
            if any(not self._matches_filter(d) for d in self._active):self._active=[]
            self._remember(self._active)
        return self.filters

    def reset(self):
        self._boundary();self._active=[];self._filters={};self._turn=0;self._recent=[]

    def _intent_text(self,query):
        # A project called "비교분석 시스템" does not itself request a comparison.
        return self._resolver.mask_titles(query)

    def _named(self,query,preferred=None):
        return self._resolver.mentions(query,preferred=self._active if preferred is None else preferred)

    def _reply(self,status,query,message,*,candidates=(),reason=None):
        candidate_ids=list(dict.fromkeys(candidates))
        if status=='needs_clarification':
            self._pending={'query':query,'candidate_doc_ids':candidate_ids,'epoch':self._epoch}
        return {'status':status,'execution_status':'not_requested','answer_generated':False,
            'query':query,'standalone_query':None,'active_doc_ids':self.active_doc_ids,
            'effective_doc_ids':[],'filters':self.filters,'claims':[],
            'markdown':message,'reason':reason,'candidate_documents':[
                copy.deepcopy(self._catalog[d]) for d in candidate_ids],
            'routing_api_calls':0,'query_rewrite_uses_assistant_answers':False}

    def plan(self,query,*,doc_ids=_UNSET,filters=_UNSET):
        if not isinstance(query,str):raise TypeError('질문은 문자열이어야 합니다.')
        query=query.strip()
        if not query or len(query)>self.config.max_question_chars:
            return self._reply('needs_clarification',query,'질문이 비어 있거나 길이 상한을 넘었습니다.',reason='invalid_question')
        # Validate all explicit scope options before changing current state.
        try:
            newfilters=self._filters if filters is _UNSET else self._validate_filters(filters)
            explicit=None if doc_ids is _UNSET else self._resolve(doc_ids)
            if explicit and any(not self._matches_filter(d,newfilters) for d in explicit):
                raise ValueError('문서 선택과 필터가 서로 다릅니다.')
        except (ValueError,TypeError):
            return self._reply('needs_clarification',query,'문서 ID와 필터를 확인해 주세요. 현재 범위는 유지했습니다.',reason='invalid_scope')
        prior_facet=self._history[-1].get('facet') if self._history else None
        # A named candidate is also a clarification reply. Resume the pending
        # subject instead of searching for "I meant <title>" as a new question.
        if self._pending and explicit is None and not _SWITCH.search(self._intent_text(query)):
            reply_ids,reply_basis=self._named(query)
            if (len(reply_ids)==1 and reply_basis=='title'
                and reply_ids[0] in self._pending['candidate_doc_ids']
                and not any(re.search(pattern,self._intent_text(query),re.I) for _,pattern in _FACETS)):
                explicit=reply_ids;query=self._pending['query']
        numeric=re.fullmatch(r'(\d+)\s*번?',query)
        if numeric and self._pending:
            candidates=self._pending['candidate_doc_ids'];number=int(numeric.group(1))-1
            if not 0<=number<len(candidates):return self._reply('needs_clarification',self._pending['query'],'후보 목록의 번호를 선택해 주세요.',candidates=candidates,reason='invalid_candidate_number')
            explicit=[candidates[number]];query=self._pending['query']
        elif explicit and self._pending and re.fullmatch(r'이거|그거|이\s*문서|그\s*문서|선택|확인',query):
            if not set(explicit).issubset(self._pending['candidate_doc_ids']):
                return self._reply('needs_clarification',self._pending['query'],'질문에 맞는 후보를 선택해 주세요.',candidates=self._pending['candidate_doc_ids'],reason='selection_outside_pending_candidates')
            query=self._pending['query']
        if explicit and any(not self._matches_filter(d,newfilters) for d in explicit):
            return self._reply('needs_clarification',query,'선택 후보와 현재 필터가 다릅니다.',reason='candidate_outside_filters')
        intent=self._intent_text(query)
        switching=bool(_SWITCH.search(intent))
        preferred=explicit if explicit is not None else () if switching else self._active
        named,basis=self._named(query,preferred=preferred)
        if (self._pending and explicit is None and not named and not switching
            and _SWITCH.search(self._intent_text(self._pending['query']))):
            return self._reply('needs_clarification',self._pending['query'],
                '전환할 문서를 먼저 확인해 주세요. 문서 ID 또는 사업명을 지정하세요.',
                candidates=self._pending['candidate_doc_ids'],reason='switch_target_still_pending')
        named_focus=None
        if explicit and named and (not set(explicit)&set(named)
            or basis=='title' and not set(named).issubset(explicit)):
            return self._reply('needs_clarification',query,'선택한 문서와 질문의 사업명이 다릅니다.',candidates=named,reason='explicit_selection_conflicts_with_question')
        years=set(re.findall(r'(?<!\d)((?:19|20)\d{2})\s*년',query))
        if named and years:
            known=set()
            for d in named:
                known.update(re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)',self._catalog[d]['project_title']))
                if self._catalog[d]['document_year'] is not None:known.add(str(self._catalog[d]['document_year']))
            if known and not years&known:
                return self._reply('needs_clarification',query,'질문 연도와 사업·문서 연도의 관계를 확인해 주세요.',candidates=named,reason='year_scope_ambiguous')
        if filters is not _UNSET:self.set_filters(newfilters)
        if explicit:self.select_documents(explicit)
        eligible=[d for d in self._catalog if self._matches_filter(d)]
        if not eligible:return self._reply('needs_clarification',query,'현재 필터에 맞는 문서가 없습니다. 필터를 확인해 주세요.',reason='empty_filtered_scope')
        if explicit is None and named:
            in_scope=[d for d in named if d in eligible]
            if not in_scope:return self._reply('needs_clarification',query,'질문에서 언급한 문서가 현재 필터 밖에 있습니다.',candidates=eligible,reason='named_document_outside_filters')
            if (self._active and not set(in_scope).issubset(self._active)
                and re.search(r'(?:이|현재|해당)\s*문서.{0,80}(?:언급|기재|적혀|나오|쓰여)|문서\s*안',query)):
                return self._reply('needs_clarification',query,'현재 문서 안의 언급을 묻는지, 다른 사업으로 전환할지 확인해 주세요.',
                    candidates=list(dict.fromkeys(self._active+in_scope)),reason='mentioned_project_or_switch')
            preserve_comparison=(bool(_PLURAL.search(intent)) and len(self._active)>=2
                and set(in_scope).issubset(self._active) and not switching)
            if not preserve_comparison and (basis=='ambiguous_titles'
                or len(in_scope)>1 and not _PLURAL.search(intent) and not (basis=='agency' and self._active and set(self._active).issubset(in_scope))):
                return self._reply('needs_clarification',query,'대상 사업이 여러 개입니다. 문서 ID 또는 후보 번호를 선택해 주세요.',candidates=in_scope,reason='ambiguous_named_document')
            if not preserve_comparison:
                if basis=='title' and len(in_scope)==1 and len(self._active)>1 and in_scope[0] in self._active and not switching:
                    named_focus=in_scope
                elif basis=='title' or not self._active or not set(self._active).issubset(in_scope):self.select_documents(in_scope)
        elif explicit is None and switching:
            return self._reply('needs_clarification',query,'전환할 사업명 또는 문서 ID를 지정해 주세요.',candidates=eligible,reason='switch_target_missing')
        if not self._active:
            return self._reply('needs_clarification',query,'먼저 질문할 사업명이나 문서를 선택해 주세요.',candidates=eligible,reason='document_not_selected')
        if switching and not re.search(r'예산|기간|조건|자격|평가|제출|금액|세금|부가세|얼마|몇|알려|요약|설명|[?？]',intent):
            self._pending=None
            return {'status':'selected','execution_status':'not_requested','answer_generated':False,
                'query':query,'standalone_query':None,'active_doc_ids':self.active_doc_ids,
                'effective_doc_ids':self.active_doc_ids,'filters':self.filters,'claims':[],
                'markdown':'선택 문서를 변경했습니다. 이어서 질문해 주세요.',
                'routing_api_calls':0,'query_rewrite_uses_assistant_answers':False}
        comparison=bool(re.search(r'비교|차이|두\s*(?:사업|문서|건)',intent))
        if comparison and len(self._active)<2:
            recent=[d for d in self._recent if d in eligible]
            if len(recent)>=2 and explicit is None and not named:
                if len(recent)>2 and not re.search(r'최근|방금',intent):
                    return self._reply('needs_clarification',query,'최근 문서 중 비교할 두 문서를 선택해 주세요.',
                        candidates=recent,reason='recent_pair_ambiguous')
                self.select_documents(recent[-2:])
            else:
                return self._reply('needs_clarification',query,'비교할 문서를 두 개 이상 선택해 주세요.',candidates=eligible,reason='comparison_needs_multiple_documents')
        scope=list(named_focus or self._active);plural=bool(_PLURAL.search(intent))
        first=bool(_FIRST.search(intent));second=bool(_SECOND.search(intent))
        if len(scope)>1 and first and not second:scope=[scope[0]]
        elif len(scope)>1 and second and not first:scope=[scope[1]]
        elif len(scope)>1 and _SINGULAR.search(intent) and (not plural or re.search(r'중.{0,15}그\s*사업',intent)):
            if len(self._focus)==1:scope=list(self._focus)
            else:return self._reply('needs_clarification',query,'어느 문서를 가리키는지 선택해 주세요.',candidates=scope,reason='ambiguous_singular_reference')
        facet=next((name for name,pattern in _FACETS if re.search(pattern,intent,re.I)),None)
        inherited_facet=None
        if facet is None and switching and _SINGULAR.search(intent) and prior_facet:
            facet=inherited_facet=prior_facet
        previous=next((h for h in reversed(self._history) if h['epoch']==self._epoch and h['doc_ids']==scope),None)
        parts=['질문: '+query]
        titles=[self._catalog[d]['project_title'] for d in scope]
        title_text='; '.join(titles)
        if len(title_text)>650:title_text=f'선택한 {len(scope)}개 문서 (문서 ID 범위는 검색 필터로 고정)'
        parts.append('대상 사업: '+title_text)
        content_question=query
        if inherited_facet:
            # Preserve only the user's requested field across an explicit
            # switch, never the former business name or its evidence.
            label=_FACET_LABELS[inherited_facet]
            parts.append('이어 묻는 항목: '+label)
            content_question+=' '+label
        # Topic context is a bounded ORIGINAL user question, never a rewritten
        # query or an answer. A budget question cannot contaminate a period turn.
        previous_question=None
        if previous and (facet is None or facet==previous.get('facet')) and (_SINGULAR.search(query) or len(query)<35):
            previous_question=previous['user'][:self.config.previous_question_chars]
            parts.append('같은 문서의 직전 사용자 질문: '+previous_question)
        standalone='\n'.join(parts)
        if len(standalone)>self.config.max_rewrite_chars:
            return self._reply('needs_clarification',query,'질문을 조금 줄여 주세요.',candidates=scope,reason='rewrite_limit')
        self._focus=list(scope);self._pending=None;self._turn+=1
        self._history.append({'user':query,'doc_ids':scope,'facet':facet,'epoch':self._epoch})
        self._history=self._history[-self.config.max_history_turns:]
        return {'status':'ready','query':query,'standalone_query':standalone,
            'retrieval_plan':plan_content_queries(content_question,titles,previous_question=previous_question,
                agencies=[self._catalog[d]['agency'] for d in scope]),
            'inherited_facet':inherited_facet,
            'active_doc_ids':self.active_doc_ids,'effective_doc_ids':scope,'filters':self.filters,
            'feature':'comparison' if len(scope)>1 else 'followup' if previous else 'qa',
            'facet':facet,'turn':self._turn,'epoch':self._epoch,'routing_api_calls':0,
            'query_rewrite_uses_assistant_answers':False,'candidate_documents':[]}

    def ask(self,query,*,doc_ids=_UNSET,filters=_UNSET):
        route=self.plan(query,doc_ids=doc_ids,filters=filters)
        if route['status']!='ready':return route
        if self.backend is None:
            return {**route,'status':'planned','execution_status':'not_requested',
                'answer_generated':False,'claims':[],'markdown':'문서 범위와 검색 질문을 준비했습니다.'}
        try:
            answer_plan=getattr(self.backend,'answer_plan',None)
            # Structured planners can separate model context from lexical
            # queries. Existing simple injected backends keep their interface.
            if callable(answer_plan):result=answer_plan(copy.deepcopy(route))
            else:result=self.backend(route['standalone_query'],doc_ids=route['effective_doc_ids'],feature=route['feature'])
            if not isinstance(result,dict):raise ValueError('backend_result_invalid')
            allowed=set(route['effective_doc_ids'])
            if set(result.get('active_doc_ids',[]))!=allowed:raise ValueError('backend_document_scope_changed')
            if any(c.get('doc_id') not in allowed or any(q.get('doc_id') not in allowed for q in c.get('citations',[])) for c in result.get('claims',[])):
                raise ValueError('backend_claim_outside_scope')
            if any(c.get('doc_id') not in allowed for c in result.get('source_chunks',[])):
                raise ValueError('backend_source_outside_scope')
            if result.get('answer_generated') and (not result.get('claims') or not (result.get('validation') or {}).get('valid')):
                raise ValueError('backend_answer_not_source_validated')
        except Exception:
            return {**route,'status':'not_generated','execution_status':'error',
                'answer_generated':False,'claims':[],'markdown':'선택한 문서 범위에서 처리하지 못했습니다.',
                'error_code':'conversation_backend_failed_closed'}
        return {**result,'conversation':route,'active_doc_ids':route['active_doc_ids'],
            'effective_doc_ids':route['effective_doc_ids'],'filters':route['filters']}

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(self.state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temporary.replace(path)
        return path

    def restore(self,path):
        value=json.loads(Path(path).read_text(encoding='utf-8'))
        legacy=value.get('version')=='rfp-conversation-v1'
        if value.get('version') not in {VERSION,'rfp-conversation-v1'} or value.get('catalog_fingerprint')!=self.catalog_fingerprint:
            raise ValueError('대화 상태의 원문·문서 버전이 현재 자료와 다릅니다.')
        allowed={'version','catalog_fingerprint','active_doc_ids','focus_doc_ids','filters','history','epoch','turn','pending'}
        if not legacy and 'recent_doc_ids' not in value:raise ValueError('최근 문서 이력이 누락됐습니다.')
        if set(value) not in (allowed,allowed|{'recent_doc_ids'}):raise ValueError('지원하지 않는 대화 상태 형식입니다.')
        filters=self._validate_filters(value['filters'])
        active=self._resolve(value['active_doc_ids']) if value['active_doc_ids'] else []
        focus=self._resolve(value['focus_doc_ids']) if value['focus_doc_ids'] else []
        recent=value.get('recent_doc_ids',active[-max(2,self.config.max_history_turns):])
        if (not isinstance(recent,list) or len(recent)>max(2,self.config.max_history_turns)
            or len(set(recent))!=len(recent)):
            raise ValueError('최근 문서 이력 형식이 잘못됐습니다.')
        recent=self._resolve(recent) if recent else []
        if not legacy and recent!=value.get('recent_doc_ids',[]):raise ValueError('최근 문서는 대표 ID여야 합니다.')
        if any(not self._matches_filter(d,filters) for d in recent):raise ValueError('최근 문서가 현재 필터 밖에 있습니다.')
        if not set(focus).issubset(active) or any(not self._matches_filter(d,filters) for d in active):raise ValueError('대화 상태의 문서 범위가 다릅니다.')
        if type(value['epoch'])is not int or value['epoch']<0 or type(value['turn'])is not int or value['turn']<0:raise ValueError('대화 상태의 순서가 잘못됐습니다.')
        history=value['history']
        if not isinstance(history,list) or len(history)>self.config.max_history_turns:raise ValueError('대화 이력 상한을 넘었습니다.')
        for h in history:
            if set(h)!={'user','doc_ids','facet','epoch'} or not isinstance(h['user'],str) or len(h['user'])>self.config.max_question_chars:
                raise ValueError('사용자 질문 이력 형식이 잘못됐습니다.')
            if h['epoch']!=value['epoch'] or not set(self._resolve(h['doc_ids'])).issubset(active):raise ValueError('대화 이력에 다른 범위가 포함됐습니다.')
            if not isinstance(h['doc_ids'],list) or h['facet'] not in {None,*[name for name,_ in _FACETS]}:raise ValueError('대화 이력의 범주가 잘못됐습니다.')
        pending=value['pending']
        if pending is not None:
            if not isinstance(pending,dict) or set(pending)!={'query','candidate_doc_ids','epoch'} or not isinstance(pending['query'],str) or len(pending['query'])>self.config.max_question_chars or not isinstance(pending['candidate_doc_ids'],list):raise ValueError('확인 질문 형식이 잘못됐습니다.')
            if pending['candidate_doc_ids']:self._resolve(pending['candidate_doc_ids'])
            if pending['epoch']!=value['epoch']:raise ValueError('확인 질문 버전이 다릅니다.')
        self._active=active;self._focus=focus;self._filters=filters;self._recent=recent
        self._history=copy.deepcopy(history);self._epoch=value['epoch'];self._turn=value['turn'];self._pending=copy.deepcopy(pending)
        return self.state
