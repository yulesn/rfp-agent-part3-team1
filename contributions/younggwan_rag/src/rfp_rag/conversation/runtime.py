"""Wire conversation plans to the existing source-validated QA implementation."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

from rfp_rag.generation.llm import LLMConfig, OpenAIGroundedLLM
from rfp_rag.generation.rag import AnswerSession
from .session import ConversationSession, ConversationConfig
from .query import prioritize_content_candidates, query_topics
from .focus import focused_question, response_focus_audit
from .literal_quotes import LiteralQuoteClient, VERSION as LITERAL_VERSION


class ScopedIndex:
    """Enforce a single turn's canonical scope even if a lower layer changes it."""
    def __init__(self,index,registry,doc_ids,*,search_queries=None,search_topics=(),content_question='',context_cache=None):
        self.index=index;self.registry=registry
        self.ids=registry.resolve_doc_ids(doc_ids)
        if not self.ids:raise ValueError('범위 없는 검색은 허용하지 않습니다.')
        if search_queries is not None and (not isinstance(search_queries,list)
            or len(search_queries)>7 or any(not isinstance(q,str) or not q.strip() for q in search_queries)):
            raise ValueError('본문 검색질문 목록이 올바르지 않습니다.')
        self.search_queries=copy.deepcopy(search_queries)
        self.search_topics=tuple(search_topics);self.content_question=content_question
        self.context_cache=context_cache if context_cache is not None else {}

    def _source_context(self,chunk,start,end):
        from rfp_rag.summary_content.context import DocumentContext
        did=chunk['doc_id'];key=('chunk_range',chunk['chunk_id'],start,end)
        if key not in self.context_cache:
            if did not in self.context_cache:
                self.context_cache[did]=DocumentContext(self.registry.documents_by_id[did])
            ranges=[]
            for span in chunk['source_spans']:
                left,right=max(start,span['chunk_start']),min(end,span['chunk_end'])
                if left<right:
                    ranges.append({**span,
                        'char_start':span['char_start']+left-span['chunk_start'],
                        'char_end':span['char_start']+right-span['chunk_start']})
            value=self.context_cache[did].classify_ranges(ranges,quote=chunk['text'][start:end])
            self.context_cache[key]={'evidence_role':value['evidence_role'],'event_scope':value['event_scope']}
        return self.context_cache[key]

    def resolve_doc_ids(self,ids):
        chosen=self.registry.resolve_doc_ids(ids)
        if not chosen or not set(chosen).issubset(self.ids):raise ValueError('선택 문서 밖으로 범위를 넓힐 수 없습니다.')
        return chosen

    def get_catalog(self):return [copy.deepcopy(r) for r in self.index.get_catalog() if r['doc_id'] in self.ids]

    def search(self,query,**kwargs):
        requested=kwargs.pop('allowed_doc_ids',None)
        selected=self.ids if requested is None else self.resolve_doc_ids(requested)
        queries=[query] if self.search_queries is None else self.search_queries
        groups=[]
        for content_query in queries:
            options=dict(kwargs)
            if self.search_queries is not None:
                # Rank a bounded candidate pool before the final top_k. This
                # does not enlarge the source text sent to the model.
                options['top_k']=max(options.get('top_k',5),options.get('fetch_k',40))
            hits=self.index.search(content_query,allowed_doc_ids=selected,**options)
            # Check every returned candidate before merging or truncation.
            self.registry.validate_chunks(hits,allowed_doc_ids=selected)
            # Each topic query keeps its own priority before round-robin.
            # A period request must not push budget clauses out of a dedicated
            # budget search in the same turn.
            topics=[t for t in query_topics(content_query) if t in self.search_topics]
            hits=prioritize_content_candidates(hits,topics=topics,
                content_question=self.content_question,source_context=self._source_context)
            groups.append(hits)
        merged=[];seen=set();limit=kwargs.get('top_k',5)
        for offset in range(max((len(h) for h in groups),default=0)):
            for hits in groups:
                if offset>=len(hits):continue
                chunk=hits[offset]
                if chunk['chunk_id'] not in seen:
                    seen.add(chunk['chunk_id']);merged.append(chunk)
        return merged[:limit]


class _FixedSelection:
    """Disable lower-layer title selection/rewrite after Phase6 resolves it."""
    def __init__(self,ids,query):self.active_doc_ids=list(ids);self.query_history=[];self.query=query

    def select_documents(self,ids):
        if list(ids)!=self.active_doc_ids:raise ValueError('턴 중 문서 선택 변경을 거절했습니다.')
        return list(self.active_doc_ids)

    def ask(self,query,**kwargs):
        return {'status':'retrieved','standalone_query':self.query,'query':self.query,
            'active_doc_ids':list(self.active_doc_ids),'hits':[],'query_history':[],
            'selection_reason':'phase6_fixed_scope','scope_preflight_only':True}


class GroundedBackend:
    def __init__(self,env,llm,retrieval=None,*,approved_doc_ids=None):
        self.env=env;self.llm=llm
        self.retrieval=dict(retrieval or {});self.retrieval.setdefault('method','rrf')
        self.approved=None if approved_doc_ids is None else set(approved_doc_ids)
        self._context_cache={}

    def answer_plan(self,route):
        question=focused_question(route['standalone_query'],route['retrieval_plan']['topics'])
        result=self(question,doc_ids=route['effective_doc_ids'],
            feature=route['feature'],search_queries=route['retrieval_plan']['queries'],
            search_topics=route['retrieval_plan']['topics'],
            content_question=route['retrieval_plan']['content_question'],request_question=route['query'])
        result.setdefault('retrieval',{})['content_query_plan']=copy.deepcopy(route['retrieval_plan'])
        result['retrieval']['candidate_selection']='content_priority_then_query_round_robin'
        result['retrieval']['ranking_is_semantic_verdict']=False
        result['answer_focus_audit']=response_focus_audit(result,route['retrieval_plan']['topics'])
        return result

    def __call__(self,query,*,doc_ids,feature,search_queries=None,search_topics=(),content_question='',request_question=None):
        if self.approved is not None and not set(doc_ids).issubset(self.approved):
            return {'status':'not_generated','execution_status':'not_approved',
                'answer_generated':False,'claims':[],'active_doc_ids':list(doc_ids),
                'markdown':'선택한 문서를 API 허용 목록에 추가한 뒤 실행해 주세요.',
                'source_chunks':[]}
        if len(query)>self.retrieval.get('max_question_chars',4000):
            raise ValueError('검색질문이 설정 상한을 넘었습니다.')
        if search_queries is not None and any(len(q)>self.retrieval.get('max_question_chars',4000) for q in search_queries):
            raise ValueError('본문 검색질문이 설정 상한을 넘었습니다.')
        index=ScopedIndex(self.env['index'],self.env['registry'],doc_ids,search_queries=search_queries,
            search_topics=search_topics,content_question=content_question,context_cache=self._context_cache)
        retrieval=dict(self.retrieval)
        literal='work_scope' in search_topics
        llm=LiteralQuoteClient(self.llm) if literal else self.llm
        if literal:
            # Source text appears in both evidence and the allowed-quote schema.
            # Bound it before normal document quotas/packing, not after dispatch.
            request_cap=getattr(self.llm.config,'max_request_bytes',60000)
            retrieval['max_context_bytes']=min(retrieval.get('max_context_bytes',42000),max(1,request_cap//3))
        answer=AnswerSession(index,llm,self.env['registry'],config=retrieval,
            active_doc_ids=index.ids)
        answer.retrieval_session=_FixedSelection(index.ids,query)
        # Intent was resolved from the current user turn, excluding titles.
        # Generation keeps the original question and business context. The
        # scoped index uses separate content queries for lexical ranking.
        result=answer.ask('비교 요청' if feature=='comparison' else '문서 질문',feature=feature,
            request_question=request_question or content_question or query)
        if literal:
            result.setdefault('retrieval',{})['literal_citation_contract']={
                'version':LITERAL_VERSION,'context_byte_limit':retrieval['max_context_bytes'],
                'request_schema_and_evidence_share_existing_cap':True}
        if result.get('execution_status')=='disabled':
            result['status']='retrieved' if result.get('source_chunks') else 'insufficient_evidence'
            result['markdown']='선택한 문서에서 원문 근거 후보를 찾았습니다. 실제 답변 생성은 꺼져 있습니다.'
        return result


def make_conversation(env,*,enabled=False,approved_doc_ids=None,client=None,
                      config=None,retrieval=None,generation=None):
    """No SDK/key/network use until an explicitly enabled turn calls generate.

    One LLM instance is retained for the entire conversation, so usage limits
    are cumulative across turns and scope switches rather than reset per call.
    """
    if type(enabled)is not bool:raise TypeError('enabled는 True/False여야 합니다.')
    registry=env['registry']
    approved=registry.resolve_doc_ids(approved_doc_ids) if approved_doc_ids else []
    if enabled and not approved:raise ValueError('API에 전송할 문서 ID를 명시적으로 허용하세요.')
    settings=dict(generation or env.get('phase5_config',{}).get('generation',{}))
    settings.pop('enabled',None)
    llm_config=replace(LLMConfig(**settings),enabled=enabled)
    allowed={did:registry.source_hashes[did] for did in approved}
    llm=OpenAIGroundedLLM(Path(env['root'])/'phase6_outputs/generation_cache',
        config=llm_config,allowed_sources=allowed,client=client)
    backend=GroundedBackend(env,llm,retrieval=retrieval,approved_doc_ids=approved if enabled else None)
    conversation=ConversationSession(env['documents'],backend=backend,
        config=ConversationConfig(**config) if isinstance(config,dict) else config)
    conversation.llm=llm  # Usage inspection only; never used for query rewriting.
    return conversation
