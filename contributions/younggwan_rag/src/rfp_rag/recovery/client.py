"""Preserve rejected response diagnostics without changing requests or validation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from rfp_rag.generation.llm import LLMError, OpenAIGroundedLLM, _field, _parse_json
from rfp_rag.summary_full.transport import ShortCitationLLM, TransportError


REVISION = 'phase5-recovery-v1'
STRUCTURE_REASONS = {
    'Model returned an invalid answer object': 'invalid_answer_keys',
    'Model returned an invalid answer status': 'invalid_answer_status',
    'Model returned invalid answer lists': 'invalid_answer_lists',
    'Model returned invalid missing-information labels': 'invalid_missing_information_labels',
    'Model answer status and claim count disagree': 'status_claim_count_mismatch',
    'Model marked incomplete information as fully answered': 'answered_with_missing',
    'Model marked an answer partial without identifying missing information': 'partial_without_missing',
    'Model returned an invalid claim object': 'invalid_claim_keys',
    'Model returned an empty claim': 'empty_claim',
    'Model returned an unsupported category or an uncited claim': 'invalid_category_or_uncited_claim',
    'Model returned an invalid citation object': 'invalid_citation_keys',
    'Model returned an empty citation': 'empty_citation',
    'API response does not contain a valid structured answer': 'invalid_structured_json',
    'The model refused this request; no factual answer was accepted': 'response_refused',
    'API response is incomplete or failed; no factual answer was accepted': 'response_not_completed',
    'API returned invalid usage counters': 'invalid_usage',
    'API returned invalid cached-input usage counters': 'invalid_usage',
}
TRANSPORT_REASONS = frozenset({
    'alias_payload_structure_invalid','document_alias_not_allowed','source_unit_alias_not_allowed',
    'source_placeholder_invalid','response_request_fingerprint_mismatch',
    'descriptor_mapping_fingerprint_mismatch','descriptor_not_bound_to_current_request',
    'descriptor_mapping_structure_invalid','descriptor_version_invalid',
    'descriptor_document_binding_mismatch','descriptor_canonical_chunk_order_mismatch',
    'descriptor_wire_schema_fingerprint_mismatch','descriptor_unit_coverage_mismatch',
})


def redact(text):
    text = text if isinstance(text,str) else ''
    key=os.environ.get('OPENAI_API_KEY','')
    if key:text=text.replace(key,'[API_KEY_REDACTED]')
    return re.sub(r'sk-[A-Za-z0-9_-]+','[API_KEY_REDACTED]',text)


class RecoveryLLM(ShortCitationLLM):
    """Original strict client plus private, credential-redacted failure records.

    Payloads are neither repaired nor silently accepted. Request bodies,
    fingerprints, schema, model and source-unit validation remain unchanged.
    """

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._active_fingerprint=None
        self._received_output=None
        self._allowed_request_fingerprints=None

    def _save_diagnostic(self,error,stage,reason):
        if reason not in set(STRUCTURE_REASONS.values())|TRANSPORT_REASONS|{'unclassified_validation_error'}:
            reason='unclassified_validation_error'
        detail={'stage':stage,'reason':reason,'diagnostic_saved':False}
        try:
            raw=self._received_output
            if raw is None and self._active_fingerprint is not None:
                raw=None  # Pre-decoding alias failures already retain their exact response in cache.
            text=redact(raw or '')
            record={'revision':REVISION,'request_fingerprint':self._active_fingerprint,
                'stage':stage,'reason':reason,'response_received':True,
                'raw_output_text':text[:2_000_000],'raw_output_truncated':len(text)>2_000_000,
                'raw_output_available':raw is not None,'accepted_as_answer':False}
            if raw is not None:
                try:
                    payload=_parse_json(raw)
                    if isinstance(payload,dict):
                        missing=payload.get('missing_information')
                        record['missing_label_count']=len(missing) if isinstance(missing,list) else None
                        record['empty_missing_label_count']=sum(isinstance(x,str) and not x.strip() for x in missing) if isinstance(missing,list) else None
                        record['claim_count']=len(payload['claims']) if isinstance(payload.get('claims'),list) else None
                except (ValueError,TypeError,RecursionError):pass
            folder=self.cache_dir/'recovery_diagnostics'
            folder.mkdir(parents=True,exist_ok=True,mode=0o700)
            path=folder/((self._active_fingerprint or 'unknown')+'_'+uuid.uuid4().hex+'.json')
            encoded=(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
            fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'wb') as handle:handle.write(encoded)
            detail.update(diagnostic_saved=True,diagnostic_file=str(path),
                          diagnostic_sha256=hashlib.sha256(encoded).hexdigest())
        except (OSError,ValueError,TypeError):
            # Failure to save a diagnostic never triggers another paid request.
            detail['diagnostic_write_failed']=True
        error.recovery_diagnostic=detail

    def _response_payload(self,response):
        value=_field(response,'output_text')
        self._received_output=value if isinstance(value,str) else None
        try:
            return OpenAIGroundedLLM._response_payload(response)
        except LLMError as error:
            message=error.args[0] if error.args and isinstance(error.args[0],str) else ''
            self._save_diagnostic(error,'response_structure',STRUCTURE_REASONS.get(message,'unclassified_validation_error'))
            raise

    def _usage(self,response):
        value=_field(response,'output_text')
        self._received_output=value if isinstance(value,str) else None
        return super()._usage(response)

    def generate(self,task,question,evidence,*,schema=None):
        prepared=self._prepare(task,question,evidence,schema)
        self._active_fingerprint=prepared['fingerprint'];self._received_output=None
        if (self._allowed_request_fingerprints is not None
                and self._active_fingerprint not in self._allowed_request_fingerprints):
            raise LLMError('Recovery request is outside the explicitly selected failed-job set')
        try:
            return super().generate(task,question,evidence,schema=schema)
        except TransportError as error:
            if self._received_output is None:
                try:
                    cached=self._cache_read(prepared)
                    if cached is not None:self._received_output=json.dumps(cached['payload'],ensure_ascii=False)
                except LLMError:pass
            self._save_diagnostic(error,'source_unit_decode',
                error.reason if error.reason in TRANSPORT_REASONS else 'unclassified_validation_error')
            raise
        except LLMError as error:
            if not hasattr(error,'recovery_diagnostic') and self._received_output is not None:
                message=error.args[0] if error.args and isinstance(error.args[0],str) else ''
                self._save_diagnostic(error,'response_structure',
                    STRUCTURE_REASONS.get(message,'unclassified_validation_error'))
            raise
