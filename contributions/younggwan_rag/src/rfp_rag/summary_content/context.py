"""Conservative, source-positioned context for RFP facts.

This module does not decide whether a claim is true and never rewrites a
citation.  It supplies the surrounding source headings/lines needed to keep a
briefing application, a tender submission and a post-contract deliverable
separate.  Rules depend on Korean document structure, not document IDs,
filenames, sample answers or particular project titles.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from difflib import SequenceMatcher
import re

VERSION = 'rfp-source-context-v1'
EVENT_SCOPES = ('tender', 'briefing', 'execution', 'subcontract', 'template', 'unknown')
EVIDENCE_ROLES = ('project_fact', 'source_form', 'example', 'agreement_clause',
                  'other_project_candidate', 'unknown')


def _compact(value):
    return re.sub(r'\s+', '', str(value))


_FORM_MARKER = re.compile(
    r'^(?:[<\[〈【]\s*(?:서\s*식|별\s*지)[^>\]〉】]*[>\]〉】]'
    r'|서\s*식\s*제?\s*\d+\s*호|\[?별지\s*제\s*\d+호서식'
    r'|[<\[〈【]\s*(?:붙임|별첨)[^>\]〉】]*[>\]〉】].{0,90}'
    r'(?:신청서|계획서|검토결과서|서약서|확약서|산출내역서)'
    r'|\[?소프트웨어\s*사업\s*영향평가\s*검토결과서\]?)')
_EXAMPLE = re.compile(r'(?:\(\s*예시\s*\)|\[\s*예시\s*\]|작성\s*예시|단순\s*예시)')
_AGREEMENT = re.compile(
    r'^(?:[<\[〈【].{0,25}[>\]〉】]\s*)?(?:비밀\s*유지\s*계약서|'
    r'개인(?:\(신용\))?\s*정보.{0,20}(?:약정서|위탁계약서)|'
    r'청렴\s*계약.{0,20}(?:서약서|이행각서)|공동\s*수급.{0,12}협정서)')
_PROJECT_LABEL = re.compile(
    r'^\s*(?:\d+[.)]?\s*)?[□ㅇ○▪●]?\s*\(?\s*'
    r'((?:사\s*업|과\s*업|용\s*역)\s*명)\s*'
    r'(?:\(\s*또는\s*(?:서비스명|과업명)\s*\))?\s*\)?\s*[:：]?\s*(.*?)\s*$')
_TENDER = re.compile(
    r'(?:입찰\s*(?:참가|등록|공고|마감|서류|접수|제출)|제안서\s*(?:및\s*가격입찰서\s*)?'
    r'(?:제출|접수)|가격\s*(?:입찰|투찰)|낙찰자\s*선정|협상적격자|나라장터|전자입찰)')
_BRIEFING = re.compile(r'(?:사전\s*설명회|사업\s*설명회|용역\s*설명회|현장\s*설명회)')
_EXECUTION = re.compile(
    r'(?:착수\s*(?:계|시|및|보고|후)|준공\s*(?:계|시|검사|완료|정산)|'
    r'산출물\s*관리|성과품\s*제출|용역\s*종료\s*시|사업\s*수행\s*결과물|'
    r'계약\s*(?:시작|개시)\s*일로부터|정기\s*보고서|완료\s*보고서|결과\s*보고서)')
_SUBCONTRACT = re.compile(r'(?:하도급|재하도급).{0,25}(?:승인|신청서|계획서|계약명|자기평가|적정성)')
_MAJOR = re.compile(r'^(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+[.．]?|[IVX]{1,6}[.．]|\d{1,2}[.．])\s*\S')
_HEADING_END = re.compile(
    r'(?:개요|현황|목적|범위|안내|유의사항|참가신청|일반사항|제출문서|제출서류|'
    r'구비서류|작성방법|작성요령|작성지침|평가기준|평가방법|선정방법|요구사항|'
    r'추진일정|수행일정|추진계획|계약기간|처리기간|기본정보|참가자격|접수방법|제출방식)\s*$')
_PLACEHOLDER = re.compile(r'(?:_{3,}|[○ㅇXx]{3,}|년\s*월\s*일|^사업명$|^과업명$|^용역명$)')
_OTHER_LABELS = {
    '발주기관', '사업기간', '과업기간', '사업목적', '사업범위', '계약번호', '계약금액',
    '계약년월일', '작성일', '신청내용', '사업자', '원도급자', '하도급자', '대표자',
}
_BUSINESS_SECTION = re.compile(
    r'^[□ㅇ○▪●]?(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVX]+[.．]?|\d+[.．])?'
    r'(?:사업개요|과업개요|사업안내|입찰안내|주요사업범위|제안요청내용|제안요청사항|'
    r'사업자선정|입찰참가자격|입찰관련사항|제안서접수및작성|제안서제출방법및유의사항)$')
_EXPLICIT_TENDER_SUBMISSION = re.compile(
    r'(?:(?:입찰|제안서|가격제안서|입찰서류).{0,15}(?:제출|접수|마감|가격입찰)|'
    r'입찰공고문.{0,5}(?:참조|의함)|(?:제출기한|제출일시).{0,15}입찰공고)')


@dataclass(frozen=True)
class _Line:
    index: int
    collection: str
    block_id: str
    locator: str
    start: int
    end: int
    text: str
    table_cell: bool = False

    def source(self):
        return {'collection': self.collection, 'block_id': self.block_id,
                'locator': self.locator, 'char_start': self.start, 'char_end': self.end,
                'text': self.text}


def _is_form_marker(line):
    # A form reference in a tender checklist is not the start of the form.
    # HWPX preserves table-cell metadata; a plural/range reference is also
    # recognizable when a PDF extractor flattened away that table metadata.
    return (not line.table_cell and bool(_FORM_MARKER.search(line.text))
            and not re.search(r'서\s*식[^>\]〉】]{0,24}(?:,|~|∼|및)',line.text))


def _is_heading(text):
    text = text.strip()
    if not text or len(text) > 160:
        return False
    if _FORM_MARKER.search(text) or _AGREEMENT.search(text):
        return True
    compact = _compact(text)
    if compact in {'사업개요', '과업개요', '사업안내', '사업자선정', '서비스현황',
                    '기본정보', '참가신청', '입찰참가자격', '제안서제출및제안발표회'}:
        return True
    # A sentence mentioning a requirement is not a section heading.
    if re.search(r'(?:한다|해야함|하여야함|있음|됩니다|합니다|하였다)[.!。]?$', compact):
        return False
    return bool((_MAJOR.match(text) or text.startswith(('□', '￭', '■')))
                and (_HEADING_END.search(text) or _BRIEFING.search(text)))


def _scope(text):
    """Return an explicit local event signal; ordering prevents briefing leakage."""
    compact = _compact(text)
    if _BRIEFING.search(text):
        return 'briefing'
    if _SUBCONTRACT.search(text):
        return 'subcontract'
    if _EXECUTION.search(text):
        return 'execution'
    if _TENDER.search(text):
        return 'tender'
    return 'unknown'


def _valid_project_value(text):
    compact = _compact(text).strip(':：()')
    return (len(re.findall('[가-힣A-Za-z]', compact)) >= 5
            and compact not in _OTHER_LABELS and not _PLACEHOLDER.search(compact)
            and not compact.startswith(('또는', '제안요청서상의', '입찰공고문'))
            and '서식' not in compact and len(text) < 240)


class DocumentContext:
    """Read-only line index for repeated classification in one immutable source."""

    def __init__(self, document):
        self.document_id = document.get('canonical_doc_id', document.get('doc_id'))
        self.source_sha256 = document.get('sha256')
        self.lines = []
        self._by_block = {}
        self._text_by_block = {}
        for collection in ('blocks', 'supplemental_blocks'):
            for block in document.get(collection, []):
                bid, text = block.get('block_id'), block.get('text')
                if not isinstance(bid, str) or not isinstance(text, str):
                    continue
                key = collection, bid
                if key in self._by_block:
                    raise ValueError('원문 블록 ID가 중복됐습니다.')
                self._text_by_block[key] = text
                indexes = []
                cursor = 0
                for value in text.splitlines(keepends=True) or ['']:
                    clean = value.rstrip('\r\n')
                    left = len(clean)-len(clean.lstrip())
                    right = len(clean.rstrip())
                    if right > left:
                        line = _Line(len(self.lines), collection, bid,
                                     str(block.get('locator', bid)), cursor+left,
                                     cursor+right, clean[left:right],
                                     bool(block.get('table_id')))
                        self.lines.append(line)
                        indexes.append(line.index)
                    cursor += len(value)
                self._by_block[key] = indexes
        self._headings = [line.index for line in self.lines if _is_heading(line.text)]
        self._forms = [line.index for line in self.lines if _is_form_marker(line)]
        self._agreements = [line.index for line in self.lines
                            if _AGREEMENT.search(line.text) and self._is_actual_agreement(line.index)]
        self._examples = [line.index for line in self.lines if _EXAMPLE.search(line.text)]
        self._business_boundaries = [line.index for line in self.lines
                                    if _BUSINESS_SECTION.fullmatch(_compact(line.text))]
        self.project_candidates = self._project_candidates()
        intro = [c for c in self.project_candidates if c['source_line_index'] <
                 (self._forms[0] if self._forms else len(self.lines))]
        self.primary_project_candidates = intro[:3]
        self._primary = intro[0] if intro else None

    def _is_actual_agreement(self, at):
        # A registration checklist can name a required NDA or pledge. Its name
        # alone does not place the following tender pages inside that agreement.
        before=self.lines[max(0,at-2):at]
        after=self.lines[at+1:min(len(self.lines),at+9)]
        return (any(_is_form_marker(line) for line in before)
                or any(re.search(r'(?:제\s*1\s*조|본\s*(?:계약|약정|각서)|당사는|위탁자|수탁자)',line.text)
                       for line in after))

    def _project_candidates(self):
        result = []
        for line in self.lines:
            match = _PROJECT_LABEL.match(line.text)
            if not match:
                continue
            value = match.group(2).strip()
            value_line = line
            if not value:
                for following in self.lines[line.index+1:line.index+5]:
                    if _compact(following.text).startswith(('(또는', '또는')):
                        continue
                    if _valid_project_value(following.text):
                        value, value_line = following.text, following
                    break
            if _valid_project_value(value):
                result.append({'value': value, 'normalized_value': _compact(value),
                               'source_line_index': value_line.index,
                               'source': value_line.source(), 'label_source': line.source()})
        return result

    @staticmethod
    def _last(indexes, at):
        n = bisect_right(indexes, at)-1
        return indexes[n] if n >= 0 else None

    def _range_lines(self, ranges):
        indexes = set()
        invalid = []
        for span in ranges:
            if not isinstance(span, dict):
                invalid.append('invalid_source_range'); continue
            key = span.get('collection', 'blocks'), span.get('block_id')
            text = self._text_by_block.get(key)
            left, right = span.get('char_start'), span.get('char_end')
            if (text is None or type(left) is not int or type(right) is not int
                    or left < 0 or right <= left or right > len(text)):
                invalid.append('source_range_not_resolved'); continue
            indexes.update(i for i in self._by_block[key]
                           if max(left,self.lines[i].start) < min(right,self.lines[i].end))
        return sorted(indexes), sorted(set(invalid))

    def _line_context(self, at):
        line = self.lines[at]
        # Heading signals stop at an intervening new section. Minor table labels
        # inherit the enclosing section instead of becoming a new global event.
        heading_candidates = [i for i in self._headings if max(0,at-160) <= i <= at]
        enclosing = []
        for i in reversed(heading_candidates):
            h = self.lines[i]
            enclosing.append(h)
            if _scope(h.text) != 'unknown':
                break
            if _MAJOR.match(h.text) or _compact(h.text) in {'서비스현황','사업개요','과업개요','사업안내'}:
                break
        local_scope = _scope(line.text)
        heading_scope = next((_scope(h.text) for h in enclosing if _scope(h.text)!='unknown'),'unknown')
        # A briefing section can contain "본 입찰" and participation sentences.
        explicit_tender = bool(_EXPLICIT_TENDER_SUBMISSION.search(_compact(line.text)))
        event = ('tender' if explicit_tender and not _BRIEFING.search(line.text)
                 else 'briefing' if heading_scope == 'briefing'
                 else local_scope if local_scope!='unknown' else heading_scope)
        form_at = self._last(self._forms,at)
        agreement_at = self._last(self._agreements,at)
        example_at = self._last(self._examples,at)
        boundary_at = self._last(self._business_boundaries,at)
        if boundary_at is not None:
            form_at = form_at if form_at is not None and form_at >= boundary_at else None
            agreement_at = agreement_at if agreement_at is not None and agreement_at >= boundary_at else None
            example_at = example_at if example_at is not None and example_at >= boundary_at else None
        latest_marker = max([x for x in (form_at,agreement_at) if x is not None],default=None)
        role = 'project_fact' if event != 'unknown' else 'unknown'
        signals = []
        trace = [line]
        if form_at is not None:
            role='source_form'; signals.append({'code':'form_heading','source':self.lines[form_at].source()})
            trace.append(self.lines[form_at])
        if agreement_at is not None and (form_at is None or agreement_at >= form_at):
            role='agreement_clause'; signals.append({'code':'agreement_heading','source':self.lines[agreement_at].source()})
            trace.append(self.lines[agreement_at])
        # An example marker applies until a new form, major section or later
        # explicit heading. A title labelled example remains a form-local role.
        if example_at is not None and (latest_marker is None or example_at >= latest_marker):
            between = [self.lines[h] for h in self._headings if example_at < h <= at]
            binding_bullet = any(
                re.match(r'^(?:[oㅇ○□●▪￭]|[-–])\s*',candidate.text)
                and re.search(r'(?:하여야|해야|공급자는|제안사는|제안업체는|수행하여|준수하여|필수)',candidate.text)
                for candidate in self.lines[example_at+1:at+1])
            if not binding_bullet and not any(_is_form_marker(h) or _MAJOR.match(h.text)
                    or re.match(r'^[가-하][.．)]',h.text) for h in between):
                role='example'; signals.append({'code':'example_heading','source':self.lines[example_at].source()})
                trace.append(self.lines[example_at])
        local_candidates=[p for p in self.project_candidates if p['source_line_index'] <= at]
        candidate=local_candidates[-1] if local_candidates else None
        warnings=[]
        if self._primary and candidate and at-candidate['source_line_index'] <= 160 and role in {'source_form','example','agreement_clause'}:
            similarity=SequenceMatcher(None,self._primary['normalized_value'],candidate['normalized_value']).ratio()
            if similarity < 0.55:
                role='other_project_candidate'; warnings.append('different_project_name_in_form')
                signals.append({'code':'different_project_name_in_form','source':candidate['source'],
                                'primary_source':self._primary['source']})
        if role in {'source_form','example','agreement_clause','other_project_candidate'} and event=='unknown':
            event='template'
        trace.extend(enclosing)
        if event!='unknown':
            source=line if local_scope!='unknown' and event==local_scope else next((h for h in enclosing if _scope(h.text)==event),line)
            signals.append({'code':'event_'+event,'source':source.source()})
        if _PLACEHOLDER.search(line.text):
            warnings.append('placeholder_or_unfilled_form_text')
        return {'event_scope':event,'evidence_role':role,
                'certainty':'explicit' if local_scope!='unknown' else 'contextual' if event!='unknown' or role!='unknown' else 'unknown',
                'warnings':warnings,'signals':signals,'trace':trace,
                'headings':[h.source() for h in enclosing]}

    def classify_ranges(self, source_ranges, quote=''):
        indexes, warnings = self._range_lines(source_ranges)
        if not indexes:
            return {'version':VERSION,'event_scope':'unknown','evidence_role':'unknown',
                    'certainty':'unknown','warnings':warnings or ['source_ranges_missing'],
                    'signals':[],'context_source_ranges':[],'heading_path':[],
                    'primary_project_candidates':self.primary_project_candidates,
                    'trusted_for_tender':False}
        contexts=[self._line_context(i) for i in indexes]
        scopes={c['event_scope'] for c in contexts if c['event_scope']!='unknown'}
        if len(scopes)>1:
            warnings.append('mixed_event_scopes')
        event=next(iter(scopes)) if len(scopes)==1 else 'unknown'
        roles={c['evidence_role'] for c in contexts if c['evidence_role']!='unknown'}
        # A single inappropriate source must not be hidden in a composite claim.
        order=('other_project_candidate','example','agreement_clause','source_form','project_fact')
        role=next((r for r in order if r in roles),'unknown')
        if len(roles)>1:
            warnings.append('mixed_evidence_roles')
        signals=[];trace={};headings={}
        for context in contexts:
            warnings.extend(context['warnings'])
            for signal in context['signals']:
                if signal not in signals:signals.append(signal)
            for line in context['trace']:
                trace[line.index]=line.source()
            for h in context['headings']:
                headings[(h['collection'],h['block_id'],h['char_start'])]=h
        # Include immediate neighboring lines for inspection, with exact source
        # offsets even when the extractor represented a PDF page as one block.
        for i in indexes:
            for line in self.lines[max(0,i-1):min(len(self.lines),i+2)]:
                trace[line.index]=line.source()
        return {'version':VERSION,'event_scope':event,'evidence_role':role,
                'certainty':'explicit' if all(c['certainty']=='explicit' for c in contexts) else 'contextual' if event!='unknown' or role!='unknown' else 'unknown',
                'warnings':sorted(set(warnings)),'signals':signals,
                'context_source_ranges':[trace[i] for i in sorted(trace)],
                'heading_path':list(headings.values()),
                'primary_project_candidates':self.primary_project_candidates,
                'trusted_for_tender': event=='tender' and role=='project_fact' and not warnings}

    def classify_claim(self, claim):
        citations=claim.get('citations',[]) if isinstance(claim,dict) else []
        ranges=[s for c in citations if isinstance(c,dict) for s in c.get('source_ranges',[])]
        quote='\n'.join(c.get('quote','') for c in citations if isinstance(c,dict))
        return self.classify_ranges(ranges,quote=quote)


def identify_project_candidates(document):
    """Return intro project values grounded in source labels, never filenames."""
    return DocumentContext(document).primary_project_candidates


def classify_claim_context(document, claim):
    return DocumentContext(document).classify_claim(claim)


def classify_source_context(document, source_ranges, quote=''):
    return DocumentContext(document).classify_ranges(source_ranges,quote=quote)
