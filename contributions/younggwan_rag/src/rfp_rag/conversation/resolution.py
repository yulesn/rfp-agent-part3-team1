"""Conservative metadata aliases with occurrence-aware project disambiguation.

No document IDs, reference answers, corpus names or fuzzy nearest-title rules
are embedded here. A shared title stays ambiguous without selected scope,
agency or year information that actually distinguishes its occurrences.
"""
from __future__ import annotations

from collections import defaultdict
import re
import unicodedata

from rfp_rag.retrieval.session import _title_variants


def compact(value):
    return ''.join(c for c in unicodedata.normalize('NFC',str(value)).casefold() if c.isalnum())


def title_variants(title):
    text=unicodedata.normalize('NFC',str(title))
    # Filename suffixes are aliases only; display names and original data stay
    # unchanged. Descriptive qualifiers such as '(전면개편)' are retained.
    clean=re.sub(r'(?:[\s_]+(?:복사본|사본\d*|최종(?:본)?|수정(?:본)?))+\s*$','',text)
    values=[text,clean]
    for t in list(values):
        values.append(re.sub(r'\([A-Za-z][A-Za-z0-9 /+.-]*\)','',t))
        # A technical project name may give an English alternative in
        # parentheses. Keep the following descriptive title when using it.
        values.append(re.sub(r'[가-힣A-Za-z0-9]+\(([A-Za-z][A-Za-z0-9 /+.-]*)\)',r'\1',t))
        values.append(re.sub(r'정보화?\s*전략\s*계획\s*(?:을\s*위한\s*)?\(?\s*ISP\s*\)?','ISP',t,flags=re.I))
    values += [re.sub(r'(?<![가-힣A-Za-z])(?:사업|용역)(?=\s|$|\()','',t) for t in list(values)]
    return list(dict.fromkeys(v for t in values for v in _title_variants(t)))


class MetadataResolver:
    def __init__(self,catalog):
        self.catalog=catalog
        self.aliases={did:title_variants(row['project_title']) for did,row in catalog.items()}

    def mentions(self,query,preferred=()):
        quoted=re.findall(r'「([^」]+)」|『([^』]+)』|“([^”]+)”',query)
        if quoted:
            ids=[];ambiguous=False
            for group in quoted:
                found,basis=self._segment(''.join(group),preferred)
                ids.extend(found);ambiguous|=basis=='ambiguous_titles'
            if ids:return list(dict.fromkeys(ids)),'ambiguous_titles' if ambiguous else 'title'
        return self._segment(query,preferred)

    def _segment(self,query,preferred=()):
        normalized=compact(query);matches=[]
        for did,aliases in self.aliases.items():
            for alias in aliases:
                start=normalized.find(alias)
                while start>=0:
                    matches.append((start,start+len(alias),did,alias))
                    start=normalized.find(alias,start+1)
        # A generic ISP title inside a more specific named project does not
        # introduce a new business. Independent occurrences remain independent.
        maximal=[m for m in matches if not any(
            (other[0]<=m[0] and m[1]<=other[1] and other[1]-other[0]>m[1]-m[0])
            for other in matches)]
        grouped=defaultdict(list)
        for start,end,did,alias in maximal:grouped[(start,end)].append((did,alias))
        chosen=[];ambiguous=False
        years=set(re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)',query))
        preferred=set(preferred)
        for (start,end),group in sorted(grouped.items()):
            ids=list(dict.fromkeys(d for d,_ in group))
            agency_ids=[d for d in ids if len(compact(self.catalog[d].get('agency') or ''))>=3
                        and compact(self.catalog[d]['agency']) in normalized]
            if agency_ids:ids=agency_ids
            if len(ids)>1 and years:
                dated=[d for d in ids if str(self.catalog[d].get('document_year')) in years
                    or years.intersection(re.findall(r'(?:19|20)\d{2}',self.catalog[d]['project_title']))]
                if dated:ids=dated
            if len(ids)>1:
                selected=[d for d in ids if d in preferred]
                if selected:ids=selected
            ambiguous|=len(ids)>1
            chosen.extend(ids)
        if chosen:return list(dict.fromkeys(chosen)),'ambiguous_titles' if ambiguous else 'title'
        agencies=[did for did,row in self.catalog.items() if len(compact(row.get('agency') or ''))>=3
                  and compact(row['agency']) in normalized]
        return agencies,'agency' if agencies else None

    def mask_titles(self,query):
        text=unicodedata.normalize('NFC',query)
        # Retain word boundaries for pronouns: 전자계약 is not the pronoun 전자.
        for alias in sorted({v for values in self.aliases.values() for v in values},key=len,reverse=True):
            text=re.sub(r'[\W_]*'.join(re.escape(c) for c in alias),' ',text,flags=re.I)
        return text
