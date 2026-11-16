"""Check duration value tokens against literal, already validated quotations.

This is an additional necessary-condition check, not semantic entailment:
negation, qualifiers and whether a period belongs to the right project still
require their separate source-context checks. Never inspect whole chunk text.
"""
from __future__ import annotations

import re


_VALUE = re.compile(r'(?<![\d.,])\d[\d,]*(?:\.\d+)?(?:개월|달|일|주|년)')
_GROUPING_COMMA = re.compile(r'(?<=\d),(?=\d{3}(?:\D|$))')


def _values(text):
    compact=re.sub(r'\s+','',text)
    # Permit thousands separators without interpreting commas as decimal
    # points, changing units or equating 3.0 with 3.
    return [_GROUPING_COMMA.sub('',m.group()) for m in _VALUE.finditer(compact)]


def missing_duration_values(claim):
    """Return ordered, unique duration tokens absent from its exact quotes.

    Whitespace and numeric grouping commas are the only normalizations. Each
    quotation is tokenized independently, so separate quotations ``3`` and
    ``개월`` cannot manufacture evidence for a duration token.
    """
    if not isinstance(claim,dict) or claim.get('category')!='duration':return []
    text=claim.get('text','')
    if not isinstance(text,str):return []
    quoted=set()
    for citation in claim.get('citations',[]):
        if not isinstance(citation,dict):continue
        quote=citation.get('quote')
        if isinstance(quote,str):quoted.update(_values(quote))
    return list(dict.fromkeys(v for v in _values(text) if v not in quoted))
