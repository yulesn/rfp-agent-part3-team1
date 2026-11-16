"""Explicit boundary conversion for the team's list[Document] retrieval contract."""
from __future__ import annotations

import copy


def to_langchain_documents(chunks):
    """Preserve provenance; LangChain is needed only when calling this adapter."""
    from langchain_core.documents import Document
    return [Document(page_content=chunk['text'], metadata=copy.deepcopy({
        key: value for key, value in chunk.items() if key != 'text'})) for chunk in chunks]
