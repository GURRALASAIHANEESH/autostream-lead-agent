from __future__ import annotations

import logging
from typing import Final

from langchain_core.documents import Document

from rag.loader import get_vectorstore

_FALLBACK_RESPONSE: Final[str] = (
    "I don't have specific information about that. Let me know if you'd like "
    "to speak with our team directly."
)

_CHUNK_SEPARATOR: Final[str] = "\n\n---\n\n"


class KnowledgeRetriever:
    """Stateless retrieval facade over the AutoStream FAISS knowledge base."""

    _logger: logging.Logger = logging.getLogger(f"{__name__}.KnowledgeRetriever")

    def __init__(self, *, k: int = 3) -> None:
        self._vectorstore = get_vectorstore()
        self._k: int = k
        self._retriever = self._vectorstore.as_retriever(
            search_kwargs={"k": self._k},
        )
        self._logger.info("Initialised with k=%d", self._k)

    def retrieve(self, query: str) -> str:
        """Return top-k KB passages joined into a single context string for the LLM."""
        docs: list[Document] = self._retriever.invoke(query)

        if not docs:
            self._logger.warning(
                "Vectorstore returned zero results for query: %r", query
            )
            return _FALLBACK_RESPONSE

        self._logger.info(
            "Retrieved %d chunks for query: %r", len(docs), query[:80]
        )
        return _CHUNK_SEPARATOR.join(doc.page_content for doc in docs)

    def retrieve_with_scores(
        self,
        query: str,
        threshold: float = 0.5,
    ) -> list[tuple[Document, float]]:
        """Return documents with distance scores at or below the given threshold."""
        raw_results: list[tuple[Document, float]] = (
            self._vectorstore.similarity_search_with_score(query, k=self._k)
        )

        filtered: list[tuple[Document, float]] = [
            (doc, score) for doc, score in raw_results if score <= threshold
        ]

        self._logger.info(
            "Score retrieval: %d/%d results within threshold %.3f for query: %r",
            len(filtered),
            len(raw_results),
            threshold,
            query[:80],
        )
        return filtered


retriever: KnowledgeRetriever = KnowledgeRetriever()