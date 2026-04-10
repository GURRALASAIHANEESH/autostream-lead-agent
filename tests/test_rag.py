from unittest.mock import MagicMock, patch
import pytest
from langchain_core.documents import Document

from rag.retriever import KnowledgeRetriever, retriever as module_retriever


@pytest.fixture
def mock_vs() -> MagicMock:
    return MagicMock()


@pytest.fixture
def fresh_retriever(mock_vs: MagicMock) -> tuple[KnowledgeRetriever, MagicMock]:
    with patch("rag.loader.get_vectorstore", return_value=mock_vs), \
         patch("rag.retriever.get_vectorstore", return_value=mock_vs):
        r = KnowledgeRetriever(k=3)
    # Override the internal LangChain retriever with a plain mock
    # so we control exactly what .invoke() returns
    r._retriever = MagicMock()
    return r, mock_vs


def test_retrieve_returns_joined_chunks(
    fresh_retriever: tuple[KnowledgeRetriever, MagicMock],
) -> None:
    r, mock_vs = fresh_retriever
    docs = [
        Document(page_content="chunk A"),
        Document(page_content="chunk B"),
        Document(page_content="chunk C"),
    ]
    r._retriever.invoke.return_value = docs

    result = r.retrieve("test query")
    assert result == "chunk A\n\n---\n\nchunk B\n\n---\n\nchunk C"


def test_retrieve_returns_fallback_on_empty_results(
    fresh_retriever: tuple[KnowledgeRetriever, MagicMock],
) -> None:
    r, _ = fresh_retriever
    r._retriever.invoke.return_value = []

    result = r.retrieve("obscure query")
    assert "I don't have specific information" in result


def test_retrieve_with_scores_filters_by_threshold(
    fresh_retriever: tuple[KnowledgeRetriever, MagicMock],
) -> None:
    r, mock_vs = fresh_retriever
    doc1 = Document(page_content="relevant")
    doc2 = Document(page_content="marginal")
    doc3 = Document(page_content="irrelevant")

    mock_vs.similarity_search_with_score.return_value = [
        (doc1, 0.3),
        (doc2, 0.6),
        (doc3, 0.8),
    ]

    results = r.retrieve_with_scores("test query", threshold=0.5)
    assert len(results) == 1
    assert results[0][1] == 0.3


def test_retrieve_with_scores_returns_all_below_threshold(
    fresh_retriever: tuple[KnowledgeRetriever, MagicMock],
) -> None:
    r, mock_vs = fresh_retriever
    mock_vs.similarity_search_with_score.return_value = [
        (Document(page_content="first"), 0.1),
        (Document(page_content="second"), 0.2),
        (Document(page_content="third"), 0.4),
    ]

    results = r.retrieve_with_scores("test query", threshold=0.5)
    assert len(results) == 3


def test_retriever_is_singleton() -> None:
    from rag import retriever as retriever_module
    assert retriever_module.retriever is module_retriever