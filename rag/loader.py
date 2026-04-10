from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings

logger: logging.Logger = logging.getLogger(__name__)

_vectorstore: FAISS | None = None


def _build_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)


def _load_and_split_kb() -> list[Document]:
    settings = get_settings()
    kb_path: Path = settings.KB_PATH

    if not kb_path.exists():
        raise FileNotFoundError(
            f"Knowledge base not found at {kb_path}. "
            "Ensure autostream_kb.md exists in the knowledge/ directory."
        )

    raw_text: str = kb_path.read_text(encoding="utf-8")

    if not raw_text.strip():
        raise ValueError(
            f"Knowledge base at {kb_path} is empty. "
            "The file must contain product content before an index can be built."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )

    chunks: list[Document] = splitter.create_documents([raw_text])
    logger.info(
        "Split knowledge base into %d chunks from %s",
        len(chunks),
        kb_path.name,
    )
    return chunks


def get_vectorstore() -> FAISS:
    """Build or load the FAISS vectorstore backing all RAG retrieval."""
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    settings = get_settings()
    embeddings: HuggingFaceEmbeddings = _build_embeddings()
    index_dir: Path = settings.FAISS_INDEX_PATH

    # save_local() writes two artefacts: index.faiss (the vector index) and

    if index_dir.is_dir() and (index_dir / "index.faiss").is_file():
        logger.info("Loading persisted FAISS index from %s", index_dir)
        _vectorstore = FAISS.load_local(
            str(index_dir),
            embeddings,

            allow_dangerous_deserialization=True,
        )
        return _vectorstore

    logger.info("No persisted index at %s — building from knowledge base", index_dir)
    chunks: list[Document] = _load_and_split_kb()

    _vectorstore = FAISS.from_documents(chunks, embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    _vectorstore.save_local(str(index_dir))
    logger.info(
        "FAISS index built and persisted to %s (%d vectors)",
        index_dir,
        len(chunks),
    )

    return _vectorstore