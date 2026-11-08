from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import Protocol

from app.services.document_chunker import DocumentChunk, DocumentChunker


class DocumentEmbedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


class DocumentVectorStore(Protocol):
    def replace_document_chunks(
        self,
        document_name: str,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None: ...


@dataclass(frozen=True)
class IndexingResult:
    document_count: int
    chunk_count: int


class DocumentIndexer:
    def __init__(
        self,
        chunker: DocumentChunker,
        embedding_service: DocumentEmbedder,
        vector_store: DocumentVectorStore,
    ) -> None:
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def index_documents(self, documents: Mapping[str, str]) -> IndexingResult:
        chunks_by_document: dict[str, list[DocumentChunk]] = {}
        all_chunks: list[DocumentChunk] = []
        for document_name, text in documents.items():
            if not document_name.strip():
                raise ValueError("Document names must not be empty.")
            if not isinstance(text, str):
                raise TypeError("Document contents must be strings.")
            chunks = self.chunker.chunk_document(document_name, text)
            chunks_by_document[document_name] = chunks
            all_chunks.extend(chunks)

        embeddings = self.embedding_service.embed_documents(
            [chunk.text for chunk in all_chunks]
        )
        if len(embeddings) != len(all_chunks):
            raise RuntimeError("The embedding service returned an unexpected number of vectors.")
        embeddings = self._validate_embeddings(embeddings)

        embeddings_by_document: dict[str, list[list[float]]] = {}
        offset = 0
        for document_name, chunks in chunks_by_document.items():
            embeddings_by_document[document_name] = embeddings[offset : offset + len(chunks)]
            offset += len(chunks)

        for document_name, chunks in chunks_by_document.items():
            self.vector_store.replace_document_chunks(
                document_name,
                chunks,
                embeddings_by_document[document_name],
            )

        return IndexingResult(
            document_count=len(documents),
            chunk_count=len(all_chunks),
        )

    @staticmethod
    def _validate_embeddings(
        embeddings: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        expected_dimensions: int | None = None
        for embedding in embeddings:
            if isinstance(embedding, (str, bytes)) or len(embedding) == 0:
                raise ValueError("The embedding service returned an invalid vector.")
            if any(
                not isinstance(value, Real) or isinstance(value, bool)
                for value in embedding
            ):
                raise ValueError("The embedding service returned non-numeric values.")
            vector = [float(value) for value in embedding]
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("The embedding service returned non-finite values.")
            if expected_dimensions is None:
                expected_dimensions = len(vector)
            elif len(vector) != expected_dimensions:
                raise ValueError("The embedding service returned inconsistent vector dimensions.")
            vectors.append(vector)
        return vectors
