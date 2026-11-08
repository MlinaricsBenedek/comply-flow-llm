import hashlib
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Protocol

from app.services.document_chunker import DocumentChunk


DEFAULT_COLLECTION_NAME = "reference_documents"
DEFAULT_VECTOR_STORE_PATH = Path.home() / ".comply-flow" / "vector_store"


@dataclass(frozen=True)
class VectorSearchResult:
    document_name: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    distance: float


class VectorStore(Protocol):
    def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None: ...

    def search(
        self,
        embedding: Sequence[float],
        top_k: int = 5,
        document_name: str | None = None,
    ) -> list[VectorSearchResult]: ...

    def delete_document(self, document_name: str) -> None: ...

    def close(self) -> None: ...


class ChromaVectorStore:
    def __init__(
        self,
        path: str | Path | None = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty.")

        configured_path = path if path is not None else os.getenv("VECTOR_STORE_DIR")
        self.path = Path(configured_path) if configured_path else DEFAULT_VECTOR_STORE_PATH

        import chromadb

        self._client = chromadb.PersistentClient(
            path=str(self.path),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each document chunk must have exactly one embedding.")
        if not chunks:
            return

        if any(not self._valid_chunk(chunk) for chunk in chunks):
            raise ValueError("Chunks must include valid source metadata and non-empty text.")
        vectors = [self._validate_embedding(embedding) for embedding in embeddings]
        dimensions = len(vectors[0])
        if any(len(vector) != dimensions for vector in vectors):
            raise ValueError("All embeddings must have the same dimensions.")
        chunk_ids = [
            self._chunk_id(chunk.document_name, chunk.chunk_index)
            for chunk in chunks
        ]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Chunks must have unique document names and indexes.")

        self._collection.upsert(
            ids=chunk_ids,
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "document_name": chunk.document_name,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }
                for chunk in chunks
            ],
        )

    def search(
        self,
        embedding: Sequence[float],
        top_k: int = 5,
        document_name: str | None = None,
    ) -> list[VectorSearchResult]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if document_name is not None and not document_name.strip():
            raise ValueError("document_name must not be empty.")
        vector = self._validate_embedding(embedding)
        filters = {"document_name": document_name} if document_name else None
        response = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=filters,
            include=["documents", "metadatas", "distances"],
        )
        return self._parse_search_response(response)

    def delete_document(self, document_name: str) -> None:
        if not document_name.strip():
            raise ValueError("document_name must not be empty.")
        self._collection.delete(where={"document_name": document_name})

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _chunk_id(document_name: str, chunk_index: int) -> str:
        source_id = f"{document_name}\0{chunk_index}".encode("utf-8")
        return hashlib.sha256(source_id).hexdigest()

    @staticmethod
    def _valid_chunk(chunk: DocumentChunk) -> bool:
        return (
            bool(chunk.document_name.strip())
            and bool(chunk.text.strip())
            and isinstance(chunk.chunk_index, int)
            and not isinstance(chunk.chunk_index, bool)
            and chunk.chunk_index >= 0
            and isinstance(chunk.start_char, int)
            and not isinstance(chunk.start_char, bool)
            and isinstance(chunk.end_char, int)
            and not isinstance(chunk.end_char, bool)
            and 0 <= chunk.start_char <= chunk.end_char
        )

    @staticmethod
    def _validate_embedding(embedding: Sequence[float]) -> list[float]:
        if isinstance(embedding, (str, bytes)) or len(embedding) == 0:
            raise ValueError("An embedding must contain at least one numeric value.")
        if any(not isinstance(value, Real) or isinstance(value, bool) for value in embedding):
            raise ValueError("Embeddings must contain only numeric values.")
        vector = [float(value) for value in embedding]
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Embeddings must contain only finite values.")
        return vector

    @staticmethod
    def _parse_search_response(
        response: Mapping[str, object],
    ) -> list[VectorSearchResult]:
        documents_by_query = response.get("documents")
        metadatas_by_query = response.get("metadatas")
        distances_by_query = response.get("distances")
        if not all(
            isinstance(values, list) and len(values) == 1
            for values in (
                documents_by_query,
                metadatas_by_query,
                distances_by_query,
            )
        ):
            raise RuntimeError("The vector store returned an invalid search response.")

        documents = documents_by_query[0]
        metadatas = metadatas_by_query[0]
        distances = distances_by_query[0]
        if not (
            isinstance(documents, list)
            and isinstance(metadatas, list)
            and isinstance(distances, list)
        ):
            raise RuntimeError("The vector store returned an invalid search response.")
        if not (len(documents) == len(metadatas) == len(distances)):
            raise RuntimeError("The vector store returned inconsistent search results.")

        results = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            if not isinstance(document, str) or not isinstance(metadata, Mapping):
                raise RuntimeError("The vector store returned malformed result metadata.")
            try:
                results.append(
                    VectorSearchResult(
                        document_name=str(metadata["document_name"]),
                        chunk_index=int(metadata["chunk_index"]),
                        text=document,
                        start_char=int(metadata["start_char"]),
                        end_char=int(metadata["end_char"]),
                        distance=float(distance),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    "The vector store returned malformed result metadata."
                ) from error
        return results
