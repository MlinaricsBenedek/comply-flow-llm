import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from app.services.vector_store import VectorSearchResult


DEFAULT_RETRIEVAL_TOP_K = 5
DEFAULT_MIN_SIMILARITY = 0.7


class QueryEmbedder(Protocol):
    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...


class SearchableVectorStore(Protocol):
    def search(
        self,
        embedding: Sequence[float],
        top_k: int = 5,
        document_name: str | None = None,
    ) -> list[VectorSearchResult]: ...

    def count(self) -> int: ...


@dataclass(frozen=True)
class RetrievedSource:
    document_name: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    similarity: float

    @property
    def source_id(self) -> str:
        return f"{self.document_name}#chunk-{self.chunk_index}"


class Retriever:
    def __init__(
        self,
        embedding_service: QueryEmbedder,
        vector_store: SearchableVectorStore,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ) -> None:
        self._validate_top_k(top_k)
        self._validate_threshold(min_similarity)
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_similarity = min_similarity

    @classmethod
    def from_environment(
        cls,
        embedding_service: QueryEmbedder,
        vector_store: SearchableVectorStore,
    ) -> "Retriever":
        top_k_value = os.getenv("RETRIEVAL_TOP_K", str(DEFAULT_RETRIEVAL_TOP_K))
        try:
            top_k = int(top_k_value)
        except ValueError as error:
            raise ValueError("RETRIEVAL_TOP_K must be a positive integer.") from error

        threshold_value = os.getenv(
            "RETRIEVAL_MIN_SIMILARITY",
            str(DEFAULT_MIN_SIMILARITY),
        )
        try:
            min_similarity = float(threshold_value)
        except ValueError as error:
            raise ValueError(
                "RETRIEVAL_MIN_SIMILARITY must be a number between 0 and 1."
            ) from error
        return cls(embedding_service, vector_store, top_k, min_similarity)

    def retrieve(
        self,
        message: str,
        facts: Mapping[str, object] | None = None,
        category: str | None = None,
        rule_result: Mapping[str, object] | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[RetrievedSource]:
        selected_top_k = self.top_k if top_k is None else top_k
        selected_threshold = (
            self.min_similarity if min_similarity is None else min_similarity
        )
        self._validate_top_k(selected_top_k)
        self._validate_threshold(selected_threshold)
        if not isinstance(message, str) or not message.strip():
            raise ValueError("A non-empty complaint message is required for retrieval.")
        if self.vector_store.count() == 0:
            return []

        query = self._build_query(message, facts, category, rule_result)
        embeddings = self.embedding_service.embed_queries([query])
        if len(embeddings) != 1:
            raise RuntimeError("The embedding service returned an unexpected number of vectors.")
        embedding = self._validate_embedding(embeddings[0])

        matches = self.vector_store.search(embedding, top_k=selected_top_k)
        results = []
        for match in matches:
            if not math.isfinite(match.distance):
                raise RuntimeError("The vector store returned a non-finite distance.")
            similarity = max(0.0, min(1.0, 1.0 - match.distance))
            if similarity >= selected_threshold:
                results.append(
                    RetrievedSource(
                        document_name=match.document_name,
                        chunk_index=match.chunk_index,
                        text=match.text,
                        start_char=match.start_char,
                        end_char=match.end_char,
                        similarity=similarity,
                    )
                )
        return results

    @staticmethod
    def _build_query(
        message: str,
        facts: Mapping[str, object] | None,
        category: str | None,
        rule_result: Mapping[str, object] | None,
    ) -> str:
        parts = [f"Panasz: {message.strip()}"]
        if category and category.strip():
            parts.append(f"Ügytípus: {category.strip()}")
        if facts:
            parts.append(
                "Előfeldolgozási tények: "
                + json.dumps(facts, ensure_ascii=False, sort_keys=True, default=str)
            )
        if rule_result:
            relevant_rule_values = {
                key: rule_result[key]
                for key in ("ruleDecision", "ruleReason", "matchedRules")
                if key in rule_result
            }
            if relevant_rule_values:
                parts.append(
                    "Szabályeredmény: "
                    + json.dumps(
                        relevant_rule_values,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                )
        return "\n".join(parts)

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

    @staticmethod
    def _validate_threshold(threshold: float) -> None:
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (float, int))
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("min_similarity must be a number between 0 and 1.")

    @staticmethod
    def _validate_embedding(embedding: Sequence[float]) -> list[float]:
        if not embedding:
            raise ValueError("The embedding service returned an empty vector.")
        try:
            vector = [float(value) for value in embedding]
        except (TypeError, ValueError) as error:
            raise ValueError("The embedding service returned a non-numeric vector.") from error
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("The embedding service returned a non-finite vector.")
        return vector
