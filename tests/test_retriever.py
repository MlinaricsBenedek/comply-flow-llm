import os
import tempfile
import unittest
from unittest.mock import patch

from app.services.document_chunker import DocumentChunk
from app.services.retriever import RetrievedSource, Retriever
from app.services.vector_store import ChromaVectorStore, VectorSearchResult


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        self.queries.extend(texts)
        return [[0.6, 0.8]]


class FakeVectorStore:
    def __init__(self, results: list[VectorSearchResult] | None = None) -> None:
        self.results = results or []
        self.query: tuple[list[float], int] | None = None

    def count(self) -> int:
        return len(self.results)

    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        document_name: str | None = None,
    ) -> list[VectorSearchResult]:
        self.query = (embedding, top_k)
        return self.results[:top_k]


class RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding_service = FakeEmbeddingService()
        self.vector_store = FakeVectorStore(
            [
                VectorSearchResult(
                    "policy.txt",
                    0,
                    "Magas relevanciájú rész.",
                    10,
                    34,
                    0.12,
                ),
                VectorSearchResult(
                    "notice.txt",
                    2,
                    "Alacsony relevanciájú rész.",
                    50,
                    76,
                    0.61,
                ),
            ]
        )
        self.retriever = Retriever(
            self.embedding_service,
            self.vector_store,
            top_k=2,
            min_similarity=0.5,
        )

    def test_builds_query_from_complaint_preprocessing_and_rule_result(self) -> None:
        results = self.retriever.retrieve(
            "A kártyadíjjal kapcsolatban kérdezek.",
            facts={"PRODUCT": ["bankkártya"]},
            category="GENERAL_COMPLAINT",
            rule_result={
                "ruleDecision": "provide_information",
                "ruleReason": "Díj ellenőrzése szükséges.",
                "matchedRules": ["CARD_FEE"],
                "ignored": "not included",
            },
        )

        query = self.embedding_service.queries[0]
        self.assertIn("A kártyadíjjal kapcsolatban kérdezek.", query)
        self.assertIn("GENERAL_COMPLAINT", query)
        self.assertIn("bankkártya", query)
        self.assertIn("CARD_FEE", query)
        self.assertNotIn("not included", query)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].document_name, "policy.txt")
        self.assertAlmostEqual(results[0].similarity, 0.88)
        self.assertEqual(self.vector_store.query, ([0.6, 0.8], 2))

    def test_top_k_and_similarity_threshold_are_configurable_per_query(self) -> None:
        results = self.retriever.retrieve(
            "Kérdésem van.",
            top_k=2,
            min_similarity=0.3,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(self.vector_store.query, ([0.6, 0.8], 2))

    def test_empty_vector_store_does_not_load_or_call_embedding_model(self) -> None:
        empty_store = FakeVectorStore()
        embedding_service = FakeEmbeddingService()
        retriever = Retriever(embedding_service, empty_store)

        results = retriever.retrieve("Milyen díjak vonatkoznak rám?")

        self.assertEqual(results, [])
        self.assertEqual(embedding_service.queries, [])

    def test_rejects_invalid_query_and_overrides(self) -> None:
        for kwargs in (
            {"top_k": 0},
            {"top_k": True},
            {"min_similarity": -0.1},
            {"min_similarity": 1.1},
            {"min_similarity": float("nan")},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.retriever.retrieve("Kérdés", **kwargs)

        with self.assertRaises(ValueError):
            self.retriever.retrieve("  ")

    def test_environment_settings_are_validated(self) -> None:
        with patch.dict(
            os.environ,
            {"RETRIEVAL_TOP_K": "3", "RETRIEVAL_MIN_SIMILARITY": "0.65"},
        ):
            retriever = Retriever.from_environment(
                self.embedding_service,
                self.vector_store,
            )

        self.assertEqual(retriever.top_k, 3)
        self.assertEqual(retriever.min_similarity, 0.65)

        with patch.dict(os.environ, {"RETRIEVAL_TOP_K": "invalid"}):
            with self.assertRaises(ValueError):
                Retriever.from_environment(
                    self.embedding_service,
                    self.vector_store,
                )

    def test_chroma_returns_the_complaint_related_document_chunk(self) -> None:
        class SemanticTestEmbedder:
            def embed_queries(self, texts: list[str]) -> list[list[float]]:
                self.query = texts[0]
                return [[1.0, 0.0]]

        with tempfile.TemporaryDirectory() as directory:
            store = ChromaVectorStore(path=directory)
            try:
                store.upsert_chunks(
                    [
                        DocumentChunk(
                            "kartyadij.txt",
                            0,
                            "A bankkártya éves díja a hirdetmény szerint.",
                            0,
                            47,
                        ),
                        DocumentChunk(
                            "lakasberlet.txt",
                            0,
                            "A bérleti szerződés felmondási ideje.",
                            0,
                            39,
                        ),
                    ],
                    [[1.0, 0.0], [0.0, 1.0]],
                )
                embedder = SemanticTestEmbedder()
                retriever = Retriever(
                    embedder,
                    store,
                    top_k=2,
                    min_similarity=0.7,
                )

                results = retriever.retrieve(
                    "Miért vontak le éves bankkártyadíjat?",
                    category="GENERAL_COMPLAINT",
                )

                self.assertIn("Miért vontak le éves bankkártyadíjat?", embedder.query)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].document_name, "kartyadij.txt")
                self.assertIn("bankkártya éves díja", results[0].text)
                self.assertGreaterEqual(results[0].similarity, 0.7)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
