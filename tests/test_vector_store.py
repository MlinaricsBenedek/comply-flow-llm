import math
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from app.services.document_chunker import DocumentChunk
from app.services.vector_store import ChromaVectorStore


class FakeCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}
        self.last_query: dict[str, object] | None = None

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, str | int]],
    ) -> None:
        for record_id, embedding, document, metadata in zip(
            ids,
            embeddings,
            documents,
            metadatas,
        ):
            self.records[record_id] = {
                "embedding": embedding,
                "document": document,
                "metadata": metadata,
            }

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, str] | None,
        include: list[str],
    ) -> dict[str, list[list[object]]]:
        self.last_query = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "where": where,
            "include": include,
        }
        query = query_embeddings[0]
        matches = [
            record
            for record in self.records.values()
            if where is None
            or record["metadata"]["document_name"] == where["document_name"]
        ]
        matches.sort(
            key=lambda record: self._cosine_distance(query, record["embedding"])
        )
        matches = matches[:n_results]
        return {
            "documents": [[record["document"] for record in matches]],
            "metadatas": [[record["metadata"] for record in matches]],
            "distances": [
                [
                    self._cosine_distance(query, record["embedding"])
                    for record in matches
                ]
            ],
        }

    def get(
        self,
        where: dict[str, str],
        include: list[str],
    ) -> dict[str, list[object]]:
        records = [
            (record_id, record)
            for record_id, record in self.records.items()
            if record["metadata"]["document_name"] == where["document_name"]
        ]
        return {
            "ids": [record_id for record_id, _ in records],
            "metadatas": [record["metadata"] for _, record in records],
        }

    def delete(
        self,
        where: dict[str, str] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        if ids is not None:
            for record_id in ids:
                self.records.pop(record_id, None)
            return
        if where is not None:
            self.records = {
                record_id: record
                for record_id, record in self.records.items()
                if record["metadata"]["document_name"] != where["document_name"]
            }

    @staticmethod
    def _cosine_distance(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Vector dimensions do not match.")
        return 1 - sum(a * b for a, b in zip(left, right)) / (
            math.sqrt(sum(value * value for value in left))
            * math.sqrt(sum(value * value for value in right))
        )


class FakeChromaClient:
    def __init__(self) -> None:
        self.collection = FakeCollection()
        self.collection_options: dict[str, object] | None = None
        self.closed = False

    def get_or_create_collection(
        self,
        name: str,
        metadata: dict[str, str],
    ) -> FakeCollection:
        self.collection_options = {"name": name, "metadata": metadata}
        return self.collection

    def close(self) -> None:
        self.closed = True


class VectorStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeChromaClient()
        chromadb = types.ModuleType("chromadb")
        chromadb.PersistentClient = lambda **kwargs: self.client
        chromadb.Settings = lambda **kwargs: kwargs
        self.chromadb_module = chromadb

    def create_store(self, path: str) -> ChromaVectorStore:
        with patch.dict(sys.modules, {"chromadb": self.chromadb_module}):
            return ChromaVectorStore(path=path)

    def test_upserts_chunks_with_source_metadata_and_cosine_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            chunk = DocumentChunk("policy.txt", 0, "Magyar szöveg.", 12, 26)

            store.upsert_chunks([chunk], [[1.0, 0.0, 0.0]])

        self.assertEqual(
            self.client.collection_options,
            {
                "name": "reference_documents",
                "metadata": {"hnsw:space": "cosine"},
            },
        )
        record = next(iter(self.client.collection.records.values()))
        self.assertEqual(record["document"], "Magyar szöveg.")
        self.assertEqual(
            record["metadata"],
            {
                "document_name": "policy.txt",
                "chunk_index": 0,
                "start_char": 12,
                "end_char": 26,
            },
        )

    def test_search_returns_nearest_chunks_and_supports_document_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            store.upsert_chunks(
                [
                    DocumentChunk("policy.txt", 0, "Első rész", 0, 9),
                    DocumentChunk("notice.txt", 0, "Másik rész", 0, 10),
                ],
                [[1.0, 0.0], [0.0, 1.0]],
            )

            results = store.search([0.9, 0.1], top_k=2, document_name="policy.txt")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].document_name, "policy.txt")
        self.assertEqual(results[0].chunk_index, 0)
        self.assertEqual(results[0].text, "Első rész")
        self.assertEqual(results[0].start_char, 0)
        self.assertEqual(results[0].end_char, 9)
        self.assertAlmostEqual(results[0].distance, 1 - 0.9 / math.sqrt(0.82))
        self.assertEqual(
            self.client.collection.last_query["where"],
            {"document_name": "policy.txt"},
        )

    def test_upserting_same_chunk_id_updates_existing_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            store.upsert_chunks(
                [DocumentChunk("policy.txt", 0, "Old text", 0, 8)],
                [[1.0, 0.0]],
            )
            store.upsert_chunks(
                [DocumentChunk("policy.txt", 0, "New text", 0, 8)],
                [[0.0, 1.0]],
            )

        self.assertEqual(len(self.client.collection.records), 1)
        self.assertEqual(
            next(iter(self.client.collection.records.values()))["document"],
            "New text",
        )

    def test_replacing_document_removes_stale_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            store.replace_document_chunks(
                "policy.txt",
                [
                    DocumentChunk("policy.txt", 0, "First", 0, 5),
                    DocumentChunk("policy.txt", 1, "Second", 6, 12),
                    DocumentChunk("policy.txt", 2, "Third", 13, 18),
                ],
                [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            )
            store.replace_document_chunks(
                "policy.txt",
                [DocumentChunk("policy.txt", 0, "Updated", 0, 7)],
                [[0.0, 1.0]],
            )

        self.assertEqual(len(self.client.collection.records), 1)
        record = next(iter(self.client.collection.records.values()))
        self.assertEqual(record["document"], "Updated")
        self.assertEqual(record["metadata"]["chunk_index"], 0)

    def test_replacing_document_with_no_chunks_removes_all_old_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            store.replace_document_chunks(
                "policy.txt",
                [DocumentChunk("policy.txt", 0, "Old", 0, 3)],
                [[1.0, 0.0]],
            )

            store.replace_document_chunks("policy.txt", [], [])

        self.assertEqual(self.client.collection.records, {})

    def test_replacing_document_removes_noncontiguous_legacy_chunk_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            store.upsert_chunks(
                [
                    DocumentChunk("policy.txt", 0, "Old first", 0, 9),
                    DocumentChunk("policy.txt", 5, "Legacy chunk", 30, 42),
                ],
                [[1.0, 0.0], [0.0, 1.0]],
            )

            store.replace_document_chunks(
                "policy.txt",
                [DocumentChunk("policy.txt", 0, "Updated", 0, 7)],
                [[0.0, 1.0]],
            )

        self.assertEqual(len(self.client.collection.records), 1)
        self.assertEqual(
            next(iter(self.client.collection.records.values()))["document"],
            "Updated",
        )

    def test_delete_document_removes_only_that_documents_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            store.upsert_chunks(
                [
                    DocumentChunk("policy.txt", 0, "Policy", 0, 6),
                    DocumentChunk("notice.txt", 0, "Notice", 0, 6),
                ],
                [[1.0, 0.0], [0.0, 1.0]],
            )

            store.delete_document("policy.txt")

        self.assertEqual(len(self.client.collection.records), 1)
        record = next(iter(self.client.collection.records.values()))
        self.assertEqual(record["metadata"]["document_name"], "notice.txt")

    def test_rejects_mismatched_counts_and_invalid_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            chunk = DocumentChunk("policy.txt", 0, "Policy", 0, 6)

            with self.assertRaises(ValueError):
                store.upsert_chunks([chunk], [])
            with self.assertRaises(ValueError):
                store.upsert_chunks([chunk, chunk], [[1.0]])
            with self.assertRaises(ValueError):
                store.upsert_chunks([chunk], [[float("nan")]])
            with self.assertRaises(ValueError):
                store.search([1.0], top_k=0)

        self.assertEqual(self.client.collection.records, {})

    def test_close_releases_the_persistent_client(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.create_store(directory)
            store.close()

        self.assertTrue(self.client.closed)


if __name__ == "__main__":
    unittest.main()
