import unittest
from collections.abc import Sequence

from app.services.document_chunker import DocumentChunker
from app.services.document_indexer import DocumentIndexer


class FakeEmbeddingService:
    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.vectors = vectors
        self.embedded_texts: list[str] | None = None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded_texts = list(texts)
        if self.vectors is not None:
            return self.vectors
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


class FakeVectorStore:
    def __init__(self) -> None:
        self.replacements: list[tuple[str, list[object], list[list[float]]]] = []

    def replace_document_chunks(
        self,
        document_name: str,
        chunks: Sequence[object],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        self.replacements.append(
            (document_name, list(chunks), [list(vector) for vector in embeddings])
        )


class DocumentIndexerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding_service = FakeEmbeddingService()
        self.vector_store = FakeVectorStore()
        self.indexer = DocumentIndexer(
            chunker=DocumentChunker(chunk_size=12, overlap=3),
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
        )

    def test_chunks_and_embeds_documents_then_replaces_each_document(self) -> None:
        result = self.indexer.index_documents(
            {
                "first.txt": "Első dokumentum szövege.",
                "empty.txt": " \n ",
                "second.txt": "Második dokumentum.",
            }
        )

        self.assertEqual(result.document_count, 3)
        self.assertEqual(
            result.chunk_count,
            sum(len(chunks) for _, chunks, _ in self.vector_store.replacements),
        )
        self.assertEqual(
            self.embedding_service.embedded_texts,
            [
                chunk.text
                for _, chunks, _ in self.vector_store.replacements
                for chunk in chunks
            ],
        )
        self.assertEqual(
            [name for name, _, _ in self.vector_store.replacements],
            ["first.txt", "empty.txt", "second.txt"],
        )
        self.assertEqual(self.vector_store.replacements[1][1:], ([], []))
        self.assertTrue(
            all(
                len(chunks) == len(embeddings)
                for _, chunks, embeddings in self.vector_store.replacements
            )
        )

    def test_reindexing_replaces_current_documents_to_apply_updates(self) -> None:
        self.indexer.index_documents({"policy.txt": "Régi szabályzat"})
        first_replacements = len(self.vector_store.replacements)

        self.indexer.index_documents({"policy.txt": "Frissített szabályzat"})

        self.assertEqual(len(self.vector_store.replacements), first_replacements + 1)
        self.assertEqual(
            [chunk.text for chunk in self.vector_store.replacements[-1][1]],
            self.embedding_service.embedded_texts,
        )

    def test_embedding_failure_does_not_change_the_vector_store(self) -> None:
        self.embedding_service.vectors = []

        with self.assertRaises(RuntimeError):
            self.indexer.index_documents({"policy.txt": "Szabályzat tartalma."})

        self.assertEqual(self.vector_store.replacements, [])

    def test_invalid_embedding_data_does_not_partially_update_documents(self) -> None:
        documents = {
            "first.txt": "Első dokumentum.",
            "second.txt": "Második dokumentum.",
        }
        chunk_count = sum(
            len(self.indexer.chunker.chunk_document(name, text))
            for name, text in documents.items()
        )
        self.embedding_service.vectors = [[1.0, 0.0]] * (chunk_count - 1) + [
            [float("nan"), 1.0]
        ]

        with self.assertRaises(ValueError):
            self.indexer.index_documents(documents)

        self.assertEqual(self.vector_store.replacements, [])

    def test_invalid_document_fails_before_embedding_or_storage(self) -> None:
        with self.assertRaises(ValueError):
            self.indexer.index_documents({" ": "Szöveg"})

        self.assertIsNone(self.embedding_service.embedded_texts)
        self.assertEqual(self.vector_store.replacements, [])


if __name__ == "__main__":
    unittest.main()
