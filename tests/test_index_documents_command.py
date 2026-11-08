import os
import unittest
from unittest.mock import Mock, patch

from app.services import index_documents
from app.services.document_indexer import IndexingResult


class IndexDocumentsCommandTests(unittest.TestCase):
    def test_command_loads_and_indexes_documents_then_closes_store(self) -> None:
        documents = {"policy.txt": "Szabályzat"}
        loader = Mock()
        loader.load.return_value = documents
        store = Mock()
        embedding_service = object()

        with (
            patch.dict(
                os.environ,
                {
                    "REFERENCE_DOCUMENTS_DIR": "C:\\reference-documents",
                    "EMBEDDING_MODEL_NAME": "test/model",
                },
                clear=False,
            ),
            patch.object(
                index_documents,
                "ReferenceDocumentLoader",
                return_value=loader,
            ) as loader_factory,
            patch.object(
                index_documents,
                "EmbeddingService",
                return_value=embedding_service,
            ) as embedding_factory,
            patch.object(index_documents, "ChromaVectorStore", return_value=store),
            patch.object(index_documents, "DocumentIndexer") as indexer_factory,
        ):
            indexer_factory.return_value.index_documents.return_value = (
                IndexingResult(document_count=1, chunk_count=2)
            )

            index_documents.main()

        loader_factory.assert_called_once_with(bank="OTP")
        loader.load.assert_called_once_with()
        embedding_factory.assert_called_once_with(model_name="test/model")
        indexer_factory.return_value.index_documents.assert_called_once_with(documents)
        store.close.assert_called_once_with()

    def test_command_selects_raiffeisen_rule_set(self) -> None:
        loader = Mock()
        loader.load.return_value = {"policy.txt": "Szabályzat"}
        store = Mock()
        with (
            patch.object(
                index_documents,
                "ReferenceDocumentLoader",
                return_value=loader,
            ) as loader_factory,
            patch.object(index_documents, "EmbeddingService", return_value=object()),
            patch.object(
                index_documents,
                "ChromaVectorStore",
                return_value=store,
            ) as store_factory,
            patch.object(index_documents, "DocumentIndexer") as indexer_factory,
        ):
            indexer_factory.return_value.index_documents.return_value = (
                IndexingResult(document_count=1, chunk_count=1)
            )
            index_documents.main(bank="Raiffeisen")

        loader_factory.assert_called_once_with(bank="Raiffeisen")
        store_factory.assert_called_once_with(bank="Raiffeisen")
        loader.load.assert_called_once_with()
        store.close.assert_called_once_with()

    def test_empty_document_directory_setting_is_reported(self) -> None:
        with (
            patch.dict(os.environ, {"REFERENCE_DOCUMENTS_DIR": "   "}),
            self.assertRaises(ValueError),
        ):
            index_documents.main()


if __name__ == "__main__":
    unittest.main()
