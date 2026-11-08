import tempfile
import unittest
from pathlib import Path

from app.services.reference_document_loader import (
    REFERENCE_DOCUMENT_FILENAMES,
    ReferenceDocumentLoader,
)


class ReferenceDocumentLoaderTests(unittest.TestCase):
    def test_loads_all_documents_and_preserves_unicode(self) -> None:
        expected = {
            REFERENCE_DOCUMENT_FILENAMES[0]: "Üzletszabályzat – árvíztűrő tükörfúrógép",
            REFERENCE_DOCUMENT_FILENAMES[1]: "2012. évi CXVI. törvény – pénzügyi",
            REFERENCE_DOCUMENT_FILENAMES[2]: "Hirdetmény: forint és ő betű",
        }

        with tempfile.TemporaryDirectory() as directory:
            for index, (filename, content) in enumerate(expected.items()):
                encoding = ("utf-8-sig", "utf-16", "cp1250")[index]
                Path(directory, filename).write_bytes(content.encode(encoding))

            documents = ReferenceDocumentLoader(directory).load()

        self.assertEqual(documents, expected)

    def test_missing_document_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, REFERENCE_DOCUMENT_FILENAMES[0]).write_text(
                "Document",
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError):
                ReferenceDocumentLoader(directory).load()


if __name__ == "__main__":
    unittest.main()
