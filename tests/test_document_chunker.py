import unittest

from app.services.document_chunker import DocumentChunker


class DocumentChunkerTests(unittest.TestCase):
    def test_chunks_at_sentence_boundaries_with_overlap_and_source_offsets(self) -> None:
        text = "Első mondat árvíztűrő tükörfúrógép. Második mondat pénzügyi szabályról. " * 8
        chunker = DocumentChunker(chunk_size=100, overlap=20)

        chunks = chunker.chunk_document("szabalyzat.txt", text)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].start_char, 0)
        self.assertEqual(chunks[-1].end_char, len(text.rstrip()))
        for index, chunk in enumerate(chunks):
            with self.subTest(chunk=index):
                self.assertEqual(chunk.document_name, "szabalyzat.txt")
                self.assertEqual(chunk.chunk_index, index)
                self.assertLessEqual(len(chunk.text), 100)
                self.assertEqual(
                    chunk.text,
                    text[chunk.start_char : chunk.end_char],
                )
                if index:
                    self.assertLessEqual(chunk.start_char, chunks[index - 1].end_char)

        self.assertIn("árvíztűrő tükörfúrógép", "".join(chunk.text for chunk in chunks))

    def test_long_unbroken_text_is_split_without_losing_characters(self) -> None:
        text = "ő" * 250
        chunks = DocumentChunker(chunk_size=60, overlap=10).chunk_document(
            "long.txt",
            text,
        )

        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk.text) <= 60 for chunk in chunks))
        self.assertEqual(chunks[0].start_char, 0)
        self.assertEqual(chunks[-1].end_char, len(text))
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(previous.end_char - current.start_char, 10)

    def test_empty_text_returns_no_chunks(self) -> None:
        self.assertEqual(DocumentChunker().chunk_document("empty.txt", " \n\t "), [])

    def test_chunker_rejects_invalid_size_and_overlap(self) -> None:
        for chunk_size, overlap in ((0, 0), (True, 0), (10, -1), (10, 10), (10.5, 0)):
            with self.subTest(chunk_size=chunk_size, overlap=overlap):
                with self.assertRaises(ValueError):
                    DocumentChunker(chunk_size=chunk_size, overlap=overlap)

    def test_chunks_multiple_documents_independently(self) -> None:
        chunker = DocumentChunker(chunk_size=8, overlap=2)

        chunks = chunker.chunk_documents({"one.txt": "one document", "two.txt": "second"})

        self.assertEqual(set(chunks), {"one.txt", "two.txt"})
        self.assertTrue(all(chunk.document_name == "one.txt" for chunk in chunks["one.txt"]))
        self.assertEqual([chunk.text for chunk in chunks["two.txt"]], ["second"])


if __name__ == "__main__":
    unittest.main()
