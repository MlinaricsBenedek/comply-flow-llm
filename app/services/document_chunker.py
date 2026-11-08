import re
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    document_name: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int


class DocumentChunker:
    _sentence_boundary = re.compile(r"[.!?][\"'»”’)]*(?:[ \t]+|\n+)")

    def __init__(self, chunk_size: int = 2400, overlap: int = 300) -> None:
        if (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or chunk_size <= 0
        ):
            raise ValueError("chunk_size must be a positive integer.")
        if (
            not isinstance(overlap, int)
            or isinstance(overlap, bool)
            or overlap < 0
        ):
            raise ValueError("overlap must be a non-negative integer.")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(
        self,
        documents: Mapping[str, str],
    ) -> dict[str, list[DocumentChunk]]:
        return {
            document_name: self.chunk_document(document_name, text)
            for document_name, text in documents.items()
        }

    def chunk_document(self, document_name: str, text: str) -> list[DocumentChunk]:
        if not text.strip():
            return []

        chunks: list[DocumentChunk] = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            if end < text_length:
                end = self._find_chunk_end(text, start, end)

            content_start = start
            while content_start < end and text[content_start].isspace():
                content_start += 1
            content_end = end
            while content_end > content_start and text[content_end - 1].isspace():
                content_end -= 1

            if content_start < content_end:
                chunks.append(
                    DocumentChunk(
                        document_name=document_name,
                        chunk_index=len(chunks),
                        text=text[content_start:content_end],
                        start_char=content_start,
                        end_char=content_end,
                    )
                )

            if end == text_length:
                break
            start = self._find_next_start(text, start, end)

        return chunks

    def _find_chunk_end(self, text: str, start: int, limit: int) -> int:
        minimum = start + max(1, (limit - start) // 2)
        segment = text[start:limit]

        paragraph_end = segment.rfind("\n\n")
        if paragraph_end >= minimum - start:
            return start + paragraph_end + 2

        sentence_ends = [
            start + match.end()
            for match in self._sentence_boundary.finditer(segment)
            if start + match.end() >= minimum
        ]
        if sentence_ends:
            return sentence_ends[-1]

        for separator in ("\n", " ", "\t"):
            separator_end = segment.rfind(separator)
            if separator_end >= minimum - start:
                return start + separator_end + len(separator)

        return limit

    def _find_next_start(self, text: str, start: int, end: int) -> int:
        next_start = max(start + 1, end - self.overlap)
        while next_start < end and text[next_start].isspace():
            next_start += 1
        return next_start
