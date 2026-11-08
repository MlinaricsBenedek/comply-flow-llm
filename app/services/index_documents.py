import argparse
import os

from app.services.document_chunker import DocumentChunker
from app.services.document_indexer import DocumentIndexer
from app.services.embedding_service import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingService,
)
from app.services.reference_document_loader import ReferenceDocumentLoader
from app.services.vector_store import ChromaVectorStore


def main(bank: str = "OTP") -> None:
    documents = ReferenceDocumentLoader(bank=bank).load()
    embedding_service = EmbeddingService(
        model_name=os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)
    )
    vector_store = ChromaVectorStore(bank=bank)
    try:
        result = DocumentIndexer(
            chunker=DocumentChunker(),
            embedding_service=embedding_service,
            vector_store=vector_store,
        ).index_documents(documents)
    finally:
        vector_store.close()

    print(
        f"Indexed {result.chunk_count} chunks from "
        f"{result.document_count} documents into {vector_store.path}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index a bank's reference documents.")
    parser.add_argument(
        "--bank",
        choices=("OTP", "Raiffeisen"),
        default="OTP",
        help="Which bank's reference documents to index (default: OTP).",
    )
    main(bank=parser.parse_args().bank)
