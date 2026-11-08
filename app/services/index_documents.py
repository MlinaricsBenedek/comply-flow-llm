import os

from app.services.document_chunker import DocumentChunker
from app.services.document_indexer import DocumentIndexer
from app.services.embedding_service import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingService,
)
from app.services.reference_document_loader import ReferenceDocumentLoader
from app.services.vector_store import ChromaVectorStore


def main() -> None:
    document_directory = os.getenv("REFERENCE_DOCUMENTS_DIR")
    if document_directory is not None:
        document_directory = document_directory.strip()
        if not document_directory:
            raise ValueError("REFERENCE_DOCUMENTS_DIR must not be empty.")
    documents = ReferenceDocumentLoader(document_directory).load()
    embedding_service = EmbeddingService(
        model_name=os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)
    )
    vector_store = ChromaVectorStore()
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
    main()
