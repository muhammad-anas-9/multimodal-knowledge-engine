from pathlib import Path

from config import load_environment
from ingest import parse_pdf_to_markdown

# Load secrets before any LlamaIndex imports.
load_environment(required_env_vars=("LLAMA_CLOUD_API_KEY",))

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


def build_and_persist_index(documents, collection_name: str = "multimodal_docs"):
    """Create a local Qdrant-backed VectorStoreIndex and persist index metadata."""
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    qdrant_client = QdrantClient(path="./qdrant_data")
    try:
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
        )
        index.storage_context.persist(persist_dir="./storage")
        return index
    finally:
        qdrant_client.close()


if __name__ == "__main__":
    default_pdf = Path("test_pdf.pdf")
    if not default_pdf.exists():
        raise FileNotFoundError(
            f"Test PDF missing: {default_pdf.resolve()}. "
            "Place a PDF named 'test_pdf.pdf' in this folder."
        )

    parsed_docs = parse_pdf_to_markdown(str(default_pdf))
    index = build_and_persist_index(parsed_docs)
    print(f"Indexed {len(parsed_docs)} parsed document chunk(s).")
    print(f"Index type: {type(index).__name__}")
    print("Persisted to ./qdrant_data and ./storage")
