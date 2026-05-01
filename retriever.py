from config import load_environment

# Load secrets before any LlamaIndex imports.
load_environment(required_env_vars=("GROQ_API_KEY",))

from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

ENTERPRISE_PROMPT = (
    "You are an elite, objective data extraction assistant. "
    "You only answer based on the provided context. "
    "If the context contains exactly one link, state there is one link. "
    "Do not change your answer, second-guess yourself, or apologize if the user asks 'Are you sure?'. "
    "Trust the data absolutely."
)


def build_query_engine(collection_name: str = "multimodal_docs"):
    """Load local Qdrant index and return a query engine."""
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = Groq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        system_prompt=ENTERPRISE_PROMPT,
    )

    qdrant_client = QdrantClient(path="./qdrant_data")
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
    )
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index.as_query_engine(similarity_top_k=5), qdrant_client


def query_local_index(question: str, collection_name: str = "multimodal_docs"):
    """Load local Qdrant index, query it with Groq, and return the response."""
    query_engine, qdrant_client = build_query_engine(collection_name=collection_name)
    try:
        return query_engine.query(question)
    finally:
        qdrant_client.close()


if __name__ == "__main__":
    query_engine, qdrant_client = build_query_engine()
    test_query = "What are the key data points or highlights mentioned in this document?"
    try:
        response = query_engine.query(test_query)

        print("\n--- AI RESPONSE ---")
        print(response)

        print("\n--- CITATIONS ---")
        for node in response.source_nodes:
            print(f"Source Text:\n{node.node.text}\n")
    finally:
        qdrant_client.close()
