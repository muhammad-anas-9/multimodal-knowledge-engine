import os
import re
import tempfile
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_parse import LlamaParse
from pydantic import BaseModel
from qdrant_client import QdrantClient

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

app = FastAPI(title="Multimodal RAG Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PORTFOLIO_LIMIT = 50000
COLLECTION_NAME = "multimodal_docs"

qdrant_client = QdrantClient(location=":memory:")
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo").encode

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


class ChatRequest(BaseModel):
    message: str
    model_id: str
    use_rag: bool
    current_device_tokens: int


def parse_thought_process(text: str):
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if match:
        thought_process = match.group(1).strip()
        clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return thought_process, clean_text
    return None, text


def get_query_engine():
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
    )
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index.as_query_engine(similarity_top_k=5)


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if request.current_device_tokens >= PORTFOLIO_LIMIT:
        raise HTTPException(status_code=403, detail="Portfolio token limit reached.")

    try:
        Settings.llm = Groq(model=request.model_id)

        if request.use_rag:
            if not qdrant_client.collection_exists(COLLECTION_NAME):
                raise HTTPException(status_code=400, detail="No indexed document available.")
            response = get_query_engine().query(request.message)
            raw_text = str(response)
        else:
            response = Settings.llm.complete(request.message)
            raw_text = str(response)
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if "429" in error_msg or "rate limit" in error_msg:
            raise HTTPException(status_code=429, detail="Model rate limited.") from e
        raise HTTPException(status_code=500, detail=str(e)) from e

    thought_process, clean_text = parse_thought_process(raw_text)
    tokens_used = len(tokenizer(request.message)) + len(tokenizer(raw_text))

    return {
        "text": clean_text,
        "thought_process": thought_process,
        "tokens_used_this_turn": tokens_used,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    temp_file_path = None
    try:
        file_suffix = Path(file.filename).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        parser = LlamaParse(result_type="markdown")
        documents = parser.load_data(temp_file_path)

        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return {"status": "success", "filename": file.filename}


@app.post("/api/clear")
async def clear():
    global qdrant_client
    qdrant_client = QdrantClient(location=":memory:")
    return {"status": "cleared"}
