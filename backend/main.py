import os
import re
import warnings
import tempfile
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="pydantic")

import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.fastembed import FastEmbedEmbedding
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
qdrant_client = QdrantClient(location=":memory:")
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo").encode

Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

class ChatRequest(BaseModel):
    message: str
    model_id: str
    use_rag: bool
    current_device_tokens: int
    device_id: str


class ClearRequest(BaseModel):
    device_id: str


def parse_thought_process(text: str):
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if match:
        thought_process = match.group(1).strip()
        clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return thought_process, clean_text
    return None, text


def get_query_engine(collection_name: str):
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
    )
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index.as_query_engine(similarity_top_k=5)


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if request.current_device_tokens >= PORTFOLIO_LIMIT:
        raise HTTPException(status_code=403, detail="Portfolio token limit reached.")

    try:
        custom_system_prompt = (
            f"You are the 'Multimodal RAG Assistant', a powerful AI software application developed and engineered by Anas. "
            f"Your underlying foundation AI model is currently {request.model_id}. "
            f"CRITICAL INSTRUCTIONS: "
            f"1. If a user asks 'Who built this app?', 'Who created you?', or 'Who is your developer?', you must answer that the application was built by Anas. "
            f"2. If a user asks 'What model are you?', 'Are you Llama?', or 'Are you Qwen?', you must truthfully state that your underlying model is {request.model_id}, but you are running inside the Multimodal RAG Assistant built by Anas."
        )
        Settings.llm = Groq(model=request.model_id, system_prompt=custom_system_prompt)

        if request.use_rag:
            if not qdrant_client.collection_exists(request.device_id):
                return {
                    "text": "The Knowledge Base is currently empty. Please upload a document first, or turn off RAG mode to chat normally.",
                    "thought_process": "",
                    "tokens_used_this_turn": 0,
                }

            query_engine = get_query_engine(request.device_id)
            try:
                rag_context = query_engine.query(request.message)
                rag_text = str(rag_context)
            except Exception as e:
                if "No indexed document" in str(e):
                    return {
                        "text": "The Knowledge Base is currently empty. Please upload a document first, or turn off RAG mode to chat normally.",
                        "thought_process": "",
                        "tokens_used_this_turn": 0,
                    }
                raise

            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=custom_system_prompt),
                ChatMessage(role=MessageRole.USER, content=f"Based on this context:\n{rag_text}\n\nAnswer the user's question: {request.message}")
            ]
            raw_response = Settings.llm.chat(messages)
            response_text = str(raw_response)
        else:
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=custom_system_prompt),
                ChatMessage(role=MessageRole.USER, content=request.message)
            ]
            raw_response = Settings.llm.chat(messages)
            response_text = str(raw_response)

        raw_text = response_text
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if "model_decommissioned" in error_msg or "decommissioned" in error_msg:
            raise HTTPException(
                status_code=400,
                detail="Selected model is no longer available. Please choose another model from the dropdown.",
            ) from e
        if "429" in error_msg or "rate limit" in error_msg:
            raise HTTPException(status_code=429, detail="Model rate limited.") from e
        raise HTTPException(status_code=500, detail=str(e)) from e

    thought_process, clean_text = parse_thought_process(raw_text)
    clean_text = re.sub(r"^\s*assistant:\s*", "", clean_text, flags=re.IGNORECASE)
    tokens_used = len(tokenizer(request.message)) + len(tokenizer(raw_text))

    return {
        "text": clean_text,
        "thought_process": thought_process,
        "tokens_used_this_turn": tokens_used,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), device_id: str = Form(...)):
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
            collection_name=device_id,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return {"status": "success", "filename": file.filename}


@app.post("/api/clear")
async def clear(request: ClearRequest):
    if qdrant_client.collection_exists(request.device_id):
        qdrant_client.delete_collection(collection_name=request.device_id)
    return {"status": "cleared"}
