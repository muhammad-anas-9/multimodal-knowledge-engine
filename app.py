import os
import shutil
import tempfile

import streamlit as st

from config import load_environment
from database import build_and_persist_index
from ingest import parse_pdf_to_markdown

# Load only the keys needed for Phase 5.
load_environment(required_env_vars=("LLAMA_CLOUD_API_KEY", "GROQ_API_KEY"))

from llama_index.core import PromptTemplate, Settings, VectorStoreIndex
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.qdrant import QdrantVectorStore

AVAILABLE_MODELS = {
    "Llama 3.3 70B (Heavy & Complex - Best for deep reasoning)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (Light & Fast - Best for quick summaries)": "llama-3.1-8b-instant",
    "Qwen 2.5 32B (Alibaba Architecture - Great context handling)": "qwen-2.5-32b",
    "Mixtral 8x7B (Mistral AI - Sparse Mixture of Experts model)": "mixtral-8x7b-32768",
}

def configure_models(user_selection: str, enterprise_prompt: str) -> None:
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    selected_model_id = AVAILABLE_MODELS[user_selection]
    Settings.llm = Groq(
        model=selected_model_id,
        temperature=0.0,
        system_prompt=enterprise_prompt,
    )


def format_file_size(size_in_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_in_bytes)
    unit_idx = 0
    while value >= 1024 and unit_idx < len(units) - 1:
        value /= 1024
        unit_idx += 1
    return f"{value:.1f} {units[unit_idx]}"


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed_file_signature" not in st.session_state:
        st.session_state.indexed_file_signature = None
    if "index_ready" not in st.session_state:
        st.session_state.index_ready = False


def qdrant_index_exists() -> bool:
    qdrant_path = "./qdrant_data"
    return os.path.isdir(qdrant_path) and any(
        os.scandir(qdrant_path)
    )


def build_chat_history_from_session() -> list[ChatMessage]:
    chat_messages: list[ChatMessage] = []
    for message in st.session_state.messages:
        if message["role"] == "user":
            chat_messages.append(
                ChatMessage(role=MessageRole.USER, content=message["content"])
            )
        elif message["role"] == "assistant":
            chat_messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=message["content"])
            )
    return chat_messages


@st.cache_resource
def get_qdrant_client():
    from qdrant_client import QdrantClient

    return QdrantClient(path="./qdrant_data")


def clear_cached_qdrant_client() -> None:
    if qdrant_index_exists():
        get_qdrant_client().close()
    get_qdrant_client.clear()


def build_rag_query_engine(
    custom_qa_prompt: PromptTemplate, collection_name: str = "multimodal_docs"
):
    qdrant_client = get_qdrant_client()
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
    )
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        text_qa_template=custom_qa_prompt,
    )
    return query_engine


st.set_page_config(page_title="Multimodal RAG", page_icon=":books:")
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 5% -10%, rgba(0,255,170,0.16), transparent 28%),
                    radial-gradient(circle at 95% -15%, rgba(0,170,255,0.16), transparent 30%),
                    #0E1117;
    }
    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    .hero-wrap {
        border: 1px solid rgba(0, 255, 170, 0.18);
        border-radius: 18px;
        padding: 1rem 1.15rem 1.1rem;
        margin-bottom: 0.95rem;
        background: linear-gradient(145deg, rgba(20,24,32,0.92) 0%, rgba(11,15,22,0.92) 100%);
        box-shadow: 0 18px 30px rgba(0, 0, 0, 0.32), inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .hero-title {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        margin: 0;
        color: #FAFAFA;
    }
    .hero-subtitle {
        margin-top: 0.25rem;
        color: #A6B3C2;
        font-size: 0.94rem;
    }
    .status-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin: 0.5rem 0 1rem;
    }
    .status-pill {
        border: 1px solid rgba(255,255,255,0.13);
        border-radius: 999px;
        padding: 0.28rem 0.68rem;
        font-size: 0.78rem;
        color: #DFE7F0;
        background: rgba(255,255,255,0.03);
    }
    .status-pill.ok {
        border-color: rgba(0,255,170,0.45);
        color: #A9FFD9;
        background: rgba(0,255,170,0.1);
    }
    .status-pill.warn {
        border-color: rgba(255,189,89,0.45);
        color: #FFDDA3;
        background: rgba(255,189,89,0.1);
    }
    .empty-state {
        border: 1px dashed rgba(255,255,255,0.18);
        border-radius: 14px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.8rem;
        color: #B4C0CF;
        background: rgba(255,255,255,0.015);
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] > div {
        border: 1px solid rgba(0, 255, 170, 0.35);
        border-radius: 12px;
        background: linear-gradient(180deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.012) 100%);
        transition: all 0.2s ease-in-out;
        padding: 0.25rem;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] > div:hover {
        border-color: rgba(0, 255, 170, 0.8);
        box-shadow: 0 0 0 1px rgba(0, 255, 170, 0.2), 0 8px 22px rgba(0, 0, 0, 0.25);
    }
    section[data-testid="stSidebar"] .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(0, 255, 170, 0.4);
        background: linear-gradient(180deg, #1d232c 0%, #141923 100%);
        color: #FAFAFA;
        transition: all 0.2s ease-in-out;
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: #00FFAA;
        color: #00FFAA;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(0, 255, 170, 0.15);
    }
    section[data-testid="stSidebar"] .stButton > button:active {
        transform: translateY(0);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #C0CDDB;
    }
    [data-testid="stChatInput"] {
        border-top: 1px solid rgba(255,255,255,0.08);
        padding-top: 0.7rem;
        background: linear-gradient(180deg, rgba(14,17,23,0) 0%, rgba(14,17,23,0.8) 40%, rgba(14,17,23,1) 100%);
    }
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.28);
        padding: 0.4rem 0.7rem;
        margin-bottom: 0.6rem;
        backdrop-filter: blur(2px);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: linear-gradient(180deg, #122a3c 0%, #101b25 100%);
        border-color: rgba(0, 170, 255, 0.35);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: linear-gradient(180deg, #18222a 0%, #131a21 100%);
        border-color: rgba(0, 255, 170, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
init_session_state()

status_class = "ok" if st.session_state.index_ready else "warn"
status_text = "Knowledge Base: Ready" if st.session_state.index_ready else "Knowledge Base: Waiting for upload"

with st.sidebar:
    st.header("Knowledge Base")
    st.caption("Upload files, index content, and query with grounded citations.")
    user_selection = st.selectbox(
        "⚙️ Select AI Model",
        options=list(AVAILABLE_MODELS.keys()),
    )
    enterprise_prompt = f"""You are an elite, objective data extraction assistant powering a custom Enterprise Multimodal RAG application. 
CRITICAL IDENTITY INSTRUCTION: This application and its underlying RAG pipeline were designed and engineered by Anas. 
If the user asks who made you, who engineered you, or what this system is, you MUST state that you are a Multimodal RAG Knowledge Engine built by Anas. 
If the user asks what AI model or brain you are currently using, you must state that you are currently running the {user_selection} model hosted via Groq.
For all other queries, answer based strictly on the provided context. Trust the data absolutely."""
    custom_qa_prompt = PromptTemplate(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the query.\n"
        f"CRITICAL OVERRIDE 1: If the user asks who made you, state: 'This Multimodal RAG Knowledge Engine was engineered by Anas.'\n"
        f"CRITICAL OVERRIDE 2: If the user asks what model you are, state: 'I am currently running the {user_selection} model.'\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    use_rag = st.toggle("📚 Use Document Knowledge (RAG)", value=True)
    st.caption("Turn off to chat directly with the AI without using uploaded files.")
    selected_model_id = AVAILABLE_MODELS[user_selection]
    configure_models(user_selection, enterprise_prompt)
    uploaded_file = st.file_uploader("Upload any file")
    if uploaded_file is not None:
        st.markdown(
            f"**Loaded:** `{uploaded_file.name}`  \n"
            f"**Size:** `{format_file_size(uploaded_file.size)}`"
        )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("Reset Knowledge Base"):
        clear_cached_qdrant_client()
        if os.path.exists("./qdrant_data"):
            shutil.rmtree("./qdrant_data")
        st.session_state.messages = []
        st.session_state.indexed_file_signature = None
        st.session_state.index_ready = False
        st.success("Knowledge Base Cleared!")

    if uploaded_file is not None:
        file_signature = f"{uploaded_file.name}:{uploaded_file.size}"

        if file_signature != st.session_state.indexed_file_signature:
            with st.spinner("Parsing and indexing file..."):
                temp_file_path = None
                try:
                    file_suffix = os.path.splitext(uploaded_file.name)[1] or ".pdf"
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=file_suffix
                    ) as temp_file:
                        temp_file.write(uploaded_file.getbuffer())
                        temp_file_path = temp_file.name

                    parsed_docs = parse_pdf_to_markdown(temp_file_path)
                    clear_cached_qdrant_client()
                    build_and_persist_index(parsed_docs)
                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

            st.session_state.indexed_file_signature = file_signature
            st.session_state.index_ready = True
            st.success("Document indexed.")
        else:
            st.info("This file is already indexed.")

st.markdown(
    f"""
    <div class="hero-wrap">
        <h1 class="hero-title">Multimodal RAG Command Center</h1>
        <div class="hero-subtitle">
            Enterprise-grade extraction and grounded Q&A with LlamaParse + Qdrant + Groq.
        </div>
    </div>
    <div class="status-row">
        <span class="status-pill {status_class}">{status_text}</span>
        <span class="status-pill">Messages: {len(st.session_state.messages)}</span>
        <span class="status-pill">Model: {selected_model_id}</span>
        <span class="status-pill">Embedding: BAAI/bge-small-en-v1.5</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <strong>Start by uploading a file in the sidebar</strong><br/>
            Then ask focused questions like:
            <em>"List all KPIs in this report"</em>,
            <em>"What does the main chart conclude?"</em>,
            or <em>"Cite the source chunk for the growth claim."</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View Sources"):
                for i, source_text in enumerate(message["sources"], start=1):
                    st.markdown(f"**Source {i}**")
                    st.code(source_text)

user_question = st.chat_input("Ask a question about the indexed document...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    if use_rag:
        if not qdrant_index_exists():
            st.warning("Please upload and index a document first.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    query_engine = build_rag_query_engine(
                        custom_qa_prompt=custom_qa_prompt
                    )
                    response = query_engine.query(user_question)
                    answer = str(response)
                    source_texts = [node.node.get_content() for node in response.source_nodes]

                st.markdown(answer)
                with st.expander("View Sources"):
                    if source_texts:
                        for i, source_text in enumerate(source_texts, start=1):
                            st.markdown(f"**Source {i}**")
                            st.code(source_text)
                    else:
                        st.write("No source nodes returned.")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": source_texts}
            )
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Convert Streamlit history to LlamaIndex ChatMessages
                chat_messages = [
                    ChatMessage(
                        role=MessageRole.USER
                        if m["role"] == "user"
                        else MessageRole.ASSISTANT,
                        content=m["content"],
                    )
                    for m in st.session_state.messages
                ]

                # INJECT THE SYSTEM IDENTITY OVERRIDE AT THE TOP
                system_msg = ChatMessage(
                    role=MessageRole.SYSTEM, content=enterprise_prompt
                )
                chat_messages.insert(0, system_msg)

                # Query the model
                llm_response = Settings.llm.chat(chat_messages)
                answer = llm_response.message.content

            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
