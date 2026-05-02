import os
import re
import tempfile

import streamlit as st
import tiktoken
import extra_streamlit_components as stx

from config import load_environment
from ingest import parse_pdf_to_markdown

# Load only the keys needed for Phase 5.
load_environment(required_env_vars=("LLAMA_CLOUD_API_KEY", "GROQ_API_KEY"))

from llama_index.core import PromptTemplate, Settings, StorageContext, VectorStoreIndex
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.qdrant import QdrantVectorStore

AVAILABLE_MODELS = {
    "Llama 3.3 70B (Meta - Heavy & Complex - Best for deep reasoning)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (Meta - Light & Fast - Best for quick summaries)": "llama-3.1-8b-instant",
    "OpenAI GPT-OSS 120B (OpenAI - Massive Scale Alternative Architecture)": "openai/gpt-oss-120b",
    "Qwen3 32B (Alibaba - Phenomenal context handling)": "qwen/qwen3-32b",
}

PORTFOLIO_LIMIT = 50000


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
    if "token_counters" not in st.session_state:
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo").encode
        st.session_state.token_counters = {
            model_name: TokenCountingHandler(tokenizer=tokenizer)
            for model_name in AVAILABLE_MODELS.keys()
        }
    if "qdrant_client" not in st.session_state:
        from qdrant_client import QdrantClient

        st.session_state.qdrant_client = QdrantClient(location=":memory:")
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0


def set_active_token_counter(model_name: str) -> None:
    Settings.callback_manager = CallbackManager([st.session_state.token_counters[model_name]])


def qdrant_index_exists() -> bool:
    return st.session_state.index_ready


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


def reset_qdrant_client() -> None:
    from qdrant_client import QdrantClient

    st.session_state.pop("qdrant_client", None)
    st.session_state.qdrant_client = QdrantClient(location=":memory:")


def build_rag_query_engine(
    custom_qa_prompt: PromptTemplate, collection_name: str = "multimodal_docs"
):
    qdrant_client = st.session_state.qdrant_client
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


def build_in_memory_index(documents, collection_name: str = "multimodal_docs"):
    qdrant_client = st.session_state.qdrant_client
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
    )


def parse_thought_process(text):
    """Extracts the think block and the clean response."""
    # Look for the think block using regex
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if match:
        thought_process = match.group(1).strip()
        # Remove the think block from the main text
        clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return thought_process, clean_text
    return None, text


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


cookie_manager = stx.CookieManager(key="global_cookie_manager")

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

CRITICAL IDENTITY INSTRUCTION: This application and its underlying RAG pipeline were designed and engineered by Anas. If the user asks who made you, who engineered you, or what this system is, you MUST state that you are a Multimodal RAG Knowledge Engine built by Anas. 

SYSTEM AWARENESS: If the user asks what AI model or brain you are currently using, you must state that you are currently running the {user_selection} model hosted via Groq.

ANTI-HALLUCINATION RULE: If you are asked about current events, political figures, or real-time data that is not provided in a document, and you are not absolutely certain, you MUST explicitly state that your knowledge is limited to your training cutoff. You must not guess, fabricate, or attempt to fill in the blanks.

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
    set_active_token_counter(user_selection)
    configure_models(user_selection, enterprise_prompt)
    st.divider()
    st.caption("📊 API Usage Telemetry (Current Session)")

    PORTFOLIO_LIMIT = 50000

    # Read from cookie, but maintain a live session state so the UI doesn't lag
    cookie_val = cookie_manager.get(cookie="device_tokens")
    if "device_tokens" not in st.session_state:
        st.session_state.device_tokens = int(cookie_val) if cookie_val else 0

    # Add live session tokens to the saved state
    live_session_tokens = st.session_state.token_counters[user_selection].total_llm_token_count
    current_usage = st.session_state.device_tokens + live_session_tokens

    usage_percentage = min(current_usage / PORTFOLIO_LIMIT, 1.0)

    # Display progress bar and stats
    st.progress(usage_percentage)
    st.write(f"**Device Demo Usage:** {current_usage:,} / {PORTFOLIO_LIMIT:,}")

    # The 85% Warning Logic
    if usage_percentage >= 0.85 and usage_percentage < 1.0:
        st.warning("⚠️ Approaching portfolio demo limit.")
    uploaded_file = st.file_uploader(
        "Upload a document to the Knowledge Base",
        key=f"uploader_{st.session_state.uploader_key}"
    )
    if uploaded_file is not None:
        st.markdown(
            f"**Loaded:** `{uploaded_file.name}`  \n"
            f"**Size:** `{format_file_size(uploaded_file.size)}`"
        )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("Clear Knowledge Base"):
        # 1. Increment the uploader key to wipe the frontend widget
        st.session_state.uploader_key += 1

        # 2. Surgically delete ONLY the RAG memory state
        if "qdrant_client" in st.session_state:
            del st.session_state.qdrant_client

        # 3. Force a UI refresh to apply changes immediately
        st.rerun()

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
                    reset_qdrant_client()
                    build_in_memory_index(parsed_docs)
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
        if message["role"] == "assistant":
            thought_process, clean_text = parse_thought_process(message["content"])
            if thought_process:
                with st.expander("🧠 AI Thought Process"):
                    st.markdown(thought_process)
            st.markdown(clean_text)
            if message.get("sources"):
                with st.expander("View Sources"):
                    for i, source_text in enumerate(message["sources"], start=1):
                        st.markdown(f"**Source {i}**")
                        st.code(source_text)
        else:
            st.markdown(message["content"])

if current_usage >= PORTFOLIO_LIMIT:
    st.error("🚨 **Demo Limit Reached!** This device has exhausted its portfolio sandbox limits. To see more, let's schedule an interview!")
    prompt = st.chat_input("Device limit reached.", disabled=True)
else:
    prompt = st.chat_input("Message the Multimodal RAG Engine...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if use_rag:
        if not qdrant_index_exists():
            st.warning("Please upload and index a document first.")
        else:
            with st.chat_message("assistant"):
                answer = None
                source_texts = []
                with st.spinner("Thinking..."):
                    try:
                        query_engine = build_rag_query_engine(
                            custom_qa_prompt=custom_qa_prompt
                        )
                        response = query_engine.query(prompt)
                        answer = str(response)
                        source_texts = [node.node.get_content() for node in response.source_nodes]

                        # After successful generation, update both layers
                        live_session_tokens = st.session_state.token_counters[user_selection].total_llm_token_count
                        new_total = st.session_state.device_tokens + live_session_tokens

                        # 1. Update session state for instant UI feedback
                        st.session_state.device_tokens = new_total

                        # 2. Update cookie for refresh-persistence
                        cookie_manager.set("device_tokens", str(new_total))

                        # Reset session counter to avoid double counting
                        st.session_state.token_counters[user_selection].reset_counts()
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "429" in error_msg or "rate limit" in error_msg:
                            st.error("⚠️ **Backend API Limit Reached!** The AI provider is currently at capacity for this specific model. Please select a different model from the sidebar to continue.")
                        else:
                            st.error(f"An unexpected error occurred: {e}")

                if answer is not None:
                    thought_process, clean_text = parse_thought_process(answer)
                    if thought_process:
                        with st.expander("🧠 AI Thought Process"):
                            st.markdown(thought_process)
                    st.markdown(clean_text)
                    with st.expander("View Sources"):
                        if source_texts:
                            for i, source_text in enumerate(source_texts, start=1):
                                st.markdown(f"**Source {i}**")
                                st.code(source_text)
                        else:
                            st.write("No source nodes returned.")

            if answer is not None:
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": source_texts}
                )
    else:
        with st.chat_message("assistant"):
            answer = None
            with st.spinner("Thinking..."):
                try:
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

                    # After successful generation, update both layers
                    live_session_tokens = st.session_state.token_counters[user_selection].total_llm_token_count
                    new_total = st.session_state.device_tokens + live_session_tokens

                    # 1. Update session state for instant UI feedback
                    st.session_state.device_tokens = new_total

                    # 2. Update cookie for refresh-persistence
                    cookie_manager.set("device_tokens", str(new_total))

                    # Reset session counter to avoid double counting
                    st.session_state.token_counters[user_selection].reset_counts()
                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "rate limit" in error_msg:
                        st.error("⚠️ **Backend API Limit Reached!** The AI provider is currently at capacity for this specific model. Please select a different model from the sidebar to continue.")
                    else:
                        st.error(f"An unexpected error occurred: {e}")

            if answer is not None:
                thought_process, clean_text = parse_thought_process(answer)
                if thought_process:
                    with st.expander("🧠 AI Thought Process"):
                        st.markdown(thought_process)
                st.markdown(clean_text)

        if answer is not None:
            st.session_state.messages.append({"role": "assistant", "content": answer})
