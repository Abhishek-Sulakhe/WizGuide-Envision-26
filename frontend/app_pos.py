import os
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pos_weighted_rag import PipelineConfig, RAGPipeline, load_and_chunk_docs


st.set_page_config(
    page_title="WizGuide",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0612 0%, #12082a 100%);
        color: #e8dcc8;
    }
    .stChatMessage {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,215,0,0.2) !important;
        border-radius: 12px !important;
    }
    .stChatInputContainer {
        border-top: 1px solid rgba(255,215,0,0.3) !important;
    }
    section[data-testid="stSidebar"] {
        background: rgba(10,6,18,0.95) !important;
        border-right: 1px solid rgba(255,215,0,0.2) !important;
    }
    .stButton button {
        background: rgba(255,215,0,0.1);
        border: 1px solid rgba(255,215,0,0.5);
        color: #ffd700;
        border-radius: 20px;
    }
    .stButton button:hover {
        background: rgba(255,215,0,0.2);
        border-color: #ffd700;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_docs():
    return load_and_chunk_docs(str(ROOT_DIR / "data"), chunk_size=800)


@st.cache_resource
def load_pipeline(api_key: str, num_chunks: int):
    os.environ["GOOGLE_API_KEY"] = api_key
    os.environ["GEMINI_API_KEY"] = api_key
    docs = load_docs()
    cfg = PipelineConfig(
        top_k_matches=num_chunks,
        enable_expansion=False,
    )
    return RAGPipeline(docs, cfg)


def get_answer(question: str, api_key: str, num_chunks: int = 3):
    pipeline = load_pipeline(api_key, num_chunks)
    result = pipeline.query(question)
    sources = [
        (
            f"Source: {doc.metadata.get('source', 'Unknown')}, "
            f"chunk {doc.metadata.get('chunk_index', '?')}\n\n"
            f"{doc.page_content}"
        )
        for doc in result.retrieved_docs
    ]
    return result.answer, sources, result


with st.sidebar:
    st.markdown("# WizGuide")
    st.markdown("*Your Harry Potter assistant*")
    st.divider()
    st.markdown("### API Key")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Paste your Gemini key here",
        help="Get a key at makersuite.google.com",
    )

    st.divider()
    st.markdown("### Settings")
    num_chunks = st.slider("Chunks to retrieve", 1, 10, 3)
    show_sources = st.toggle("Show source chunks", value=True)

    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_submitted_question = None
        st.rerun()

    st.divider()
    st.markdown("#### Suggested Questions")
    sample_qs = [
        "Who is Albus Dumbledore?",
        "Tell me about Hogwarts houses",
        "What is a Horcrux?",
        "Describe Harry's first year",
    ]
    for q in sample_qs:
        if st.button(q, key=q):
            st.session_state.pending_question = q


col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.markdown(
        "<h1 style='text-align:center; "
        "font-family:Georgia,serif; color:#ffd700; "
        "text-shadow: 0 0 20px rgba(255,215,0,0.5);'>"
        "WizGuide</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#a89bc0;'>"
        "Ask me anything about the Wizarding World"
        "</p>",
        unsafe_allow_html=True,
    )
    st.divider()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "last_submitted_question" not in st.session_state:
    st.session_state.last_submitted_question = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            with st.expander("Source Passages"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Passage {i}:**")
                    st.markdown(f"> {src}")
                    st.divider()


user_input = st.chat_input("Ask about Harry Potter...")

if st.session_state.pending_question:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None

if user_input:
    if st.session_state.last_submitted_question == user_input:
        st.stop()
    st.session_state.last_submitted_question = user_input

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching the Wizarding archives..."):
            try:
                if not api_key:
                    answer = "Please enter your Gemini API key in the sidebar."
                    sources = []
                    result = None
                else:
                    answer, sources, result = get_answer(user_input, api_key, num_chunks)

                st.markdown(answer)
                if result is not None:
                    st.caption(
                        "Generation calls for this query: "
                        f"{result.generation_call_count} | "
                        f"Retrieval: {result.latency_ms.get('matching_ms', 0)} ms | "
                        f"Generation: {result.latency_ms.get('generation_ms', 0)} ms"
                    )
                if sources and show_sources:
                    with st.expander("Source Passages"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**Passage {i}:**")
                            st.markdown(f"> {src}")
                            st.divider()

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )
            except Exception as exc:
                err = f"Error: {exc}"
                st.error(err)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": err,
                        "sources": [],
                    }
                )
            finally:
                st.session_state.last_submitted_question = None
