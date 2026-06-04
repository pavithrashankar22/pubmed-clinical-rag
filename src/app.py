import os
import sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))
from rag_chain import load_vectorstore, build_rag_chain, ask

st.set_page_config(
    page_title="Clinical Q&A Bot — PubMed RAG",
    page_icon="🏥",
    layout="wide"
)

if "question" not in st.session_state:
    st.session_state["question"] = ""

@st.cache_resource
def get_chain():
    vs = load_vectorstore()
    chain, retriever = build_rag_chain(vs)
    return chain, retriever

st.markdown("## 🏥 Clinical Q&A Bot")
st.markdown("*Powered by PubMed RAG · Evidence-based answers with citations*")
st.divider()

with st.sidebar:
    st.header("About")
    st.markdown("""
    This bot answers clinical questions using **RAG**:
    1. Your question is embedded into a vector
    2. FAISS finds the most similar PubMed abstracts
    3. Those abstracts are passed to the LLM as context
    4. The LLM generates a grounded answer with citations
                
    **Stack:** LangChain · FAISS · HuggingFace · Groq · Streamlit
    """)
    st.divider()
    st.header("Try These Questions")
    sample_qs = [
        "How do LLMs help with clinical decision support?",
        "What are the risks of hallucination in medical AI?",
        "How does RAG reduce errors in medical question answering?",
        "How accurate is AI for clinical trial matching?",
    ]
    for q in sample_qs:
        if st.button(q, use_container_width=True):
            st.session_state["question"] = q

# ← this is OUTSIDE the sidebar block
question = st.text_input(
    "Ask a clinical question:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. How do LLMs reduce hallucination in clinical settings?",
)

ask_btn = st.button("Ask", type="primary")
st.warning("Research demo only. Not for clinical use.")

if ask_btn and question.strip():
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not found in .env file.")
    else:
        with st.spinner("Searching PubMed literature..."):
            chain, retriever = get_chain()
            result = ask(chain, retriever, question)

        st.subheader("Answer")
        st.success(result["answer"])

        st.subheader("Sources Retrieved")
        for src in result["sources"]:
            st.markdown(f"- {src}")

        with st.expander("View raw retrieved chunks"):
            for i, doc in enumerate(retriever.invoke(question), 1):
                st.markdown(f"**Chunk {i}** — {doc.metadata['source']}")
                st.text(doc.page_content)
                st.divider()