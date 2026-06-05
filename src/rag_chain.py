import os
import sys
import json
import time
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()

FAISS_INDEX_PATH = "faiss_index"
GROQ_MODEL       = "llama-3.1-8b-instant"
FAISS_TOP_K      = 15
MAX_HISTORY      = 4

# Load cross-encoder once at module level — not per query
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

CLINICAL_RAG_PROMPT = PromptTemplate.from_template("""
You are a clinical research assistant helping healthcare professionals
find evidence-based answers from PubMed literature.

INSTRUCTIONS:
- Answer ONLY using the context provided below
- Always cite the source article(s) you used
- If the context doesn't contain enough information, say:
  "The retrieved articles do not fully address this question."
- Do not make up information or use outside knowledge
- Keep answers concise and clinically relevant
- If the question refers to something from chat history, use that context

CHAT HISTORY:
{chat_history}

CONTEXT FROM PUBMED ARTICLES:
{context}

QUESTION: {question}

ANSWER (with citations):
""")


def load_vectorstore():
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print(f"Loading FAISS index from {FAISS_INDEX_PATH}/...")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Vector store ready!")
    return vectorstore


def build_bm25_index(vectorstore):
    """
    Build a BM25 keyword index from the same documents in FAISS.
    BM25 finds exact keyword matches — complements FAISS semantic search.
    """
    docs      = list(vectorstore.docstore._dict.values())
    tokenized = [doc.page_content.lower().split() for doc in docs]
    bm25      = BM25Okapi(tokenized)
    return bm25, docs


def hybrid_retrieve(question, vectorstore, bm25, docs, k=FAISS_TOP_K):
    """
    Hybrid retrieval = FAISS semantic search + BM25 keyword search.
    Merges both result lists and deduplicates.
    """
    # FAISS semantic search
    faiss_results = vectorstore.similarity_search(question, k=k)

    # BM25 keyword search
    tokens       = question.lower().split()
    bm25_scores  = bm25.get_scores(tokens)
    top_bm25_idx = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]
    bm25_results = [docs[i] for i in top_bm25_idx]

    # Merge and deduplicate
    seen     = set()
    combined = []
    for doc in faiss_results + bm25_results:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)

    return combined


def rerank(question, candidates, top_k=3):
    """
    Cross-encoder reranking.
    Scores each (question, chunk) pair together for precise relevance.
    """
    print("  Reranking candidates...")
    pairs  = [(question, doc.page_content) for doc in candidates]
    scores = RERANKER.predict(pairs)

    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )
    return [doc for _, doc in ranked[:top_k]]


def format_docs_with_sources(docs):
    """Format retrieved chunks with source labels for the prompt."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown source")
        formatted.append(f"[Source {i}] {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def format_chat_history(history):
    """
    Format conversation history for prompt injection.
    Keeps last MAX_HISTORY exchanges only.
    """
    if not history:
        return "No previous conversation."
    recent = history[-MAX_HISTORY:]
    lines  = []
    for turn in recent:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)


def build_rag_chain(vectorstore):
    """Build the RAG chain — returns chain, bm25, docs."""
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.1,
        max_tokens=512
    )
    bm25, docs = build_bm25_index(vectorstore)
    chain      = CLINICAL_RAG_PROMPT | llm | StrOutputParser()
    return chain, bm25, docs


def ask(chain_tuple, question, history=None):
    """
    Full pipeline:
      1. Hybrid retrieval (FAISS + BM25)
      2. Cross-encoder reranking
      3. Inject chat history
      4. Generate grounded answer with citations
    """
    chain, vectorstore, bm25, docs = chain_tuple

    if history is None:
        history = []

    print(f"\nQuestion: {question}")
    print("Running hybrid retrieval...")

    candidates   = hybrid_retrieve(question, vectorstore, bm25, docs)
    print(f"  Retrieved {len(candidates)} candidates")

    top_docs     = rerank(question, candidates)
    print(f"  Reranked to top {len(top_docs)}")

    context      = format_docs_with_sources(top_docs)
    chat_history = format_chat_history(history)

    answer = chain.invoke({
        "context":      context,
        "chat_history": chat_history,
        "question":     question
    })

    sources = [doc.metadata["source"] for doc in top_docs]

    return {
        "question": question,
        "answer":   answer,
        "sources":  sources
    }


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not found in .env file")
        exit(1)

    vectorstore       = load_vectorstore()
    chain_tuple       = build_rag_chain(vectorstore)
    chain, bm25, docs = chain_tuple
    history           = []

    questions = [
        "What are the risks of hallucination in medical AI?",
        "How can those risks be mitigated?",
        "Which of those methods is most practical?",
    ]

    for question in questions:
        result = ask(
            (chain, vectorstore, bm25, docs),
            question,
            history
        )
        print("=" * 60)
        print(f"ANSWER:\n{result['answer']}")
        print(f"\nSOURCES:")
        for src in result["sources"]:
            print(f"  - {src}")
        print("=" * 60)

        history.append({
            "question": question,
            "answer":   result["answer"]
        })