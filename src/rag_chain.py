import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
FAISS_INDEX_PATH = "faiss_index"
GROQ_MODEL       = "llama-3.1-8b-instant"
TOP_K            = 3

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

def format_docs_with_sources(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown source")
        formatted.append(f"[Source {i}] {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_rag_chain(vectorstore):
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.1,
        max_tokens=512
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    chain = (
        {
            "context":  retriever | format_docs_with_sources,
            "question": RunnablePassthrough()
        }
        | CLINICAL_RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever

def ask(chain, retriever, question):
    print(f"\nQuestion: {question}")
    print("Retrieving articles and generating answer...\n")

    answer  = chain.invoke(question)
    sources = [doc.metadata["source"] for doc in retriever.invoke(question)]

    return {
        "question": question,
        "answer":   answer,
        "sources":  sources
    }


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not found in .env file")
        exit(1)

    vectorstore      = load_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)

    test_questions = [
        "How do large language models help with clinical decision support?",
        "What are the risks of hallucination in medical AI systems?",
        "How does RAG reduce errors in medical question answering?",
    ]

    for question in test_questions:
        result = ask(chain, retriever, question)
        print("=" * 60)
        print(f"ANSWER:\n{result['answer']}")
        print(f"\nSOURCES:")
        for src in result["sources"]:
            print(f"  - {src}")
        print("=" * 60)

