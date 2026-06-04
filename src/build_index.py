import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DATA_PATH        = "data/abstracts.json"
FAISS_INDEX_PATH = "faiss_index"
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50

def load_articles(path):
    with open(path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    print(f"Loaded {len(articles)} articles from {path}")
    return articles


def articles_to_documents(articles):
    docs = []
    for article in articles:
        content = f"Title: {article['title']}\n\nAbstract: {article['abstract']}"
        doc = Document(
            page_content=content,
            metadata={
                "pmid":   article["pmid"],
                "title":  article["title"],
                "year":   article["year"],
                "source": article["source"],
            }
        )
        docs.append(doc)
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks")
    return chunks

def build_and_save_index(chunks):
    print("\nLoading embedding model (downloads ~80MB first time)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Building FAISS index — embedding all chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"Index saved to {FAISS_INDEX_PATH}/")
    return vectorstore

def test_retrieval(vectorstore):
    print("\n── Retrieval Test ──────────────────────")
    query = "How do LLMs help with clinical decision making?"
    print(f"Query: '{query}'")
    results = vectorstore.similarity_search(query, k=3)
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Source : {doc.metadata['source']}")
        print(f"  Content: {doc.page_content[:150]}...")


if __name__ == "__main__":
    articles  = load_articles(DATA_PATH)
    docs      = articles_to_documents(articles)
    chunks    = chunk_documents(docs)
    vs        = build_and_save_index(chunks)
    test_retrieval(vs)
    print("\nIndex build complete! Ready for RAG.")