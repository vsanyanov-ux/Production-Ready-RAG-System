import os
from loader import load_pdf, load_markdown, load_web_url
from splitter import split_documents
from vector_store import get_vector_store, add_documents_to_store
from rag_chain import get_rag_chain

def ingest_data(path_or_url: str, doc_type: str = "pdf"):
    """Process and index documents."""
    print(f"Loading {doc_type} from {path_or_url}...")
    
    if doc_type == "pdf":
        docs = load_pdf(path_or_url)
    elif doc_type == "md":
        docs = load_markdown(path_or_url)
    elif doc_type == "web":
        docs = load_web_url(path_or_url)
    else:
        raise ValueError("Unsupported document type")
        
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    
    store = get_vector_store()
    add_documents_to_store(store, chunks)
    print("Successfully indexed documents.")

from hybrid_retriever import get_hybrid_retriever
from reranker import get_reranker, rerank_documents

def query_system(question: str):
    """Retrieve relevant chunks for a question using hybrid search and re-ranking."""
    store = get_vector_store()
    
    print("Executing Hybrid Search (BM25 + Vector)...")
    # For Phase 2, we fetch a larger initial pool (top-k=10) for reranking
    retriever = store.as_retriever(search_kwargs={"k": 10}) 
    
    _, prompt_temp = get_rag_chain(retriever)
    
    initial_results = retriever.get_relevant_documents(question)
    print(f"Retrieved {len(initial_results)} initial documents. Re-ranking...")
    
    # Initialize reranker and re-rank the documents
    reranker_model = get_reranker()
    final_results = rerank_documents(question, initial_results, reranker_model, top_n=3)
    
    print(f"\nFinal Top Results for: {question}")
    print("-" * 50)
    for i, doc in enumerate(final_results):
        source = doc.metadata.get('source', 'Unknown')
        print(f"Rank {i+1} [Source: {source}]:")
        print(doc.page_content[:500] + "...")
        print("-" * 30)

if __name__ == "__main__":
    # Example:
    # ingest_data("data/my_doc.pdf", "pdf")
    # query_system("Specific keyword query")
    print("RAG System (Phase 2 with Re-ranking) Ready.")
