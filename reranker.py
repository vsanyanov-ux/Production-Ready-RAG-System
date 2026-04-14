from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# Global cache for the reranker model
_RERANKER_CACHE = None

def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Initialize the cross-encoder model with caching."""
    global _RERANKER_CACHE
    if _RERANKER_CACHE is None:
        print(f"Loading reranker model: {model_name}...")
        _RERANKER_CACHE = CrossEncoder(model_name)
    return _RERANKER_CACHE

def rerank_documents(query: str, documents: List[Document], model: CrossEncoder, top_n: int = 3) -> List[Document]:
    """
    Rerank documents based on the query using a cross-encoder.
    """
    if not documents:
        return []
        
    doc_texts = [str(doc.page_content) for doc in documents if doc.page_content]
    if not doc_texts:
        return []
        
    pairs = [(query, text) for text in doc_texts]
    scores = model.predict(pairs)
    
    # Sort docs by score
    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    
    # Return top_n documents
    return [doc for _, doc in scored_docs[:top_n]]

if __name__ == "__main__":
    print("Reranker module ready.")
