import yaml
from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document

def get_hybrid_retriever(vector_store: Chroma, documents: List[Document], k: int = 4):
    """
    Create an ensemble retriever combining BM25 and Vector search.
    """
    if not documents:
        return vector_store.as_retriever(search_kwargs={"k": k})

    # Initialize BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    
    # Vector store retriever
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    # Ensemble them (Reciprocal Rank Fusion by default in LangChain)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever

def load_prompts(config_path: str = "config/prompts.yaml"):
    """Load prompts from YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    print("Hybrid retriever module ready.")
