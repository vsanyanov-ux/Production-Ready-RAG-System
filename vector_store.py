import os
from typing import List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Use a common open-source embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Global cache for the embedding model
_EMBEDDINGS_CACHE = None

def get_vector_store(persist_directory: str = "./chroma_db"):
    """Initialize or load the Chroma vector store with cached embeddings."""
    global _EMBEDDINGS_CACHE
    
    if _EMBEDDINGS_CACHE is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        _EMBEDDINGS_CACHE = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=_EMBEDDINGS_CACHE
    )
    return vector_store

def add_documents_to_store(vector_store: Chroma, documents: List[Document]):
    """Add documents to the vector store."""
    vector_store.add_documents(documents)
    # Chroma automatically persists if persist_directory is set in newer versions,
    # but we can explicitly call it if using older LangChain integrations.
    # vector_store.persist() 

if __name__ == "__main__":
    print(f"Vector store utility ready with model: {EMBEDDING_MODEL_NAME}")
