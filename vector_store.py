import os
from typing import List
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

EMBEDDING_MODEL_NAME = "nomic-embed-text"
_EMBEDDINGS_CACHE = None

def get_vector_store(persist_directory: str = "./chroma_db_local"):
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} via Ollama...")
        _EMBEDDINGS_CACHE = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=_EMBEDDINGS_CACHE
    )
    return vector_store

def add_documents_to_store(vector_store: Chroma, documents: List[Document]):
    vector_store.add_documents(documents)
