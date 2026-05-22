import os
from vector_store import get_vector_store
print("Getting vector store...")
db = get_vector_store("chroma_db_local")
print("Running search...")
try:
    results = db.similarity_search("отпуск", k=2)
    print("Results:", results)
except Exception as e:
    print("Error:", e)
