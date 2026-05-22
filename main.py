import os
from diskcache import Cache
from dotenv import load_dotenv

load_dotenv()

from loader import load_pdf, load_markdown, load_web_url
from splitter import split_documents
from vector_store import get_vector_store, add_documents_to_store
from langchain_core.documents import Document
from rag_chain import get_rag_chain
from hybrid_retriever import get_hybrid_retriever
from reranker import get_reranker, rerank_documents
from query_expansion import generate_multi_queries

# --- LANGFUSE CONFIGURATION ---
from langfuse import observe, propagate_attributes, Langfuse

# Define a masking function that scrubs all text data to protect corporate secrets
def mask_corporate_data(data):
    if isinstance(data, str):
        return "[АНОНИМИЗИРОВАНО (Корпоративная тайна)]"
    elif isinstance(data, dict):
        return {k: mask_corporate_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [mask_corporate_data(i) for i in data]
    return data

# Initialize the global Langfuse client with the masking function
# This ensures that no actual queries or corporate documents are sent to the cloud.
# We set max_retries and timeout to low values to prevent offline crashes.
try:
    langfuse_client = Langfuse(mask=mask_corporate_data, max_retries=1, timeout=5)
except Exception as e:
    print(f"Warning: Langfuse init failed: {e}")
    langfuse_client = None

cache = Cache("data/rag_cache")

def get_all_documents(store):
    results = store.get()
    docs = []
    for content, metadata in zip(results['documents'], results['metadatas']):
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

@observe()
def perform_retrieval(queries: list[str], store):
    all_docs = get_all_documents(store)
    if not all_docs:
        print("Vector store is empty!")
        return []
        
    hybrid_retriever = get_hybrid_retriever(store, all_docs, k=25)
    all_results = []
    seen_content = set()
    
    for q in queries:
        docs = hybrid_retriever.invoke(q)
        for d in docs:
            content_key = f"{d.page_content}_{d.metadata.get('source', '')}"
            if content_key not in seen_content:
                all_results.append(d)
                seen_content.add(content_key)
    return all_results

@observe()
def perform_reranking(question: str, initial_results):
    if not initial_results: return []
    reranker_model = get_reranker()
    return rerank_documents(question, initial_results, reranker_model, top_n=5)

@observe(name="RAG-Query-Pipeline")
def query_system(question: str, session_id: str = None, model_name: str = None, **kwargs):
    question = str(question).strip()
    if not question: return None, []
    
    cache_key = f"{question}_{model_name if model_name else 'default'}"
    bypass_cache = os.getenv("BYPASS_CACHE", "false").lower() == "true"
    
    if not bypass_cache:
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result["answer"], cached_result["contexts"]
    
    with propagate_attributes(session_id=session_id, tags=["production", "local-rag"]) if session_id else propagate_attributes(tags=["production", "local-rag"]):
        store = get_vector_store()

        query_variants = generate_multi_queries(question, model_name=model_name)
        initial_results = perform_retrieval(query_variants, store)
        final_results = perform_reranking(question, initial_results)

        context_text = "\n\n".join([f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}" for doc in final_results])
        
        contexts = []
        for i, doc in enumerate(final_results):
            source = doc.metadata.get('source', 'Unknown')
            contexts.append(f"[Source: {source}]\n{doc.page_content}")
            
        from langchain_ollama import ChatOllama
        from langchain_core.output_parsers import StrOutputParser

        def get_llm(model="qwen3.5:9b"):
            return ChatOllama(model=model, temperature=0.0)

        primary_model = model_name if model_name else "qwen3.5:9b"
        
        try:
            print(f"Connecting to local LLM: {primary_model} via Ollama...")
            llm = get_llm(primary_model)
            
            retriever_placeholder = store.as_retriever(search_kwargs={"k": 10})
            _, prompt_temp = get_rag_chain(retriever_placeholder)
            chain = prompt_temp | llm | StrOutputParser()
            
            response = chain.invoke({"context": context_text, "question": question})
            answer = response.content if hasattr(response, 'content') else response
            
        except Exception as e:
            print(f"⚠️ Local LLM failed: {e}")
            answer = f"⚠️ **Error:** Failed to connect to local Ollama. Exception: {e}"

        cache.set(cache_key, {"answer": answer, "contexts": contexts})
        
        if langfuse_client:
            langfuse_client.flush()
            
        return answer, contexts
