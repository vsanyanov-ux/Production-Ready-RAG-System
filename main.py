import os
import sys
import time
import logging
from diskcache import Cache
from dotenv import load_dotenv

# Load environment variables early for Langfuse and other components
load_dotenv()

# Force UTF-8 for Windows console output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')




from loader import load_pdf, load_markdown, load_web_url
from splitter import split_documents
from vector_store import get_vector_store, add_documents_to_store
from langchain_core.documents import Document
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
from query_expansion import generate_multi_queries

def get_all_documents(store):
    """Fetch all documents from Chroma for BM25 initialization."""
    results = store.get() # Get all items from the collection
    docs = []
    for content, metadata in zip(results['documents'], results['metadatas']):
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

from langfuse import observe, propagate_attributes, Langfuse, get_client


# --- Cache Setup ---
cache = Cache("data/rag_cache")

@observe()
def perform_retrieval(queries: list[str], store):
    """Retrieval step using Multi-Query Hybrid Search (BM25 + Vector)"""
    all_docs = get_all_documents(store)
    hybrid_retriever = get_hybrid_retriever(store, all_docs, k=25) # Increased k
    
    all_results = []
    seen_content = set()
    
    for q in queries:
        print(f"Retrieving for variant: {q}")
        docs = hybrid_retriever.invoke(q)
        for d in docs:
            content_key = f"{d.page_content}_{d.metadata.get('source', '')}"
            if content_key not in seen_content:
                all_results.append(d)
                seen_content.add(content_key)
    
    print(f"Total unique docs retrieved after multi-query: {len(all_results)}")
    return all_results

@observe()
def perform_reranking(question: str, initial_results):
    """Reranking step traced separately"""
    reranker_model = get_reranker()
    # Rerank from potentially large pool down to top_n=5 for better relevance
    return rerank_documents(question, initial_results, reranker_model, top_n=5)

@observe(name="RAG-Query-Pipeline")
def query_system(question: str, session_id: str = None, model_name: str = None, **kwargs):
    """Retrieve relevant chunks for a question using hybrid search and re-ranking with Langfuse tracing."""
    
    # Sanitize input
    question = str(question).strip()
    if not question:
        print("Empty question received.")
        return None, []
    
    # Check cache BEFORE expensive retrieval/reranking
    cache_key = f"{question}_{model_name if model_name else 'default'}"
    bypass_cache = os.getenv("BYPASS_CACHE", "false").lower() == "true"
    
    if not bypass_cache:
        cached_result = cache.get(cache_key)
        if cached_result:
            print("🚀 Result retrieved from cache (instant)!")
            return cached_result["answer"], cached_result["contexts"]
    
    # Any extra kwargs (like langfuse_trace_id) are automatically handled by @observe
    with propagate_attributes(session_id=session_id, tags=["production", "mistral-rag"]) if session_id else propagate_attributes(tags=["production", "mistral-rag"]):
        store = get_vector_store()

        print("Expanding query into variants...")
        query_variants = generate_multi_queries(question, model_name=model_name)
        print(f"Generated {len(query_variants)} variants.")

        print("Executing Multi-Query Hybrid Search...")
        initial_results = perform_retrieval(query_variants, store)
        
        print(f"Retrieved {len(initial_results)} unique documents. Re-ranking against original question...")
        final_results = perform_reranking(question, initial_results)

    
        # Prepare context
        context_text = "\n\n".join([f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}" for doc in final_results])
        
        # Display results locally
        print(f"\nFinal Top Results for: {question}")
        print("-" * 50)
        contexts = []
        for i, doc in enumerate(final_results):
            source = doc.metadata.get('source', 'Unknown')
            print(f"Rank {i+1} [Source: {source}]:")
            print(doc.page_content[:200] + "...")
            print("-" * 30)
            contexts.append(f"[Source: {source}]\n{doc.page_content}")
            
        print("\nGenerating AI Answer...")
        
        # --- Step 3: Generation (with Fallback) ---
        from langchain_openai import ChatOpenAI
        from langchain_core.output_parsers import StrOutputParser
        from openai import APIConnectionError

        def get_llm(base_url, api_key, model, timeout=60):
            return ChatOpenAI(
                api_key=api_key,
                base_url=base_url,
                model=model,
                temperature=0.0,
                timeout=timeout 
            )

        # Primary attempt: Local Proxy/LiteLLM
        primary_api_key = os.getenv("OPENAI_API_KEY")
        primary_base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:4000")
        primary_model = model_name if model_name else os.getenv("OPENAI_MODEL", "mistral-large")
        
        @observe(as_type="generation")
        def generate_answer_with_metrics(chain, question, context_text, model_name, attempt="primary"):
            # Use the global decorator context helper if available, or the client
            client = get_client()
            
            # Start/Update generation with input and model name
            # Standard v4 pattern: update the current generation trace
            client.update_current_generation(
                input={"question": question, "context": context_text},
                model=model_name,
                metadata={"attempt": attempt}
            )
            
            response = chain.invoke({"context": context_text, "question": question})
            answer = response.content
            usage = response.response_metadata.get("token_usage", {})
            
            if usage:
                in_tk = usage.get("prompt_tokens", 0)
                out_tk = usage.get("completion_tokens", 0)
                # Mistral Large 3 Pricing: $0.50/1M input, $1.50/1M output
                cost_usd = (in_tk * 0.50 + out_tk * 1.50) / 1000000
                
                # Standard v4 keys for this version: usage_details and cost_details
                client.update_current_generation(
                    usage_details={"input": in_tk, "output": out_tk},
                    cost_details={"total": cost_usd},
                    output=answer
                )
                print(f"💰 [{model_name}] Cost: ${cost_usd:.6f} | Tokens: {in_tk}i/{out_tk}o")
            else:
                client.update_current_generation(output=answer)
            return answer

        try:
            print(f"Connecting to primary LLM: {primary_model} @ {primary_base_url}...")
            llm = get_llm(primary_base_url, primary_api_key, primary_model)
            
            retriever_placeholder = store.as_retriever(search_kwargs={"k": 10})
            _, prompt_temp = get_rag_chain(retriever_placeholder)
            chain = prompt_temp | llm
            
            # Using the v4.x helper for cost capture
            answer = generate_answer_with_metrics(chain, question, context_text, primary_model, attempt="primary")
            
        except (APIConnectionError, Exception) as e:
            print(f"⚠️ Primary LLM failed: {e}")
            print("🔄 Switching to Fallback (Aitunnel.ru)...")
            
            fallback_api_key = os.getenv("AITUNNEL_API_KEY")
            fallback_base_url = os.getenv("AITUNNEL_BASE_URL", "https://api.aitunnel.ru/v1")
            fallback_model = os.getenv("AITUNNEL_MODEL", "mistral-large-2512")
            
            if not fallback_api_key or fallback_api_key == "your_key_here":
                print("❌ Error: AITUNNEL_API_KEY not set. Cannot fallback.")
                return "Error: Primary failed and no fallback key provided.", contexts

            try:
                llm = get_llm(fallback_base_url, fallback_api_key, fallback_model, timeout=120)
                chain = prompt_temp | llm
                
                answer = generate_answer_with_metrics(chain, question, context_text, fallback_model, attempt="fallback")
            except Exception as fallback_e:
                print(f"❌ Fallback LLM also failed: {fallback_e}")
                return "⚠️ **Connection Error:** Both the Primary and Fallback LLMs failed to connect. Please check your internet connection, API keys, or try again later.", contexts


        # Save to cache
        cache.set(cache_key, {"answer": answer, "contexts": contexts})
        
        # Ensure traces are sent
        Langfuse().flush()
        
        return answer, contexts




if __name__ == "__main__":
    print("RAG System (Phase 2 with Re-ranking) Ready.")
    
    # You can uncomment this to index a new document:
    # ingest_data("data/Progress_and_Poverty.pdf", "pdf")
    
    # Interactive query loop
    while True:
        try:
            user_question = input("\nAsk a question (or type 'exit' to quit): ")
            if user_question.lower() in ['exit', 'quit', 'q']:
                break
            if not user_question.strip():
                continue
                
            try:
                query_system(user_question)
            except Exception as e:
                import traceback
                print("\n❌ Error during query processing:")
                traceback.print_exc()
        except KeyboardInterrupt:
            print("\nExiting...")
            break
