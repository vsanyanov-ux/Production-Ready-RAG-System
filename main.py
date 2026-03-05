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
    
    initial_results = retriever.invoke(question)
    print(f"Retrieved {len(initial_results)} initial documents. Re-ranking...")
    
    # Initialize reranker and re-rank the documents
    reranker_model = get_reranker()
    final_results = rerank_documents(question, initial_results, reranker_model, top_n=3)
    
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
    
    from langchain_community.chat_models import ChatYandexGPT
    from langchain_core.output_parsers import StrOutputParser
    from dotenv import load_dotenv
    
    load_dotenv()
    
    yc_api_key = os.getenv("YC_API_KEY")
    yc_folder_id = os.getenv("YC_FOLDER_ID")
    
    if not yc_api_key or not yc_folder_id:
        print("❌ Error: YC_API_KEY and YC_FOLDER_ID not found in environment. Cannot generate answer.")
        return
        
    llm = ChatYandexGPT(
        api_key=yc_api_key,
        folder_id=yc_folder_id,
        model_uri=f"gpt://{yc_folder_id}/yandexgpt/latest",
        temperature=0.1
    )
    
    chain = prompt_temp | llm | StrOutputParser()
    
    context_text = "\n\n".join(contexts)
    answer = chain.invoke({"context": context_text, "question": question})
    
    print("\n" + "="*50)
    print("🤖 AI ANSWER:")
    print("="*50)
    print(answer)
    print("="*50 + "\n")
    
    return answer, contexts

if __name__ == "__main__":
    print("RAG System (Phase 2 with Re-ranking) Ready.")
    
    # You can uncomment this to index a new document:
    ingest_data("data/Progress_and_Poverty.pdf", "pdf")
    
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
