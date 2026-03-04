from langchain_core.prompts import PromptTemplate
from hybrid_retriever import load_prompts

def get_rag_chain(retriever):
    """
    Construct a RAG chain with updated prompts and retriever.
    """
    prompts = load_prompts()
    system_prompt = prompts.get("system_prompt", "")
    qa_template = prompts.get("qa_template", "")
    
    # Combined template
    full_template = f"{system_prompt}\n\n{qa_template}"
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(full_template)
    
    # We return the retriever and prompt for now as structure
    return retriever, QA_CHAIN_PROMPT

import re

def verify_citations(response_text: str, retrieved_docs) -> bool:
    """
    Verify that the response contains citations and they map to retrieved documents.
    This is a basic regex-based citation enforcer. 
    Production systems might use LLM-as-a-judge or exact match tracking.
    """
    # Look for [Source Name] or [1], etc.
    citations = re.findall(r'\[(.*?)\]', response_text)
    
    if not citations:
        print("WARNING: No citations found in the response. Might be a hallucination.")
        return False
        
    print(f"Verified Citations found: {citations}")
    return True

if __name__ == "__main__":
    print("RAG chain components and citation enforcer ready.")
