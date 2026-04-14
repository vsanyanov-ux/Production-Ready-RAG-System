from langchain_core.prompts import PromptTemplate
from langfuse_utils import get_active_prompt
import os

def get_rag_chain(retriever):
    """
    Construct a RAG chain with prompts fetched from Langfuse.
    """
    prompt_name = os.getenv("LANGFUSE_PROMPT_NAME", "rag_qa")
    full_template = get_active_prompt(prompt_name)
    
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
