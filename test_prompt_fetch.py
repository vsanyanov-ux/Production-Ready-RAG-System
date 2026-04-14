from langfuse_utils import get_active_prompt
import os
from dotenv import load_dotenv

load_dotenv()

prompt_name = os.getenv("LANGFUSE_PROMPT_NAME", "rag_qa")
print(f"Testing fetch for prompt: {prompt_name}")

try:
    content = get_active_prompt(prompt_name)
    print("--- FETCHED CONTENT START ---")
    print(content[:500] + "...")
    print("--- FETCHED CONTENT END ---")
    
    if "production-grade RAG assistant" in content:
        print("VERIFICATION SUCCESS: Content matches expected template.")
    else:
        print("VERIFICATION FAILURE: Content does not match expected template.")
except Exception as e:
    print(f"VERIFICATION ERROR: {e}")
