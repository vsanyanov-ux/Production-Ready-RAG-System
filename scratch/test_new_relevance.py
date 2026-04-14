import os
import sys
# Add parent directory to path to allow importing main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import query_system
import os
from dotenv import load_dotenv

load_dotenv()

def test_relevance():
    # Use a question that benefits from multiple perspectives
    test_question = "Explain the connection between land ownership and poverty in Henry George's view."
    
    print(f"--- Running Test Query: {test_question} ---")
    answer, contexts = query_system(test_question)
    
    print("\n[FINAL ANSWER]")
    print(answer)
    
    print("\n[CONTEXTS RETRIEVED]")
    for i, ctx in enumerate(contexts):
        print(f"\nSnippet {i+1}:")
        print(ctx[:300] + "...")

if __name__ == "__main__":
    test_relevance()
