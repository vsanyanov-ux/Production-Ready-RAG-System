import sys
import os
from main import query_system
from dotenv import load_dotenv

load_dotenv()

def test_query():
    print("Testing query_system...")
    try:
        question = "What is the cause of poverty according to Henry George?"
        answer, contexts = query_system(question)
        print("\nSUCCESS!")
        print(f"Answer: {answer[:100]}...")
        print(f"Contexts count: {len(contexts)}")
    except Exception as e:
        print("\nFAILED!")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_query()
