import os
import sys
import time

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "lib"))
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from dotenv import load_dotenv

load_dotenv()

print("Checkpoint 2: Importing components...")
try:
    from vector_store import get_vector_store
    from main import query_system
    print("Checkpoint 3: Imports successful.")
except Exception as e:
    print(f"FAILED during imports: {e}")
    sys.exit(1)

def test_verbose():
    try:
        print("Checkpoint 4: Initializing vector store...")
        start_time = time.time()
        store = get_vector_store()
        print(f"Checkpoint 5: Vector store initialized in {time.time() - start_time:.2f}s")
        
        print("Checkpoint 6: Testing query_system...")
        answer, contexts = query_system("Tell me about land ownership.")
        print("Checkpoint 7: Query successful.")
        print(f"Answer: {answer[:50]}...")
    except Exception as e:
        print(f"\nFAILED during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_verbose()
