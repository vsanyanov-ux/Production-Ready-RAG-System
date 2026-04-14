import time
import sys
import os

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "lib"))
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)


def test_import(module_name):
    print(f"Testing import of {module_name}...", end=" ", flush=True)
    start = time.time()
    try:
        __import__(module_name)
        print(f"OK ({time.time() - start:.2f}s)", flush=True)
    except Exception as e:
        print(f"FAILED: {e}", flush=True)

if __name__ == "__main__":
    test_import("dotenv")
    test_import("langfuse")
    test_import("langchain_huggingface")
    test_import("sentence_transformers")
    test_import("chromadb")
    test_import("langchain_chroma")
    test_import("yandexcloud")
    test_import("pypdf")
    print("All heavy imports checked.")
