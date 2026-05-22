import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("Initializing PyTorch in main thread...")
import torch
import sentence_transformers
import chromadb

import sys
from streamlit.web import cli

if __name__ == "__main__":
    print("Starting Streamlit...")
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(cli.main())
