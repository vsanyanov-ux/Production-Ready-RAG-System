import json
import os
import sys

# Force UTF-8 for Windows console output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from ragas import evaluate
from ragas.metrics import faithfulness
from datasets import Dataset
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# For Mistral (via OpenAI compatible API), we use langchain-openai
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

def load_golden_dataset(filepath: str = "data/golden_dataset.json"):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_evaluation():
    print("Starting Evaluation Pipeline with Mistral Large model...")
    data = load_golden_dataset()
    
    # Prepare data for Ragas expected format
    eval_data = {
        "question": [item["question"] for item in data],
        "answer": [item["answer"] for item in data],
        "contexts": [item["context"] for item in data],
    }
    
    dataset = Dataset.from_dict(eval_data)
    
    print("Evaluating Faithfulness...")
    try:
        # Initialize Mistral model
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.mistral.ai/v1")
        openai_model = os.getenv("OPENAI_MODEL", "mistral-large-latest")
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY missing in .env")
            
        llm = ChatOpenAI(
            api_key=openai_api_key,
            base_url=openai_base_url,
            model=openai_model,
            temperature=0.0,
            timeout=180
        )
        
        ragas_llm = LangchainLLMWrapper(llm)

        result = evaluate(
            dataset,
            metrics=[faithfulness],
            llm=ragas_llm
        )
        
        print("Raw Evaluation Result:", result)
        
        # Determine pass/fail based on a threshold
        try:
            import ast
            score_dict = ast.literal_eval(str(result))
            score = float(score_dict.get("faithfulness", 0.0))
        except Exception:
            try:
                score = float(result["faithfulness"])
            except Exception:
                score = 0.0
                
        import math
        if math.isnan(score):
            score = 0.0
            
        threshold = 0.85
        
        if score >= threshold:
            print(f"✅ PASSED: Faithfulness score ({score:.2f}) meets the {threshold} threshold.")
            return 0
        else:
            print(f"❌ FAILED: Faithfulness score ({score:.2f}) is below the {threshold} threshold.")
            return 1
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Evaluation Failed due to error: {repr(e)}")
        print("Note: Ensure you have set OPENAI_API_KEY and OPENAI_BASE_URL variables in .env")
        return 1

if __name__ == "__main__":
    exit_code = run_evaluation()
    os._exit(exit_code)
