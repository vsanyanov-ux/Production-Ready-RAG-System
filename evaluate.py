import json
import os
from ragas import evaluate
from ragas.metrics import faithfulness
from datasets import Dataset
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# For GigaChat, we use the LangChain integration
from langchain_gigachat.chat_models import GigaChat
from ragas.llms import LangchainLLMWrapper

def load_golden_dataset(filepath: str = "data/golden_dataset.json"):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_evaluation():
    print("Starting Evaluation Pipeline with GigaChat...")
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
        # Initialize GigaChat
        # It requires GIGACHAT_CREDENTIALS environment variable
        llm = GigaChat(
            verify_ssl_certs=False, 
            timeout=6000, 
            model="GigaChat-Pro" # or standard GigaChat
        )
        
        # Wrap it for Ragas
        ragas_llm = LangchainLLMWrapper(llm)

        result = evaluate(
            dataset,
            metrics=[faithfulness],
            llm=ragas_llm
        )
        print("\nEvaluation Results:")
        print(result)
        
        # Determine pass/fail based on a threshold
        score = result.get('faithfulness', 0)
        threshold = 0.85
        
        if score >= threshold:
            print(f"✅ PASSED: Faithfulness score ({score:.2f}) meets the {threshold} threshold.")
            return 0
        else:
            print(f"❌ FAILED: Faithfulness score ({score:.2f}) is below the {threshold} threshold.")
            return 1
            
    except Exception as e:
        print(f"Evaluation Failed due to error: {e}")
        print("Note: Ensure you have set GIGACHAT_CREDENTIALS environment variable.")
        return 1

if __name__ == "__main__":
    exit_code = run_evaluation()
    os._exit(exit_code)
