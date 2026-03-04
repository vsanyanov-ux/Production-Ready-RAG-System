import json
import os
from ragas import evaluate
from ragas.metrics import faithfulness
from datasets import Dataset
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# For YandexGPT, we use the LangChain community integration
from langchain_community.chat_models import ChatYandexGPT
from ragas.llms import LangchainLLMWrapper

def load_golden_dataset(filepath: str = "data/golden_dataset.json"):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_evaluation():
    print("Starting Evaluation Pipeline with YandexGPT...")
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
        # Initialize YandexGPT
        yc_api_key = os.getenv("YC_API_KEY")
        yc_folder_id = os.getenv("YC_FOLDER_ID")
        
        if not yc_api_key or not yc_folder_id:
            raise ValueError("YC_API_KEY and YC_FOLDER_ID environment variables must be set in .env")
            
        llm = ChatYandexGPT(
            api_key=yc_api_key,
            folder_id=yc_folder_id,
            model_uri=f"gpt://{yc_folder_id}/yandexgpt/latest",
            temperature=0.0
        )
        
        ragas_llm = LangchainLLMWrapper(llm)

        result = evaluate(
            dataset,
            metrics=[faithfulness],
            llm=ragas_llm
        )
        
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
