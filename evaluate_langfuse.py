import os
import sys
import json



from langfuse import Langfuse
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from dotenv import load_dotenv
from main import query_system
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

load_dotenv()


def run_langfuse_evaluation(dataset_name: str = "RAG_Golden_Dataset"):
    """
    Runs evaluation on a Langfuse dataset and logs scores back.
    """
    print(f"🚀 Starting Evaluation for dataset: {dataset_name} using Mistral Large (via Proxy)...")
    lf = Langfuse()
    
    try:
        dataset = lf.get_dataset(dataset_name)
    except Exception as e:
        print(f"❌ Error: Dataset '{dataset_name}' not found. Run langfuse_utils.py first.")
        return

    # Initialize LLM for RAGAS evaluation (using Aitunnel as robust fallback/primary)
    api_key = os.getenv("AITUNNEL_API_KEY")
    base_url = os.getenv("AITUNNEL_BASE_URL", "https://api.aitunnel.ru/v1")
    model = os.getenv("AITUNNEL_MODEL", "mistral-large-2512")
    
    if not api_key:
        # Fallback to standard OpenAI envs if AITUNNEL is not set
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:4000")
        model = os.getenv("OPENAI_MODEL", "mistral-large")

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.0,
        max_retries=5,
        timeout=180
    )
    ragas_llm = LangchainLLMWrapper(llm)

    from ragas.metrics import faithfulness, answer_relevancy
    metrics = [faithfulness, answer_relevancy]

    results = []
    
    for item in dataset.items:
        print(f"\nEvaluating Item ID: {item.id} | Question: {item.input[:50]}...")
        
        try:
            # 1. Run inference using our traced query_system
            import uuid
            trace_id = uuid.uuid4().hex
            
            # Pass our generated trace_id to ensure linkage
            answer, contexts = query_system(
                item.input, 
                session_id=f"eval-{dataset_name}-{item.id}",
                langfuse_trace_id=trace_id
            )
            
            if not answer:
                print(f"⚠️ Warning: Model returned no answer for item {item.id}")
                continue

            # 2. Prepare data for Ragas
            sample = {
                "question": [item.input],
                "answer": [answer],
                "contexts": [[c for c in contexts]],
                "ground_truth": [item.expected_output or ""]
            }
            dataset_ragas = Dataset.from_dict(sample)
            
            # 3. Calculate metrics
            from langchain_huggingface import HuggingFaceEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.run_config import RunConfig
            
            hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

            # Robust evaluation configuration for Mistral
            run_config = RunConfig(max_retries=3, timeout=180)

            eval_result = evaluate(
                dataset_ragas,
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                run_config=run_config,
                raise_exceptions=True
            )
            
            print(f"Metrics results: {eval_result}")
            
            # 4. Log scores to both the Dataset Item and the Trace in Langfuse
            import math
            for metric in metrics:
                metric_name = metric.name
                try:
                    score = eval_result[metric_name]
                    
                    # Robust numeric check
                    if isinstance(score, (int, float)) and not math.isnan(score):
                         lf.create_score(
                            name=metric_name,
                            value=float(score),
                            trace_id=trace_id,
                            comment="RAGAS Evaluation (Mistral-Large-2512)"
                        )
                    elif isinstance(score, list) and len(score) > 0:
                        # Sometimes Ragas returns a list, extract the first value
                        inner_score = score[0]
                        if isinstance(inner_score, (int, float)) and not math.isnan(inner_score):
                            lf.create_score(
                                name=metric_name,
                                value=float(inner_score),
                                trace_id=trace_id,
                                comment="RAGAS Evaluation (Mistral-Large-2512)"
                            )
                    else:
                        print(f"⚠️ Skipping score for {metric_name} (Value: {score})")
                        
                except Exception as inner_e:
                    print(f"Skipping metric {metric_name} due to: {inner_e}")
                
            results.append(eval_result)
        except Exception as e:
            print(f"❌ Failed to evaluate item {item.id}: {e}")
            continue

    print("\n✅ Evaluation Complete.")
    print(f"Check your results here: {os.getenv('LANGFUSE_HOST')}/datasets/{dataset_name}")
    
    # Ensure all traces and scores are sent before exiting
    lf.flush()

if __name__ == "__main__":
    run_langfuse_evaluation()
