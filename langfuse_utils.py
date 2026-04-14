import os
import sys
import json
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "lib"))
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()

def get_langfuse_client():
    """
    Initialize and return the Langfuse client.
    """
    return Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )

def get_active_prompt(name: str, fallback_path: str = "config/prompts.yaml"):
    """
    Fetches the active prompt from Langfuse with a production tag.
    Falls back to local YAML if Langfuse is unavailable or prompt is missing.
    """
    lf = get_langfuse_client()
    try:
        # Try to get prompt labeled as 'production'
        prompt = lf.get_prompt(name, label="production")
        logger.info(f"Fetched prompt '{name}' (version {prompt.version}) from Langfuse.")
        return prompt.prompt
    except Exception as e:
        logger.warning(f"Could not fetch prompt '{name}' from Langfuse: {e}")
        logger.info(f"Falling back to local prompts from {fallback_path}")
        
        if not os.path.exists(fallback_path):
            logger.error(f"Fallback file {fallback_path} not found!")
            return None
            
        with open(fallback_path, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)
            # Reconstruct the template from YAML common fields
            system = prompts.get("system_prompt", "")
            qa = prompts.get("qa_template", "")
            return f"{system}\n\n{qa}"

def seed_prompts_to_langfuse(name: str = None, fallback_path: str = "config/prompts.yaml"):
    """
    Reads local YAML prompts and uploads them to Langfuse.
    """
    if name is None:
        name = os.getenv("LANGFUSE_PROMPT_NAME", "rag_qa")
        
    if not os.path.exists(fallback_path):
        print(f"Error: {fallback_path} not found.")
        return
        
    with open(fallback_path, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
        system = prompts.get("system_prompt", "")
        qa = prompts.get("qa_template", "")
        full_template = f"{system}\n\n{qa}"
        
    lf = get_langfuse_client()
    try:
        # Create or update prompt
        lf.create_prompt(
            name=name,
            prompt=full_template,
            config={
                "model": os.getenv("OPENAI_MODEL", "mistral-large"),
                "temperature": 0.0
            },
            labels=["production"] # Tag as production immediately for this demo
        )
        print(f"Successfully seeded prompt '{name}' to Langfuse.")
    except Exception as e:
        print(f"Failed to seed prompt: {e}")

def sync_dataset_to_langfuse(dataset_name: str, file_path: str = "data/golden_dataset.json"):
    """
    Uploads a local JSON dataset to Langfuse.
    """
    lf = get_langfuse_client()
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Check if dataset exists, or create it
    try:
        dataset = lf.get_dataset(dataset_name)
        print(f"Dataset '{dataset_name}' already exists in Langfuse.")
    except Exception:
        dataset = lf.create_dataset(name=dataset_name)
        print(f"Created dataset '{dataset_name}' in Langfuse.")
        
    # Add items to dataset
    for item in data:
        # Langfuse expects input and expected_output
        lf.create_dataset_item(
            dataset_name=dataset_name,
            input=item["question"],
            expected_output=item["answer"],
            metadata={"context": item["context"]}
        )
    
    print(f"Successfully synced {len(data)} items to Langfuse dataset '{dataset_name}'.")

if __name__ == "__main__":
    # Test connection and sync dataset
    print("Testing Langfuse connection...")
    try:
        client = get_langfuse_client()
        print("Connection successful.")
        
        # Seed themes/prompts
        seed_prompts_to_langfuse()
        
        # Sync the golden dataset
        sync_dataset_to_langfuse("RAG_Golden_Dataset")
    except Exception as e:
        print(f"Connection failed: {e}")
