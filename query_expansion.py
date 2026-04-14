import os
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class QueryVariations(BaseModel):
    queries: List[str] = Field(description="List of 3 search query variations")

def generate_multi_queries(question: str, model_name: str = None) -> List[str]:
    """
    Generate 3 variations of the input question to improve retrieval recall.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:4000")
    model = model_name if model_name else os.getenv("OPENAI_MODEL", "mistral-large")

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.1 # Low temperature for consistency but some variety
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI language model assistant. Your task is to generate three different versions of the given user query to retrieve relevant documents from a vector database. By generating multiple perspectives on the user query, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative queries as a JSON list."),
        ("user", "Original query: {question}")
    ])

    parser = JsonOutputParser(pydantic_object=QueryVariations)
    chain = prompt | llm | parser

    try:
        response = chain.invoke({"question": question})
        queries = response.get("queries", [])
        # Ensure we always include the original question
        if question not in queries:
            queries.append(question)
        return list(set(queries))[:4] # Max 4 including original
    except Exception as e:
        print(f"⚠️ Query expansion failed: {e}. Falling back to original question.")
        return [question]

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    test_q = "What are the core principles of Henry George's philosophy?"
    print(f"Testing multi-query for: {test_q}")
    results = generate_multi_queries(test_q)
    for i, q in enumerate(results):
        print(f"{i+1}. {q}")
