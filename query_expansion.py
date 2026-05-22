import os
from typing import List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class QueryVariations(BaseModel):
    queries: List[str] = Field(description="List of 3 search query variations")

def generate_multi_queries(question: str, model_name: str = None) -> List[str]:
    model = model_name if model_name else "qwen3.5:9b"

    llm = ChatOllama(
        model=model,
        temperature=0.1,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI language model assistant. Your task is to generate three different versions of the given user query to retrieve relevant documents from a vector database. Provide these alternative queries as a JSON list in a key named 'queries'."),
        ("user", "Original query: {question}")
    ])

    parser = JsonOutputParser(pydantic_object=QueryVariations)
    chain = prompt | llm | parser

    try:
        response = chain.invoke({"question": question})
        if isinstance(response, list):
            queries = response
        elif isinstance(response, dict):
            queries = response.get("queries", [])
        else:
            queries = []
            
        if question not in queries:
            queries.append(question)
        return list(set(queries))[:4]
    except Exception as e:
        print(f"⚠️ Query expansion failed: {e}. Falling back to original question.")
        return [question]
