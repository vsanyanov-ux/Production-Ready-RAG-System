# Production-Ready RAG System

A robust, modular, and production-ready Retrieval-Augmented Generation (RAG) backend. This project goes beyond basic prototypes by implementing advanced retrieval techniques (Hybrid Search + Reciprocal Rank Fusion), re-ranking (CrossEncoder), an automated evaluation pipeline (LLM-as-a-Judge using Ragas and YandexGPT), and a beautiful Streamlit chat UI.

## 🌟 Key Features

*   **Multi-Format Document Ingestion:** Supports loading context from PDF files, Markdown documents, and Web URLs.
*   **Vector Content Storage:** Uses local ChromaDB combined with standard `SentenceTransformers` embeddings.
*   **Hybrid Search (Lexical + Semantic):** Combines standard BM25 keyword search with dense vector search to retrieve documents accurately even using specific IDs, acronyms, or misspellings.
*   **Reciprocal Rank Fusion (RRF):** Custom robust implementation to mathematically merge and normalize search results from BM25 and Vector retrievers.
*   **Cross-Encoder Re-Ranking:** Implements a second-stage retrieval pipeline using MS MARCO MiniLM cross-encoder to accurately score and re-order the retrieved chunks for maximum relevance to the user's query.
*   **Citation & Prompt Management:** Strict system prompts managed externally (`config/prompts.yaml`) forcing the LLM to ground its answers exclusively in retrieved contexts and cite sources.
*   **Automated Evaluation Pipeline (CI/CD Ready):** Includes a `golden_dataset.json` and a script (`evaluate.py`) that utilizes the **Ragas** framework to evaluate the Faithfulness of the system using YandexGPT, natively returning exit codes suitable for GitHub Actions.
*   **Conversational Web UI:** A beautiful web interface built with **Streamlit** (`app.py`), featuring chat history, AI typing indicators, and expandable source context wrappers.
*   **Observability & Tracing:** Full integration with **LangSmith** for deep visibility into LLM calls, token usage, latency, and retrieval performance without any code changes.

## 🛠️ Tech Stack
*   **Frameworks:** LangChain, HuggingFace Transformers, Streamlit
*   **Databases:** ChromaDB
*   **Algorithms:** BM25 (Rank-BM25), RRF, CrossEncoder
*   **Evaluation & Observability:** Ragas, YandexGPT API, LangSmith
*   **CI/CD:** GitHub Actions

## 📂 Project Structure & File Index

* **`app.py`** — The Streamlit graphical web interface. Run this to chat with your documents in the browser.
* **`main.py`** — The backend system core. Exports the `query_system` and `ingest_data` functions to the frontend.
* **`loader.py`** — Parsers for loading content from PDFs, Markdown files, and Web URLs.
* **`splitter.py`** — Text chunking logic using `RecursiveCharacterTextSplitter`. Optimized for 1200 character chunks with 200 overlap.
* **`vector_store.py`** — Manages the local ChromaDB vector database and text embeddings.
* **`hybrid_retriever.py`** — Implements Hybrid Search (BM25 + Semantic Vector) with Reciprocal Rank Fusion (RRF).
* **`reranker.py`** — Implements second-stage retrieval using a HuggingFace `CrossEncoder` to re-order the retrieved chunks by strict relevance.
* **`rag_chain.py`** — Connects the prompt and the LLM using LangChain Expression Language (LCEL).
* **`evaluate.py`** — Automated evaluation pipeline script using the **Ragas** framework to score AI responses for Faithfulness.
* **`config/prompts.yaml`** — Externalized management of the System Prompt and generation rules.
* **`data/golden_dataset.json`** — The ground-truth testing dataset (Questions, Contexts, Answers) used for validation.

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the dependencies:
```bash
git clone <your-repo-url>
cd RAG
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory and add your Yandex Cloud and LangSmith credentials:
```env
# Required for LangChain LLM generation
YC_API_KEY=your_yandex_api_key
YC_FOLDER_ID=your_yandex_folder_id

# Required for LangSmith full-stack tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT="YandexGPT-RAG"
```

### 3. Usage (Web Interface)

The easiest way to interact with the system is via the beautiful Streamlit UI:
```bash
streamlit run app.py
```
This will launch a conversational interface on `http://localhost:8501`.

### 4. Running the Evaluation

To check the system's performance and ensure the LLM isn't hallucinating, run the evaluation script against the Golden Dataset:
```bash
python evaluate.py
```
*Note: This utilizes YandexGPT as an LLM judge to score the Faithfulness metric and ensure answers meet the 0.85 strict threshold.*

## 📈 System Architecture Pipeline
1. **Load -> Chunk -> Embed -> ChromaDB**
2. **User Query -> BM25 Retriever & Vector Retriever -> RRF Normalization**
3. **Top 10 Chunks -> CrossEncoder Re-Ranking -> Top 3 Chunks**
4. **Top 3 Chunks + Prompt -> ChatYandexGPT -> Streamlit Interface**
5. **Background Logging -> LangSmith Trace Export**

```mermaid
graph TD
    %% Define Styles
    classDef ui fill:#4a148c,stroke:#ab47bc,stroke-width:2px,color:#fff;
    classDef core fill:#1565c0,stroke:#64b5f6,stroke-width:2px,color:#fff;
    classDef data fill:#2e7d32,stroke:#81c784,stroke-width:2px,color:#fff;
    classDef llm fill:#e65100,stroke:#ffb74d,stroke-width:2px,color:#fff;
    classDef config fill:#616161,stroke:#e0e0e0,stroke-width:2px,color:#fff;

    subgraph "Поток 1: Загрузка файлов (Ingestion)"
        loader[("loader.py\n(Чтение PDF/Web)")]:::data --> splitter[("splitter.py\n(Нарезка на чанки)")]:::data
        splitter --> vs_add[("vector_store.py\n(Превращение в векторы)")]:::data
        vs_add --> chroma[("ChromaDB\n(База данных)")]:::data
    end

    subgraph "Поток 2: Общение с ботом (Query System)"
        user["Пользователь"] --> app[/"app.py\n(Streamlit UI)"/]:::ui
        app -- Вопрос --> main["main.py\n(Главный контроллер)"]:::core
        
        main -- 1. Запрос 10 кусков --> hybrid["hybrid_retriever.py\n(Hybrid + RRF)"]:::core
        hybrid --> vs_read["vector_store.py\n(Векторный поиск)"]:::data
        vs_read --> chroma
        
        main -- 2. Фильтрация до 3 кусков --> reranker["reranker.py\n(CrossEncoder)"]:::core
        
        main -- 3. Сборка промпта --> ragchain["rag_chain.py\n(Промпт + Цепочка)"]:::core
        
        prompts[/"config/prompts.yaml\n(Инструкции)"/]:::config -.-> ragchain
        
        ragchain -- 4. Запрос + Топ-3 Куска --> yandex(["YandexGPT API"]):::llm
        yandex -- Ответ --> main
        main -- Итоговый ответ + Источники --> app
    end
    
    %% Evaluation Pipeline
    subgraph "Поток 3: Тестирование перед релизом"
        golden[/"data/golden_dataset.json\n(Эталонные вопросы)"/]:::config -.-> eval["evaluate.py\n(Оценщик Ragas)"]:::core
        eval --> yandex_judge(["YandexGPT (Судья)"]):::llm
        eval -.-> github["GitHub Actions CI/CD"]:::ui
    end

