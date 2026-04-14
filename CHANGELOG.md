# Changelog

All notable changes to this project will be documented in this file.

## [v2.0.0] - 2026-04-14
### Added
- **Multi-Query Expansion**: Rewrites questions from multiple perspectives to drastically increase context retrieval recall (`query_expansion.py`).
- **Hybrid Search Architecture**: Combined keyword search (BM25) and dense vector similarity search with Reciprocal Rank Fusion (`hybrid_retriever.py`).
- **CrossEncoder Reranking**: Re-sorts fetched documents using a second-stage local cross-encoder for maximal relevance (`reranker.py`).
- **Langfuse Observability**: Full end-to-end tracing and tracking of queries, generations, and context retrieval (`langfuse_utils.py`).
- **Automated RAGAS Evaluation**: Automated script that calculates Faithfulness and Answer Relevancy metrics, saving them directly to Langfuse (`evaluate_langfuse.py`).
- **Mistral Large Integration**: Migrated standard chat routing through a reliable local proxy using Mistral Large.
- Unified evaluation logic and detailed README presentation architecture diagram.

## [v1.0.0] - 2026-04-13
### Added
- Initial stable release.
- Core document loaders, chunkers, and ChromaDB vector store.
- Langchain LCEL rag pipeline.
- Streamlit User Interface.
