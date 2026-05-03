# RAG-Based Visualization Agent

> Retrieval-Augmented Generation pipeline that queries unstructured PDFs and auto-generates charts from the results.

## Role
**Full-Stack AI Developer** — Built the RAG pipeline and Streamlit visualization interface.

## Overview
Combines Retrieval-Augmented Generation (RAG) with a visualization engine. Extracts text from PDFs, generates embeddings, and lets users query documents with auto-generated charts.

## Architecture
```
PDF Upload → Text Chunking → OpenAI Embeddings → ChromaDB
                                                      ↓
                User Query → Embedding → Vector Search → Top-K Chunks
                                                              ↓
                                          LLM Generation → Chart Engine
```

## Key Features
- **PDF Ingestion** — Recursive chunking with overlap for context retention
- **Vector Store** — ChromaDB with OpenAI `text-embedding-3-small`
- **Semantic Search** — Top-K retrieval with relevance scoring
- **Auto-Visualization** — LLM selects chart type and generates Plotly figures
- **Containerized** — Docker Compose deployment with persistent volumes

## Tech Stack
`RAG` · `ChromaDB` · `OpenAI Embeddings` · `Streamlit` · `Docker` · `Python`

## Impact
- **0.89 Recall@5** and **0.91 Precision@5** at optimal chunk size (512 tokens)
- End-to-end query latency **under 1.5 seconds**

## Project Structure
```
src/
├── rag_pipeline.py    # End-to-end RAG orchestration
├── embeddings.py      # Document embedding & chunking
├── vector_store.py    # ChromaDB wrapper
├── chart_engine.py    # LLM-driven visualization
└── app.py             # Streamlit frontend
Dockerfile
docker-compose.yml
```

## License
MIT
