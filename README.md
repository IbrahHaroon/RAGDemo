# Simple RAG Chatbot (Ollama)

A minimal Retrieval-Augmented Generation (RAG) chatbot using **Ollama**, **NumPy**, and an **in-memory vector database**.

## What it does
- Loads text data (`cat-facts.txt`)
- Generates embeddings using a local embedding model
- Stores embeddings in memory
- Retrieves the most relevant chunks via cosine similarity
- Uses retrieved context to answer questions with an LLM

## Requirements
- Python 3.9+
- Ollama installed and running
- NumPy

## Models Used
- Embeddings: `bge-base-en-v1.5`
- LLM: `Llama-3.2-1B-Instruct`

## How to Run
```bash
pip install numpy
python main.py
