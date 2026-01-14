# RAG Project

> [!NOTE]
> **Work in Progress**: This project is under active development.

A Retrieval-Augmented Generation (RAG) system built with LangChain, Ollama, ChromaDB, and OpenRouter. This project allows you to ingest text documents, create vector embeddings locally, and perform semantic retrieval to generate accurate answers using LLMs (e.g., GPT-4o-mini).

## Features

- **Document Ingestion**: Load and process text documents from the `docs/` directory.
- **Smart Chunking**: Split documents into semantic chunks for efficient embedding.
- **Vector Storage**: Store embeddings locally using **ChromaDB**.
- **Local Embeddings**: Uses **Ollama** (`nomic-embed-text`) for privacy-preserving, local embeddings.
- **Conversational Retrieval**: Maintains chat history to allow follow-up questions using `history_aware_generation.py`.

## Tech Stack

- **Python 3.8+**
- **[LangChain](https://python.langchain.com/)**: Orchestration framework.
- **[Ollama](https://ollama.com/)**: Local model runner (for embeddings).
- **[ChromaDB](https://www.trychroma.com/)**: Vector database.
- **[OpenRouter](https://openrouter.ai/)**: Unified API for LLMs.

## Prerequisites

1.  **Python 3.8+**
2.  **Ollama** installed and running.
    *   Pull the embedding model:
        ```bash
        ollama pull nomic-embed-text:v1.5
        ```
3.  **OpenRouter API Key**: Sign up at [openrouter.ai](https://openrouter.ai/) to get an API key.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Arghajit-Saha/rag-system
    cd rag-system
    ```

2.  Create a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Configure Environment Variables:
    Create a `.env` file in the root directory:
    ```ini
    OPENROUTER_API_KEY=your_api_key_here
    ```

## Usage

### 1. Ingest Documents

Place your `.txt` files in the `docs/` directory. Then run the ingestion pipeline:

```bash
python ingestion_pipeline.py
```

This will load the documents, chunk them, generate embeddings locally with Ollama, and store them in `db/chroma_db`.

### 2. Retrieval (Test)

To test the retrieval mechanism (top 3 relevant chunks):

```bash
python retrieval_pipeline.py
```
*(Note: This uses a hardcoded query in the script)*

### 3. Generate Answers (Single Turn)

To ask a single question:

```bash
python answer_generation.py
```

### 4. Conversational Chat (History Aware)

To have a multi-turn conversation where the system remembers context:

```bash
python history_aware_generation.py
```
This script rephrases follow-up questions to be standalone before retrieving context, allowing for a natural conversation flow.

## Project Structure

```
.
├── db/                        # ChromaDB persistence directory
├── docs/                      # Source text documents
├── answer_generation.py       # Single-turn RAG QA
├── history_aware_generation.py # Multi-turn conversational RAG
├── ingestion_pipeline.py      # Document processing and embedding
├── retrieval_pipeline.py      # Context retrieval testing
├── .env                       # Environment variables (API keys)
├── .gitignore
└── README.md
```


