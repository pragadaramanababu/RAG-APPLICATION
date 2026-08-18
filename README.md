# 🤖 Company Docs RAG Assistant

An AI-powered Retrieval-Augmented Generation application that answers questions from company documents using semantic search, ChromaDB, Sentence Transformers, Groq, and a Streamlit chat interface.

## Features

- Document-based question answering
- Semantic retrieval
- Sentence Transformer embeddings
- Persistent ChromaDB
- Groq LLM integration
- Streamlit chat UI
- Source display
- Knowledge-base rebuilding
- Adjustable top-K retrieval
- Clear chat

## Architecture

Documents → Chunking → Sentence Transformer → ChromaDB → Retrieval → Groq LLM → Streamlit UI

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```text
GROQ_API_KEY=your_groq_api_key
```

Put your `.txt` documents inside `data/`.

Build the database:

```bash
python ingest.py
```

Run the GUI:

```bash
streamlit run app.py
```

## Project Structure

```text
RAG-APPLICATION/
├── app.py
├── rag.py
├── ingest.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/
└── chroma_db/
```

## Future Improvements

- PDF/DOCX upload
- File upload through GUI
- Reranking
- Hybrid search
- Conversation-aware retrieval
- RAG evaluation metrics
- Deployment
