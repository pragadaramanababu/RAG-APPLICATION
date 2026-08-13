# Company Docs RAG Assistant

A Retrieval-Augmented Generation (RAG) pipeline that answers questions over a set of internal company documents (HR policy, engineering standards, onboarding guide, product knowledge base, and security policy) using semantic search + an LLM.

Built and run as a Google Colab notebook (`first.ipynb`).

## How It Works

1. **Load documents** — Reads five `.txt` company documents (HR policy, engineering standards, onboarding guide, product knowledge base, security policy).
2. **Chunk documents** — Splits each document into paragraph-level chunks (on double newlines), filtering out short fragments (<50 chars) and section separators (`===`).
3. **Store & index chunks** — Adds all chunks to a [ChromaDB](https://www.trychroma.com/) collection (`company_docs`), with each chunk tagged by its source document.
4. **Embed text** — Uses the `sentence-transformers/all-MiniLM-L6-v2` model to convert text into vector embeddings (demonstrated separately with a cosine-similarity example).
5. **Retrieve** — Given a question, queries the ChromaDB collection for the top-N most relevant chunks and returns their text + source metadata.
6. **Generate** — (In progress) Passes the retrieved context to an LLM via the [Groq API](https://groq.com/) (`llama-3.3-70b-versatile`) to produce a grounded answer.

## Tech Stack

| Component | Tool |
|---|---|
| LLM inference | Groq API (`llama-3.3-70b-versatile`) |
| Vector database | ChromaDB |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Similarity check | scikit-learn cosine similarity |
| Environment | Google Colab |

## Setup

### 1. Install dependencies

```bash
pip install openai chromadb python-dotenv groq sentence-transformers scikit-learn
```

### 2. API key

The notebook currently reads the Groq API key from Colab secrets:

```python
from google.colab import userdata
groqapikey = userdata.get('GROQ_AYPI_KE')
```

**Note:** the secret name `GROQ_AYPI_KE` appears to be a typo for `GROQ_API_KEY` — fix this in Colab's Secrets panel (🔑 icon) or rename the variable to match your stored key.

If running outside Colab, use a `.env` file instead:

```
GROQ_API_KEY=your_key_here
```

```python
from dotenv import load_dotenv
import os
load_dotenv()
groqapikey = os.getenv("GROQ_API_KEY")
```

### 3. Source documents

Place the following text files in your data directory (the notebook expects `/content/.config/_data/` on Colab):

- `company_hr_policy.txt`
- `engineering_standards.txt`
- `onboarding_guide.txt`
- `product_knowledge_base.txt`
- `security_policy.txt`

## Usage

Run the notebook top to bottom:

1. Install packages and load API credentials.
2. Load and chunk the five source documents.
3. Create the ChromaDB collection and add all chunks with metadata.
4. Retrieve relevant chunks for a query:

```python
chunks, sources = retrieve("what is work from home policy?", 3)
```

5. Pass retrieved chunks + the question to the LLM to generate a final answer (see **Status** below).

## Status / Known Issues

This notebook is a work in progress:

- The final `ask_rag()` function (meant to tie retrieval + generation into one call) is **incomplete/unfinished** — it needs a body that builds a prompt from retrieved chunks and calls the Groq chat completion endpoint.
- The `GROQ_AYPI_KE` secret name typo noted above should be corrected.
- Embeddings are currently generated implicitly by ChromaDB's default embedding function during `collection.add()`/`collection.query()`; the explicit `SentenceTransformer` example (cells 25–27) is a standalone demo and isn't yet wired into the retrieval pipeline.

## Suggested Next Steps

- [ ] Finish `ask_rag(question)`: retrieve chunks → build a system/user prompt with context → call `client.chat.completions.create(...)` → return the answer with cited sources.
- [ ] Explicitly pass the `all-MiniLM-L6-v2` embedding function into the ChromaDB collection for consistency between the demo and the retrieval pipeline.
- [ ] Add source citations to generated answers.
- [ ] Move hardcoded file paths and the Colab-specific secrets loading into a portable config (e.g., `.env` + `os.getenv`) for use outside Colab.
- [ ] Add persistent ChromaDB storage (`chromadb.PersistentClient`) so the index survives notebook restarts.