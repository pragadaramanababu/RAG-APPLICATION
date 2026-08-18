import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "company_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Create a .env file with GROQ_API_KEY=your_api_key")

client = Groq(api_key=GROQ_API_KEY)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

def chunk_document(text, source_name):
    paragraphs = text.strip().split("\n\n")
    chunks = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if len(paragraph) < 50 or paragraph.startswith("==="):
            continue
        chunks.append({"text": paragraph, "source": source_name})
    return chunks

def load_documents():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    documents = []
    for file_path in DATA_DIR.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        source_name = file_path.stem.replace("_", " ").title()
        documents.extend(chunk_document(text, source_name))
    return documents

def build_vector_database():
    documents = load_documents()
    if not documents:
        raise ValueError("No .txt documents found inside the data folder.")

    global collection

    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    texts = [doc["text"] for doc in documents]
    ids = [f"chunk_{i}" for i in range(len(documents))]
    metadatas = [{"source": doc["source"]} for doc in documents]

    collection.add(documents=texts, ids=ids, metadatas=metadatas)
    return len(documents)

def get_document_count():
    try:
        return collection.count()
    except Exception:
        return 0

def retrieve(question, n_results=3):
    if collection.count() == 0:
        raise ValueError("Knowledge base is empty. Please build the knowledge base first.")

    n_results = min(n_results, collection.count())

    results = collection.query(query_texts=[question], n_results=n_results)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {
            "text": document,
            "source": metadata.get("source", "Unknown"),
            "distance": distance
        }
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]

def ask_rag(question, n_results=3):
    retrieved_chunks = retrieve(question, n_results)

    context = "\n\n".join(
        f"SOURCE: {chunk['source']}\n\n{chunk['text']}"
        for chunk in retrieved_chunks
    )

    system_prompt = """
You are a helpful company knowledge assistant.
Answer questions using ONLY the provided context.

Rules:
1. Use only information from the context.
2. Do not use outside knowledge.
3. Do not invent or assume information.
4. If the context does not contain the answer, clearly say you do not have enough information.
5. Keep the answer clear and concise.
6. When useful, mention the relevant source.
"""

    user_prompt = f"CONTEXT:\n\n{context}\n\nQUESTION:\n\n{question}"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": retrieved_chunks
    }
