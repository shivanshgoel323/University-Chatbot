import json
import os
import re
import shutil

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_FOLDER = "data"
DATASET_FOLDER = "dataset"
DB_FOLDER = "db"
COLLECTION_NAME = "university"

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Initializing ChromaDB...")
if os.path.exists(DB_FOLDER):
    shutil.rmtree(DB_FOLDER)

client = chromadb.PersistentClient(path=DB_FOLDER)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def clean_text(text: str) -> str:
    """Cleans text while preserving table and list structure."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 300) -> list[str]:
    """Chunks text with sliding window to prevent cutting off tables and sentences."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


documents = []
metadatas = []
ids = []

# ------------------------------------------------------------
# 1. Ingest PDFs with College Context Tagging
# ------------------------------------------------------------
print("\nReading PDFs from:", DATA_FOLDER)
if os.path.exists(DATA_FOLDER):
    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(".pdf")]
    for filename in sorted(pdf_files):
        filepath = os.path.join(DATA_FOLDER, filename)
        print(f"📄 Processing: {filename}")
        try:
            reader = PdfReader(filepath)
            full_pdf_text = ""

            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    full_pdf_text += f"\n--- Page {page_idx + 1} ---\n" + page_text

            cleaned_full_text = clean_text(full_pdf_text)
            if not cleaned_full_text:
                print(f"⚠️ No readable text in {filename}")
                continue

            chunks = chunk_text(cleaned_full_text, chunk_size=1600, overlap=350)

            doc_title = filename.rsplit(".", 1)[0].replace("_", " ").title()

            for chunk_idx, chunk in enumerate(chunks):
                # Prefacing chunks with college keywords ensures "abes fee", "abes syllabus" always match!
                enriched_chunk = (
                    f"Institution: ABES Engineering College (ABES University / AKTU)\n"
                    f"Document: {doc_title} ({filename})\n"
                    f"Content:\n{chunk}"
                )
                documents.append(enriched_chunk)
                metadatas.append({
                    "source": filename,
                    "title": doc_title,
                    "type": "pdf",
                })
                ids.append(f"pdf_{filename}_{chunk_idx}")

        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

# ------------------------------------------------------------
# 2. Ingest JSON Intent Datasets
# ------------------------------------------------------------
print("\nReading JSON datasets from:", DATASET_FOLDER)
if os.path.exists(DATASET_FOLDER):
    json_files = [f for f in os.listdir(DATASET_FOLDER) if f.lower().endswith(".json")]
    SKIP_INTENTS = {"greeting", "goodbye", "creator", "name", "random", "swear", "salutaion", "task"}

    for filename in json_files:
        filepath = os.path.join(DATASET_FOLDER, filename)
        print(f"📑 Processing JSON: {filename}")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            intents = data.get("intents", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])

            for idx, item in enumerate(intents):
                intent = item.get("intent", "")
                if intent.lower() in SKIP_INTENTS:
                    continue

                questions = item.get("text", [])
                responses = item.get("responses", [])

                content = (
                    f"Institution: ABES Engineering College\n"
                    f"Topic/Intent: {intent}\n"
                )
                if questions:
                    content += "Common Questions:\n" + "\n".join(f"- {q}" for q in questions) + "\n"
                if responses:
                    content += "Official Answers:\n" + "\n".join(f"- {r}" for r in responses)

                chunks = chunk_text(content, chunk_size=1200, overlap=200)
                for chunk_idx, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({"source": filename, "type": "json", "intent": intent})
                    ids.append(f"json_{idx}_{chunk_idx}")

        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

if not documents:
    print("❌ No documents found. Please check your 'data' or 'dataset' folders.")
    exit()

print(f"\nEmbedding {len(documents)} document chunks...")
embeddings = model.encode(documents, show_progress_bar=True, batch_size=32).tolist()

print("Storing into ChromaDB...")
collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

print("\n" + "=" * 45)
print("✅ Knowledge base built successfully!")
print(f"Total searchable chunks indexed: {collection.count()}")
print("=" * 45)