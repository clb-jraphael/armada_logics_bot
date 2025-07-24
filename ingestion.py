# ingestion.py

import os
import hashlib
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "chatbot-docs"

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
qdrant = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embedding_model)

def hash_chunk(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def preprocess_text(text: str) -> str:
    return text.replace("●", "-").strip()

def maybe_chunk_docs(docs: List[Document], filename: str) -> List[Document]:
    total_pages = len(docs)
    for d in docs:
        d.page_content = preprocess_text(d.page_content)
    if total_pages <= 2:
        return docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

def is_hash_in_qdrant(content_hash: str) -> bool:
    existing = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter={"must": [{"key": "hash", "match": {"value": content_hash}}]},
        limit=1,
    )
    return bool(existing[0])

def tag_and_filter_new_chunks(chunks: List[Document], filename: str, category: str) -> Tuple[List[Document], int]:
    new_chunks = []
    dupes = 0
    for chunk in chunks:
        chunk.page_content = chunk.page_content.strip()
        content_hash = hash_chunk(chunk.page_content)
        chunk.metadata.update({"hash": content_hash, "source": filename, "category": category})
        if is_hash_in_qdrant(content_hash):
            dupes += 1
            continue
        new_chunks.append(chunk)
    return new_chunks, dupes

def ingest_pdfs(folder_path: str = "documents") -> None:
    if not os.path.exists(folder_path):
        print("❌ 'documents' folder not found.")
        return
    total_new, total_dupes, total_files = 0, 0, 0
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if not filename.lower().endswith(".pdf"):
                continue
            total_files += 1
            file_path = os.path.join(root, filename)
            category = os.path.relpath(root, folder_path)
            print(f"\n📄 Processing: {filename} (Category: {category})")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            chunks = maybe_chunk_docs(docs, filename)
            new_chunks, dupes = tag_and_filter_new_chunks(chunks, filename, category)
            print(f"  • New chunks: {len(new_chunks)} | Duplicates skipped: {dupes}")
            if new_chunks:
                qdrant.add_documents(new_chunks)
                total_new += len(new_chunks)
            total_dupes += dupes
    print("\n================ SUMMARY ================")
    print(f"Files processed:          {total_files}")
    print(f"New chunks added:         {total_new}")
    print(f"Duplicate chunks skipped: {total_dupes}")
    if total_new:
        print("✅ Ingestion complete.")
    else:
        print("📭 No new chunks to add. All documents already exist.")

if __name__ == "__main__":
    ingest_pdfs()
