import streamlit as st
from dotenv import load_dotenv
import os
import hashlib
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "chatbot-docs"

# Init embedding model and vector store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
qdrant = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embedding_model,
)

def hash_chunk(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def is_hash_in_qdrant(content_hash: str) -> bool:
    existing = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter={"must": [{"key": "hash", "match": {"value": content_hash}}]},
        limit=1,
    )
    return bool(existing[0])

def prepare_document_chunks(content: str, category="manual", source="admin-input") -> list:
    document = Document(page_content=content)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents([document])

    final_chunks = []
    for chunk in chunks:
        chunk_text = chunk.page_content.strip()
        content_hash = hash_chunk(chunk_text)
        if is_hash_in_qdrant(content_hash):
            continue
        chunk.metadata.update({
            "hash": content_hash,
            "source": source,
            "category": category
        })
        final_chunks.append(chunk)
    return final_chunks

# Streamlit UI
st.set_page_config(page_title="Admin - Add Knowledge", layout="centered")
st.title("🔐 Admin Panel - Add to Knowledge Base")

admin_input = st.text_area("Enter new knowledge to add:", height=200)

if st.button("➕ Add to Knowledge Base"):
    if admin_input.strip():
        with st.spinner("Processing..."):
            chunks = prepare_document_chunks(admin_input)
            if chunks:
                qdrant.add_documents(chunks)
                st.success(f"✅ Added {len(chunks)} new chunk(s) to knowledge base.")
            else:
                st.info("📭 No new content to add. All chunks are already present.")
    else:
        st.warning("⚠️ Please enter some content.")
