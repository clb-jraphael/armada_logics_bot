# chatbot.py

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "chatbot-docs"

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
llm = OllamaLLM(model="gemma3:1b")  # adjust as needed

# All documents for this organization are used
retriever = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embedding_model,
).as_retriever(search_type="mmr", search_kwargs={"k": 4})

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

def _source_info(doc):
    meta = doc.metadata or {}
    source = meta.get("source", "unknown")
    category = meta.get("category", "unknown")
    page = meta.get("page_label") or meta.get("page")
    snippet = doc.page_content.strip().replace("\n", " ")
    if len(snippet) > 120:
        snippet = snippet[:117] + "..."
    return {"source": source, "category": category, "page": page, "snippet": snippet}

def answer_query(query: str):
    result = qa_chain({"query": query})
    answer = result["result"]
    docs = result.get("source_documents", [])
    sources = [_source_info(d) for d in docs]
    return answer, sources
