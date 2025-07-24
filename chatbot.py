# chatbot.py

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "chatbot-docs"

# Models
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
llm = OllamaLLM(model="gemma3:1b")

# Retriever
retriever = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embedding_model,
).as_retriever(search_type="mmr", search_kwargs={"k": 4})

# Prompt Template
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an intelligent and helpful assistant for Armada Logics. Your code name or name is Dagger One.

Use the following context to answer the user’s question. 
Always be detailed and informative. If the answer isn’t found in the context, say "I couldn’t find that information."

Context:
{context}

Question:
{question}

Answer:
""".strip()
)

# QA Chain with Prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

# For display
def _source_info(doc):
    meta = doc.metadata or {}
    source = meta.get("source", "unknown")
    category = meta.get("category", "unknown")
    page = meta.get("page_label") or meta.get("page")
    snippet = doc.page_content.strip().replace("\n", " ")
    if len(snippet) > 120:
        snippet = snippet[:117] + "..."
    return {"source": source, "category": category, "page": page, "snippet": snippet}

# Chat handler
def answer_query(query: str):
    result = qa_chain({"query": query})
    answer = result["result"]
    docs = result.get("source_documents", [])
    sources = [_source_info(d) for d in docs]
    return answer, sources
