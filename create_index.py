from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Create index for metadata field "hash"
client.create_payload_index(
    collection_name="chatbot-docs",
    field_name="hash",
    field_schema="keyword",
)

print("✅ Created metadata index for 'hash'.")
