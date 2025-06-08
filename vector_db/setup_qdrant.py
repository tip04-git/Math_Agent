import json
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Initialize embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight & fast
# Load Q&A from JSON
with open("data/math_kb.json", "r") as f:
    kb_data = json.load(f)

# Connect to Qdrant Cloud (create one project and collection online first)
qdrant = QdrantClient(
    url="https://1a3f0e99-06cc-40f1-a2ab-9c47668a0504.us-west-2-0.aws.cloud.qdrant.io",  # Get this from Qdrant dashboard
    api_key=QDRANT_API_KEY,
)


# Define collection name
COLLECTION_NAME = "math_kb"

# ...existing code...
collections = [c.name for c in qdrant.get_collections().collections]
if COLLECTION_NAME in collections:
    qdrant.delete_collection(collection_name=COLLECTION_NAME)
# ...existing code...

# Recreate collection (safe start)
if COLLECTION_NAME in qdrant.get_collections().collections:
    qdrant.delete_collection(collection_name=COLLECTION_NAME)

qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)


# Add data
for i, item in enumerate(kb_data):
    question = item["question"]
    embedding = embedding_model.encode(question).tolist()
    payload = {
        "content": item["question"],
        "answer": item["answer"]
    }
    point = PointStruct(id=i, vector=embedding, payload=payload)
    qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])

print("✅ Qdrant Vector DB setup complete!")
