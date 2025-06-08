import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient

# Use your Qdrant Cloud URL and API key
QDRANT_URL = os.getenv("QDRANT_URL", "https://1a3f0e99-06cc-40f1-a2ab-9c47668a0504.us-west-2-0.aws.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "math_kb"  # Make sure this matches your setup

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

qdrant = Qdrant(
    client=QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY),
    collection_name=COLLECTION_NAME,
    embeddings=embedding_model,
    content_payload_key="content",
)

question = input("Enter your math question: ")
docs = qdrant.similarity_search(question, k=3)

print("\n🔍 Top 3 relevant chunks:\n")
for i, doc in enumerate(docs, 1):
    print(f"{i}. {doc.page_content}\n")