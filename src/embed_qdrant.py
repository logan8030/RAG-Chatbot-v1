import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

CHUNK_FILE = "data/chunks/chunked.jsonl"
COLLECTION_NAME = "document_chunks"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBED_DIM = 384  # This depends on the embedding model

def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def embed_chunks(model, chunks):
    texts = [chunk["text"] for chunk in chunks]
    return model.encode(texts, show_progress_bar=True)

def connect_qdrant():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def create_collection(client, name, dim):
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

def ensure_collection_exists(client, name, dim):
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

def stable_id(chunk_id):
    return hash(chunk_id) & 0x7FFFFFFF

def upload_chunks(client, name, embeddings, chunks, batch_size=500):
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_vectors = embeddings[i:i+batch_size]

        points = []
        for chunk, vec in zip(batch_chunks, batch_vectors):
            point = PointStruct(
                id=stable_id(chunk["chunk_id"]),
                vector=vec.tolist(),
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "doc_id": chunk["doc_id"],
                    "version": chunk.get("version"),
                    "page": chunk["page"],
                    "chunk_index": chunk["chunk_index"],
                    "topics": chunk.get("topics", []),
                    "entities": chunk.get("entities", []),
                }
            )
            points.append(point)

        client.upsert(collection_name=name, points=points)
        print(f"✅ Uploaded {i + len(points)} / {total}")

if __name__ == "__main__":
    print("🔍 Loading chunks...")
    chunks = load_chunks(CHUNK_FILE)

    print("🧠 Embedding chunks...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_chunks(model, chunks)

    print("🔌 Connecting to Qdrant...")
    client = connect_qdrant()

    print("📦 Ensuring collection exists...")
    ensure_collection_exists(client, COLLECTION_NAME, EMBED_DIM) # I kept getting an error

    print("🚀 Uploading to Qdrant...")
    upload_chunks(client, COLLECTION_NAME, embeddings, chunks)