import json
import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# === CONFIG ===
COLLECTION_NAME = "document_chunks"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"
TOP_K = 5

# === SETUP ===
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# === CORE ===
def search_chunks(query_text, top_k=TOP_K):
    query_vec = embedder.encode(query_text).tolist()

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        with_payload=True
    )

    return response.points

def build_prompt(query, results):
    context = ""
    for r in results:
        payload = r.payload
        context += (
            f"[{payload['source']} - page {payload['page']}]: {payload['text']}\n\n"
        )

    prompt = f"""You are a helpful assistant that answers questions using ONLY the information provided in the sources below.
Cite the document name and page number when appropriate.

### User question:
{query}

### Sources:
{context}

### Answer:"""

    return prompt

def query_ollama(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    })

    if response.status_code != 200:
        raise Exception(f"Ollama error: {response.status_code} - {response.text}")

    return response.json()["response"]

# === INTERACTIVE LOOP ===
if __name__ == "__main__":
    print("🧠 Edwin is ready. Ask your question (Ctrl+C to quit):\n")

    while True:
        try:
            query = input("❓ > ")
            results = search_chunks(query)
            prompt = build_prompt(query, results)
            answer = query_ollama(prompt)
            print(f"\n💬 {answer}\n")
        except KeyboardInterrupt:
            print("\n👋 Exiting chat.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
