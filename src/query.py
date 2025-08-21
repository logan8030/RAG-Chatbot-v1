import requests

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "document_chunks"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mistral"
TOP_K = 5


client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
model = SentenceTransformer(EMBED_MODEL_NAME)
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_user_query",
            "description": "Scans user queries for topics, entities, and year",
            "parameters": {
                "type": "object",
                "properties": {
                    "topics":   {"type":"array","items":{"type":"string"}},
                    "entities": {"type":"array","items":{"type":"string"}},
                    "year":     {"type":["integer","null"]}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "refine_user_query",
            "description": "Refines the user query for improved hybrid search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "updated_query": {
                        "type": "string",
                        "description": "Rewritten query integrating keywords"
                    }
                },
                "required": ["updated_query"]
            }
        }
    }
]

def search_chunks(
    query_text: str, 
    topics: list[str] | None = None,
    entities: list[str] | None = None,
    version_filter: int| None = None, 
    top_k: int = TOP_K, 
    min_score: float | None = None,
):
    query_vector = model.encode(query_text).tolist()

    filters = list()
    if topics is not None:
        filters.append(
            FieldCondition(
                key="topics",
                match=MatchAny(any=topics),
            )
        )
    if entities is not None:
        filters.append(
            FieldCondition(
                key="entities",
                match=MatchAny(any=entities),
            )
        )
    if version_filter is not None:
        filters.append(
            FieldCondition(
                key="version",
                match=MatchAny(any=[version_filter]),
            )
        )
    
    query_filter = Filter(must=filters) if filters else None

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        score_threshold=min_score,
        with_payload=True,
        with_vectors=False,
    ).points  

    return results


def detect_filters_and_refine(query):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "Extract metadata using tools and refine the user query for vector search with keyword filtering."},
            {"role": "user",   "content": query}
        ],
        "tools":       tools,
        "tool_choice": "auto"
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()

def print_results(results):
    if not results:
        print("No results found.")
        return
    for i, result in enumerate(results):
        payload = result.payload or {}  
        print(f"\n🔹 Result {i + 1}:")
        print(f"Source: {payload.get('source', 'Unknown')} (page {payload.get('page', 'N/A')})")
        print(f"Version: {payload.get('version', 'N/A')}")
        print(f"Score: {result.score:.4f}")
        text = payload.get('text', '')
        print(f"Text: {text[:500] + '...' if text else 'No text available'}")

if __name__ == "__main__":
    print("📥 Ask a question (Ctrl+C to quit):\n")
    while True:
        try:
            query = input("❓ > ")
            result = detect_filters_and_refine(query)
            result_json = result.json()
            text = result_json.get('response', 'No response found.')
            #results = search_chunks(result["query"], version_filter=result["year"], topics=result["topics"], entities=result["entities"], min_score=0.3)
            print_results(text)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
