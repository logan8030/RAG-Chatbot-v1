edwin_RAG/
├── data/
│   ├── pdfs/                # Place your source PDFs here
│   └── chunks/              # Generated JSONL chunks
├── src/
│   ├── extract.py           # PDF → per-page text w/ metadata
│   ├── chunk.py             # Text → semantically chunked JSONL
│   ├── embed_qdrant.py      # Chunked text → Qdrant vector DB
│   ├── query.py             # CLI: query chunks (no LLM)
│   ├── chat.py              # CLI: query → retrieve → LLM
│   └── app.py               # FastAPI server for chat API
├── ui/
│   └── app.py               # Streamlit web UI (will add below)
├── docker-compose.yml       # Qdrant service
└── requirements.txt


🧪 Individual Script Usage
--------------------------

--extract.py--
📝 Extract text and metadata from PDFs

bash
python src/extract.py

Input: data/pdfs/*.pdf
Output: data/chunks/raw_chunks.jsonl


--chunk.py--
✂️ Chunk long text into ~700-character segments

bash
python src/chunk.py

Input: raw_chunks.jsonl
Output: chunked.jsonl


--embed_qdrant.py--
📌 Embed and store chunks in Qdrant

bash
docker compose up -d   # Start Qdrant (if not already running)
python src/embed_qdrant.py


--query.py--
🔍 Test retrieval from Qdrant

bash
python src/query.py

Input: user question
Output: relevant document chunks (no LLM yet)

--chat.py--
🧠 Run full RAG pipeline with local LLM (Ollama)

bash
ollama run mistral      # Seperate terminal, start Ollama LLM

bash
python src/chat.py

--🌐 API Usage — app.py--
Starts FastAPI server with /chat endpoint

bash
uvicorn src.app:app --reload
🔗 Swagger UI:
http://localhost:8000/docs

Example POST /chat:
bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the eviction rules in the 2024 guidebook?"}'