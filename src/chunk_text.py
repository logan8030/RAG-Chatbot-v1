import os
import json
import re
from tqdm import tqdm

INPUT_PATH = "data/chunks/raw_chunks.jsonl"
OUT_PATH = "data/chunks/chunked.jsonl"

CHUNK_SIZE = 700  # in characters
CHUNK_OVERLAP = 100

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

def sentence_split(text):
    # Basic English sentence splitting
    return re.split(r'(?<=[.!?])\s+', text)

def split_into_chunks(text, chunk_size, overlap):
    sentences = sentence_split(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            # apply overlap
            overlap_text = current_chunk[-overlap:] if overlap else ""
            current_chunk = overlap_text + " " + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def chunk_pages(input_path, out_path, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(out_path, "w", encoding="utf-8") as outfile:

        for line in tqdm(infile, desc="Chunking text"):
            page = json.loads(line)
            page_text = page["text"]
            chunk_list = split_into_chunks(page_text, chunk_size, overlap)

            for i, chunk_text in enumerate(chunk_list):
                chunk = {
                    "chunk_id": f"{page['doc_id']}_p{page['page']}_c{i}",
                    "text": chunk_text,
                    "source": page["filename"],
                    "doc_id": page["doc_id"],
                    "version": page.get("version"),
                    "page": page["page"],
                    "topics": page.get("topics", []),
                    "entities": page.get("entities", []),
                    "chunk_index": i
                }
                outfile.write(json.dumps(chunk) + "\n")

if __name__ == "__main__":
    chunk_pages(INPUT_PATH, OUT_PATH)
    print(f"✅ Chunking complete. Output saved to {OUT_PATH}")