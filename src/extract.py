import os
import re
import json
import fitz # PyMuPDF
from tqdm import tqdm

from get_metadata import call_extraction_model

PDF_DIR = "data/pdfs"
OUT_DIR = "data/chunks/raw_chunks.jsonl"

def extract_year_from_filename(filename):
    match = re.search(r'(20\d{2}|19\d{2})', filename)
    return int(match.group(1)) if match else None

def extract_text_pymupdf(filepath):
    """
    – Opens the PDF
    – Pulls the full text for metadata extraction
    – Then re‑walks each page to return per‑page chunks with the same topics/entities
    """
    doc = fitz.open(filepath)
    
    # 1) Extract full document text and get its metadata once
    full_text = ""
    for page in doc:  
        full_text += page.get_text("text") + "\n"
    doc_id   = os.path.basename(filepath)
    topics, entities = call_extraction_model(doc_id, full_text)

    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        metadata = {
            "page": i + 1,
            "text": text.strip(),
            "topics": topics,
            "entities": entities,
        }
        pages.append(metadata)
    return pages

def extract_all_pdf_data(pdf_dir):
    all_chunks = []
    all_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]

    for filename in tqdm(all_files, desc="Extracting PDFs"):
        filepath = os.path.join(pdf_dir, filename)
        base = os.path.splitext(filename)[0]
        version = extract_year_from_filename(filename)

        try:
            pages = extract_text_pymupdf(filepath)
            for page in pages:
                chunk = {
                    "filename": filename,
                    "doc_id": base,
                    "version": version,
                    "page": page["page"],
                    "text": page["text"],
                    "topics": page["topics"],
                    "entities": page["entities"], 
                }
                all_chunks.append(chunk)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return all_chunks

def save_jsonl(chunks, out_path):
    with open(out_path, "w", encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

if __name__ == "__main__":
    chunks = extract_all_pdf_data(PDF_DIR)
    save_jsonl(chunks, OUT_DIR)
    print(f"Extracted {len(chunks)} chunks and saved to {OUT_DIR}")
