# chunk_json_demo.py
from __future__ import annotations
import json
import uuid
from pathlib import Path
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer                     
from docling.chunking import HybridChunker          
from docling_core.types.doc import DoclingDocument         
from app.pdf_to_json import pdf_to_json

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXTRACTED_DIR = Path("../data/extracted-jsons/")
CHROMA_DIR  = Path("../data/chroma_db")        # local folder for the DB

COLLECTION  = "pdf_chunks"                   # name inside Croma
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_TOKENS = 512
embedder = SentenceTransformer(MODEL_NAME)
client   = chromadb.PersistentClient(path=str(CHROMA_DIR))

def convert_pdf_to_json(pdf_path: str | Path, out_dir: str | Path | None = None) -> Path:
    """
    Convert a PDF file to JSON format using Docling.
    Returns the path to the generated JSON file.
    """
    json_path = pdf_to_json(pdf_path, write_file=True, out_dir=out_dir)
    return json_path

def chunk_json_files() -> List:
    """
    Process all JSON files in the extracted directory using HybridChunker.
    Uses chunker.serialize(chunk) for context-enriched text and prints page number information
    for a few chunks to verify that it's captured.

    Returns:
        A list of all chunks generated from all JSON files.
    """
    json_files = list(EXTRACTED_DIR.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {EXTRACTED_DIR.resolve()}.")
        return []

    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=MAX_TOKENS,
        merge_peers=True,
    )

    all_chunks = []

    for file_index, json_file in enumerate(json_files):
        print(f"\nProcessing file {file_index + 1}/{len(json_files)}: {json_file.name}")
        # Load the DoclingDocument from JSON by parsing the file then validating with the pydantic API.
        with open(json_file, "r", encoding="utf-8") as f:
            json_str = f.read()
        doc_dict = json.loads(json_str)
        doc = DoclingDocument.model_validate(doc_dict)
        if not doc:
            print(f"Loading document failed for {json_file.name}.")
            continue

        chunk_iter = chunker.chunk(dl_doc=doc)
        chunks = list(chunk_iter)

        non_empty_chunks = []
        empty_chunk_count = 0

        for i, chunk in enumerate(chunks):
            # Use the enriched serialization instead of chunk.text
            serialized_text = chunker.serialize(chunk)
            if not serialized_text.strip():
                empty_chunk_count += 1
                continue

            non_empty_chunks.append(chunk)

            # For the first file, print page numbers for the first 3 chunks only.
            if file_index == 0 and i < 3:
                page_numbers = sorted(
                    set(
                        prov.page_no
                        for item in chunk.meta.doc_items
                        for prov in item.prov
                        if hasattr(prov, "page_no")
                    )
                )
                print(f"Chunk {i}, Text Preview: {repr(serialized_text[:40])}…, Page Numbers: {page_numbers}")

        if empty_chunk_count > 0:
            print(f"Warning: Found {empty_chunk_count} empty chunks in {json_file.name}")

        print(f"Generated {len(non_empty_chunks)} valid chunks from {json_file.name}")
        all_chunks.extend(non_empty_chunks)

    print(f"\nTotal non-empty chunks generated: {len(all_chunks)}")
    return all_chunks, chunker

def store_chunks_to_chroma(chunks: list, chunker: HybridChunker) -> None:
    """
    Takes the list of Docling chunks, embeds them, and stores them in ChromaDB.
    Each record contains:
        id, text, page_number, embedding
    """
    if not chunks:
        print("No chunks to store.")
        return

    collection = client.get_or_create_collection(name=COLLECTION)

    ids, texts, metas, embeddings = [], [], [], []

    for chk in chunks:
        # Serialize the enriched text
        serialized = chunker.serialize(chk).strip()
        if not serialized:
            continue

        # Collect page numbers
        page_numbers = sorted(
            set(
                prov.page_no
                for item in chk.meta.doc_items
                for prov in item.prov
                if hasattr(prov, "page_no")
            )
        )
        page_no = page_numbers[0] if page_numbers else 1

        # Unique ID
        chunk_id = str(uuid.uuid4())

        # Build metadata dict
        meta = {
            "page_number": page_no,
            "all_pages": ",".join(map(str, page_numbers)),
            "text_preview": serialized[:200] + "..." if len(serialized) > 200 else serialized
        }

        # Embed
        emb = embedder.encode(serialized, show_progress_bar=False)

        ids.append(chunk_id)
        texts.append(serialized)
        metas.append(meta)
        embeddings.append(emb.tolist())

    # Batch insert into Chroma
    collection.add(
        documents=texts,
        metadatas=metas,
        embeddings=embeddings,
        ids=ids
    )
    print(f"✅ Stored {len(ids)} chunks in Chroma collection '{COLLECTION}'")

if __name__ == "__main__":
    # Convert PDF to JSON
    #pdf_path = Path("../data/Constitution-1973.pdf")
    #json_path = convert_pdf_to_json(pdf_path, out_dir="../data/extracted-jsons")

    json_path = Path("../data/extracted-jsons/Constitution-1973.json")
    
    # Chunk the JSON files
    chunks, chunker = chunk_json_files()
    print(f"Chunking complete: {len(chunks)} chunks generated.")
    
    # Store the chunks in ChromaDB
    store_chunks_to_chroma(chunks, chunker)
    print(f"Chunking and storage complete. Chunks stored in collection '{COLLECTION}' at {CHROMA_DIR.resolve()}.")