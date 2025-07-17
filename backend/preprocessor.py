"""
Preprocessing module for handling PDF to JSON conversion and text chunking and embedding.
"""

from __future__ import annotations
import json
import uuid
from pathlib import Path
from typing import List, Dict, Tuple

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument
from backend.pdf_to_json import pdf_to_json
from backend.config import settings

class Preprocessor:
    def __init__(self):
        self.model_name = settings.embed_model
        self.extracted_dir = settings.json_dir
        self.chroma_dir = settings.chroma_dir
        self.collection_name = settings.collection_name
        self.max_tokens = settings.chunk_max_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.embedder = SentenceTransformer(self.model_name)
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))

    def convert_pdf_to_json(self, pdf_path: str | Path, out_dir: str | Path | None = None) -> Path:
        """
        Convert a PDF file to JSON format using Docling.
        Returns the path to the generated JSON file.
        """
        json_path = pdf_to_json(pdf_path, write_file=True, out_dir=out_dir)
        return json_path

    def chunk_json_files(self) -> Tuple[List, HybridChunker]:
        """
        Process all JSON files in the extracted directory using HybridChunker.
        Returns a list of all chunks generated from all JSON files and the chunker.
        """
        json_files = list(self.extracted_dir.glob("*.json"))

        if not json_files:
            print(f"No JSON files found in {self.extracted_dir.resolve()}.")
            return [], None

        chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            merge_peers=True,
        )

        all_chunks = []

        for file_index, json_file in enumerate(json_files):
            print(f"\nProcessing file {file_index + 1}/{len(json_files)}: {json_file.name}")
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
                serialized_text = chunker.serialize(chunk)
                if not serialized_text.strip():
                    empty_chunk_count += 1
                    continue

                non_empty_chunks.append(chunk)

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

    def store_chunks_to_chroma(self, chunks: list, chunker: HybridChunker) -> None:
        """
        Takes the list of Docling chunks, embeds them, and stores them in ChromaDB.
        Each record contains: id, text, page_number, embedding
        """
        if not chunks:
            print("No chunks to store.")
            return

        collection = self.client.get_or_create_collection(name=self.collection_name)

        ids, texts, metas, embeddings = [], [], [], []

        for chk in chunks:
            serialized = chunker.serialize(chk).strip()
            if not serialized:
                continue

            page_numbers = sorted(
                set(
                    prov.page_no
                    for item in chk.meta.doc_items
                    for prov in item.prov
                    if hasattr(prov, "page_no")
                )
            )
            page_no = page_numbers[0] if page_numbers else 1

            chunk_id = str(uuid.uuid4())

            meta = {
                "page_number": page_no,
                "all_pages": ",".join(map(str, page_numbers)),
                "text_preview": serialized[:200] + "..." if len(serialized) > 200 else serialized
            }

            emb = self.embedder.encode(serialized, show_progress_bar=False)

            ids.append(chunk_id)
            texts.append(serialized)
            metas.append(meta)
            embeddings.append(emb.tolist())

        collection.add(
            documents=texts,
            metadatas=metas,
            embeddings=embeddings,
            ids=ids
        )
        print(f"✅ Stored {len(ids)} chunks in Chroma collection '{self.collection_name}'")

if __name__ == "__main__":
    preprocessor = Preprocessor()
    # Example usage:
    # pdf_path = Path(settings.raw_pdf_dir) / settings.pdf_filename
    # json_path = preprocessor.convert_pdf_to_json(pdf_path, out_dir=settings.json_dir)

    chunks, chunker = preprocessor.chunk_json_files()
    print(f"Chunking complete: {len(chunks)} chunks generated.")

    preprocessor.store_chunks_to_chroma(chunks, chunker)
    print(f"Chunking and storage complete. Chunks stored in collection '{settings.collection_name}' at {settings.chroma_dir.resolve()}.")