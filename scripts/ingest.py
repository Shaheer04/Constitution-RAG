# scripts/ingest.py
"""
One-shot ingestion:
python scripts/ingest.py
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from backend.preprocessor import Preprocessor
from backend.config import settings

def main():
    preprocessor = Preprocessor()

    # Convert all PDFs to JSON
    for pdf in settings.raw_pdf_dir.glob("*.pdf"):
        print(f"Processing {pdf.name} ...")
        preprocessor.convert_pdf_to_json(pdf, out_dir=settings.json_dir)
    print("PDF converted to JSON.")

    # Chunk all JSON files and store in ChromaDB
    chunks, chunker = preprocessor.chunk_json_files()
    print(f"Chunking complete: {len(chunks)} chunks generated.")
    preprocessor.store_chunks_to_chroma(chunks, chunker)
    print("All chunks stored in ChromaDB.")

if __name__ == "__main__":
    main()