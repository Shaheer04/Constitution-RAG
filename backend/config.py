from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    embed_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-transformer model to embed text"
    )
    chroma_dir: Path = Field(
        default=Path(__file__).resolve().parent.parent / "data" / "chroma_db",
        description="Folder that persists ChromaDB"
    )
    collection_name: str = Field(default="pdf_chunks")

    raw_pdf_dir: Path = Field(
        default=Path(__file__).resolve().parent.parent / "data" / "raw_pdfs"
    )
    json_dir: Path = Field(
        default=Path(__file__).resolve().parent.parent / "data" / "extracted-jsons"
    )

    chunk_max_tokens: int = Field(default=256)
    chunk_overlap_ratio: float = Field(default=0.1)
    ollama_model: str = Field(default="llama3.1", description="Ollama model name")
    ollama_url: str = Field( description="Ollama server URL")

    class Config:
        env_file = Path(__file__).resolve().parent / ".env"
        env_file_encoding = "utf-8"

# singleton instance
settings = Settings()