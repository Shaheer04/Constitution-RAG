"""
PreProcessor Component with to Process Document and Create Hybrid Chunks and Save in Vector Database
"""

import uuid
from typing import List, Dict
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from collections import Counter


class PreProcessor:
    def __init__(self, 
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chroma_db_path: str = "./constitution_db",
        ollama_model: str = "qwen3",
        ollama_url: str = None):
        """
        Initialize the RAG pipeline
        
        Args:
            embedding_model_name: Name of the sentence transformer model
            chroma_db_path: Path to store ChromaDB
            ollama_model: Ollama model name for generation
            ollama_url: URL for remote Ollama instance (optional)
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.collection_name = "documents"
        
        # Initialize Ollama client with custom URL if provided
        if ollama_url:
            self.ollama_client = ollama.Client(host=ollama_url)
        else:
            self.ollama_client = ollama.Client()
        
        # Initialize document converter
        self.converter = DocumentConverter()
        
        # Initialize hybrid chunker
        self.chunker = HybridChunker()
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name
            )

    def convert_document(self, source_path: str):
        """
        Convert document using Docling
        
        Args:
            source_path: Path to the document
            
        Returns:
            Docling Document object
        """
        # Convert document with updated API
        result = self.converter.convert(source_path)
        
        return result.document

    def create_hybrid_chunks(self, document, max_tokens: int = 1000, overlap_ratio: float = 0.1) -> List[Dict]:
        """
        Create hybrid chunks from document using Docling's HybridChunker
        
        Args:
            document: Docling Document object
            max_tokens: Maximum tokens per chunk
            overlap_ratio: Overlap ratio between chunks
            
        Returns:
            List of chunk dictionaries with hierarchical information
        """
        # Use Docling's hybrid chunker
        chunks = self.chunker.chunk(document, tokenizer=None, max_tokens=max_tokens, overlap_ratio=overlap_ratio)
        
        # Convert to our format with enhanced metadata
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            
            chunk_dict = {
                'id': str(uuid.uuid4()),
                'text': chunk.text,
                'chunk_id': i,
                'doc_items': [item.__dict__ if hasattr(item, '__dict__') else str(item) for item in chunk.doc_items] if hasattr(chunk, 'doc_items') and chunk.doc_items else [],
                'level': self._get_chunk_level(chunk),
                'heading': self._get_chunk_heading(chunk),
                'page_number': self._get_page_number(chunk),
                'element_type': self._get_element_type(chunk),
                'length': len(chunk.text)
            }
            processed_chunks.append(chunk_dict)
        
        return processed_chunks

    def _get_chunk_level(self, chunk) -> int:
        """Get the hierarchical level of the chunk"""
        if hasattr(chunk, 'doc_items') and chunk.doc_items:
            for item in chunk.doc_items:
                if hasattr(item, 'label') and item.label and item.label.startswith('#'):
                    return len(item.label.split('#')[0])
        return 0

    def _get_chunk_heading(self, chunk) -> str:
        """Get the main heading of the chunk"""
        if hasattr(chunk, 'doc_items') and chunk.doc_items:
            for item in chunk.doc_items:
                if hasattr(item, 'label') and item.label and item.label.startswith('#'):
                    return item.label.replace('#', '').strip()
        return ""

    def _get_page_number(self, chunk) -> int:
        """Get page number from chunk"""
        if hasattr(chunk, 'doc_items') and chunk.doc_items:
            for item in chunk.doc_items:
                if hasattr(item, 'prov') and hasattr(item.prov, 'page_no'):
                    return item.prov.page_no
        return 0

    def _get_element_type(self, chunk) -> str:
        """Get the type of document element"""
        if hasattr(chunk, 'doc_items') and chunk.doc_items:
            types = []
            for item in chunk.doc_items:
                if hasattr(item, 'label'):
                    if item.label.startswith('#'):
                        types.append('heading')
                    elif item.label.startswith('Table'):
                        types.append('table')
                    elif item.label.startswith('Figure'):
                        types.append('figure')
                    else:
                        types.append('text')
            return ', '.join(set(types)) if types else 'text'
        return 'text'

    def embed_and_store(self, chunks: List[Dict], document_name: str = "document"):
        """
        Create embeddings and store in ChromaDB with hierarchical metadata
        
        Args:
            chunks: List of hierarchical chunks
            document_name: Name of the source document
        """
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in chunks]
        
        # Create embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        
        # Prepare data for ChromaDB with hierarchical metadata
        ids = [chunk['id'] for chunk in chunks]
        metadatas = [
            {
                'document_name': str(document_name),
                'chunk_id': str(chunk['chunk_id']),
                'level': int(chunk['level']) if chunk['level'] is not None else 0,
                'heading': str(chunk['heading']) if chunk['heading'] else "",
                'page_number': int(chunk['page_number']) if chunk['page_number'] is not None else 0,
                'element_type': str(chunk['element_type']),
                'length': int(chunk['length'])
            }
            for chunk in chunks
        ]
        
        # Store in ChromaDB
        try:
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            print(f"Error storing chunks in ChromaDB: {e}")
            raise