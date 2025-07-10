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
import hashlib


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

    def create_hybrid_chunks_with_citations(self, document, max_tokens: int = 1000, overlap_ratio: float = 0.1) -> List[Dict]:
        """
        Create hybrid chunks from document using Docling's HybridChunker with citation metadata
        
        Args:
            document: Docling Document object
            max_tokens: Maximum tokens per chunk
            overlap_ratio: Overlap ratio between chunks
            
        Returns:
            List of chunk dictionaries with hierarchical information and citation metadata
        """
        # Use Docling's hybrid chunker
        chunks = self.chunker.chunk(document, tokenizer=None, max_tokens=max_tokens, overlap_ratio=overlap_ratio)
        
        # Convert to our format with enhanced metadata including citations
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            
            # Get basic information
            page_number = self._get_page_number(chunk)
            heading = self._get_chunk_heading(chunk)
            
            # Generate citation metadata
            paragraph_id = self._generate_paragraph_id(chunk.text, page_number, i)
            citation_text = self._create_citation_text(chunk.text, page_number, heading)
            paragraph_preview = self._create_paragraph_preview(chunk.text)
            
            chunk_dict = {
                'id': str(uuid.uuid4()),
                'text': chunk.text,
                'chunk_id': i,
                'doc_items': [item.__dict__ if hasattr(item, '__dict__') else str(item) for item in chunk.doc_items] if hasattr(chunk, 'doc_items') and chunk.doc_items else [],
                'level': self._get_chunk_level(chunk),
                'heading': heading,
                'page_number': page_number,
                'element_type': self._get_element_type(chunk),
                'length': len(chunk.text),
                
                # Citation-specific metadata
                'paragraph_id': paragraph_id,
                'citation_text': citation_text,
                'paragraph_preview': paragraph_preview,
                'section_info': self._get_section_info(chunk),
                'content_type': self._get_detailed_content_type(chunk)
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

    def _generate_paragraph_id(self, text: str, page_number: int, chunk_index: int) -> str:
        """Generate unique paragraph ID using text content"""
        # Create hash from text content for uniqueness
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"p{page_number}_{chunk_index}_{content_hash}"

    def _create_citation_text(self, text: str, page_number: int, heading: str = "") -> str:
        """Create citation text for the chunk"""
        # Get first 150 characters as preview
        preview = text[:150].strip()
        if len(text) > 150:
            preview += "..."
        
        # Include heading if available
        if heading:
            return f"Page {page_number}, Section '{heading}': \"{preview}\""
        else:
            return f"Page {page_number}: \"{preview}\""

    def _create_paragraph_preview(self, text: str) -> str:
        """Create a longer preview of the paragraph for context"""
        preview = text[:300].strip()
        if len(text) > 300:
            preview += "..."
        return preview

    def _get_section_info(self, chunk) -> Dict:
        """Get detailed section information from Docling chunk"""
        section_info = {
            'section_title': '',
            'section_number': '',
            'parent_section': '',
            'section_level': 0
        }
        
        if hasattr(chunk, 'doc_items') and chunk.doc_items:
            for item in chunk.doc_items:
                if hasattr(item, 'label') and item.label:
                    if item.label.startswith('#'):
                        # Extract section information from heading
                        heading_text = item.label.replace('#', '').strip()
                        section_info['section_title'] = heading_text
                        section_info['section_level'] = len(item.label.split('#')[0])
                        
                        # Try to extract section number if present
                        import re
                        section_match = re.search(r'^(\d+\.?\d*)', heading_text)
                        if section_match:
                            section_info['section_number'] = section_match.group(1)
        
        return section_info

    def _get_detailed_content_type(self, chunk) -> str:
        """Get detailed content type for better citation context"""
        content_types = []
        
        if hasattr(chunk, 'doc_items') and chunk.doc_items:
            for item in chunk.doc_items:
                if hasattr(item, 'label') and item.label:
                    if item.label.startswith('#'):
                        content_types.append('heading')
                    elif 'table' in item.label.lower():
                        content_types.append('table')
                    elif 'figure' in item.label.lower():
                        content_types.append('figure')
                    elif 'list' in item.label.lower():
                        content_types.append('list')
                    else:
                        content_types.append('paragraph')
        
        return ', '.join(set(content_types)) if content_types else 'paragraph'

    def embed_and_store(self, chunks: List[Dict], document_name: str = "document"):
        """
        Create embeddings and store in ChromaDB with hierarchical metadata and citations
        
        Args:
            chunks: List of hierarchical chunks with citation metadata
            document_name: Name of the source document
        """
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in chunks]
        
        # Create embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        
        # Prepare data for ChromaDB with hierarchical metadata and citations
        ids = [chunk['id'] for chunk in chunks]
        metadatas = [
            {
                'document_name': str(document_name),
                'chunk_id': str(chunk['chunk_id']),
                'level': int(chunk['level']) if chunk['level'] is not None else 0,
                'heading': str(chunk['heading']) if chunk['heading'] else "",
                'page_number': int(chunk['page_number']) if chunk['page_number'] is not None else 0,
                'element_type': str(chunk['element_type']),
                'length': int(chunk['length']),
                
                # Citation metadata
                'paragraph_id': str(chunk['paragraph_id']),
                'citation_text': str(chunk['citation_text']),
                'paragraph_preview': str(chunk['paragraph_preview']),
                'section_title': str(chunk['section_info']['section_title']),
                'section_number': str(chunk['section_info']['section_number']),
                'section_level': int(chunk['section_info']['section_level']),
                'content_type': str(chunk['content_type'])
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
            print(f"✅ Stored {len(chunks)} chunks with citation metadata")
        except Exception as e:
            print(f"Error storing chunks in ChromaDB: {e}")
            raise