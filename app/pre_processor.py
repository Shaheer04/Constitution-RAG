"""
PreProcessor with Advanced Performance Features
"""

import uuid
import gc
import os
import psutil
import time
import threading
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from functools import lru_cache
from collections import defaultdict
import weakref
import re

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import hashlib
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for processing parameters"""
    min_batch_size: int = 16
    max_batch_size: int = 256
    target_memory_usage: float = 0.7  # 70% of available memory
    max_workers: int = 4
    chunk_overlap_ratio: float = 0.1
    max_tokens_per_chunk: int = 1000
    connection_pool_size: int = 5
    retry_attempts: int = 3
    retry_delay: float = 1.0
    progress_update_interval: int = 100

class ProcessingError(Exception):
    """Base exception for processing errors"""
    pass

class MemoryError(ProcessingError):
    """Memory-related processing error"""
    pass

class DatabaseError(ProcessingError):
    """Database-related processing error"""
    pass

class ValidationError(ProcessingError):
    """Input validation error"""
    pass

class MemoryMonitor:
    """Monitor and manage memory usage"""
    
    def __init__(self, target_usage: float = 0.7):
        self.target_usage = target_usage
        self.process = psutil.Process()
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information"""
        memory_info = self.process.memory_info()
        available_memory = psutil.virtual_memory().available
        
        return {
            'used_mb': memory_info.rss / (1024 * 1024),
            'available_mb': available_memory / (1024 * 1024),
            'usage_percent': psutil.virtual_memory().percent
        }
    
    def is_memory_available(self, required_mb: float) -> bool:
        """Check if sufficient memory is available"""
        memory_info = self.get_memory_info()
        return memory_info['available_mb'] > required_mb
    
    def get_optimal_batch_size(self, base_size: int, min_size: int, max_size: int) -> int:
        """Calculate optimal batch size based on memory usage"""
        memory_info = self.get_memory_info()
        usage_ratio = memory_info['usage_percent'] / 100
        
        if usage_ratio > self.target_usage:
            # Reduce batch size if memory usage is high
            return max(min_size, int(base_size * (1 - usage_ratio + self.target_usage)))
        else:
            # Increase batch size if memory is available
            return min(max_size, int(base_size * (1 + (self.target_usage - usage_ratio))))

class HashCache:
    """Cache for hash calculations to avoid duplicates"""
    
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = defaultdict(int)
        
    def get_hash(self, text: str) -> str:
        """Get hash with caching"""
        if text in self.cache:
            self.access_count[text] += 1
            return self.cache[text]
        
        # Calculate hash
        hash_value = hashlib.md5(text.encode()).hexdigest()[:8]
        
        # Add to cache with LRU eviction
        if len(self.cache) >= self.max_size:
            self._evict_least_used()
        
        self.cache[text] = hash_value
        self.access_count[text] = 1
        return hash_value
    
    def _evict_least_used(self):
        """Evict least recently used items"""
        if not self.cache:
            return
        
        # Find the least accessed item
        least_used = min(self.access_count.items(), key=lambda x: x[1])
        text_to_remove = least_used[0]
        
        # Remove from cache
        del self.cache[text_to_remove]
        del self.access_count[text_to_remove]

class ConnectionPool:
    """Connection pool for ChromaDB clients"""
    
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self.pool = []
        self.lock = threading.Lock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        for _ in range(self.pool_size):
            try:
                client = chromadb.PersistentClient(path=self.db_path)
                self.pool.append(client)
            except Exception as e:
                logger.warning(f"Failed to create connection: {e}")
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with context manager"""
        connection = None
        try:
            with self.lock:
                if self.pool:
                    connection = self.pool.pop()
                else:
                    # Create new connection if pool is empty
                    connection = chromadb.PersistentClient(path=self.db_path)
            
            yield connection
        finally:
            if connection:
                with self.lock:
                    if len(self.pool) < self.pool_size:
                        self.pool.append(connection)

class ProgressTracker:
    """Track and report progress for long-running operations"""
    
    def __init__(self, total_items: int, update_interval: int = 100):
        self.total_items = total_items
        self.processed_items = 0
        self.start_time = time.time()
        self.last_update = 0
        self.update_interval = update_interval
        
    def update(self, count: int = 1):
        """Update progress counter"""
        self.processed_items += count
        
        if self.processed_items - self.last_update >= self.update_interval:
            self._log_progress()
            self.last_update = self.processed_items
    
    def _log_progress(self):
        """Log current progress"""
        elapsed_time = time.time() - self.start_time
        progress_percent = (self.processed_items / self.total_items) * 100
        
        if elapsed_time > 0:
            items_per_second = self.processed_items / elapsed_time
            eta = (self.total_items - self.processed_items) / items_per_second if items_per_second > 0 else 0
            
            logger.info(f"Progress: {self.processed_items}/{self.total_items} ({progress_percent:.1f}%) "
                       f"- {items_per_second:.1f} items/sec - ETA: {eta:.1f}s")

class InputValidator:
    """Validate and sanitize inputs"""
    
    @staticmethod
    def validate_file_path(path: str) -> str:
        """Validate and sanitize file path"""
        if not path or not isinstance(path, str):
            raise ValidationError("File path must be a non-empty string")
        
        path = path.strip()
        if not os.path.exists(path):
            raise ValidationError(f"File does not exist: {path}")
        
        if not os.path.isfile(path):
            raise ValidationError(f"Path is not a file: {path}")
        
        return path
    
    @staticmethod
    def validate_chunks(chunks: List[Dict]) -> List[Dict]:
        """Validate chunk data structure"""
        if not isinstance(chunks, list):
            raise ValidationError("Chunks must be a list")
        
        if not chunks:
            raise ValidationError("Chunks list cannot be empty")
        
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValidationError(f"Chunk {i} must be a dictionary")
            
            if 'text' not in chunk or not isinstance(chunk['text'], str):
                raise ValidationError(f"Chunk {i} must have 'text' field as string")
            
            if not chunk['text'].strip():
                raise ValidationError(f"Chunk {i} text cannot be empty")
        
        return chunks
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text content"""
        if not isinstance(text, str):
            return str(text)
        
        # Remove null bytes and control characters
        text = text.replace('\x00', '')
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        return text.strip()

class PreProcessor:
    """Optimized PreProcessor with advanced performance features"""
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 chroma_db_path: str = "./constitution_db",
                 ollama_model: str = "qwen3",
                 ollama_url: str = None,
                 config: Optional[ProcessingConfig] = None):
        
        # Configuration
        self.config = config or ProcessingConfig()
        
        # Basic parameters
        self.embedding_model_name = embedding_model_name
        self.chroma_db_path = chroma_db_path
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.collection_name = "documents"
        
        # Lazy initialization
        self._embedding_model = None
        self._converter = None
        self._chunker = None
        self._ollama_client = None
        
        # Performance components
        self.memory_monitor = MemoryMonitor(self.config.target_memory_usage)
        self.hash_cache = HashCache()
        self.connection_pool = ConnectionPool(chroma_db_path, self.config.connection_pool_size)
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Weak references for cleanup
        self._cleanup_refs = weakref.WeakSet()
        
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
            
            # Clear caches
            if hasattr(self, 'hash_cache'):
                self.hash_cache.cache.clear()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    @property
    def embedding_model(self):
        """Lazy loading of embedding model"""
        if self._embedding_model is None:
            try:
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                raise ProcessingError(f"Failed to load embedding model: {e}")
        return self._embedding_model

    @property
    def converter(self):
        """Lazy loading of document converter"""
        if self._converter is None:
            try:
                self._converter = DocumentConverter()
                logger.info("Document converter initialized")
            except Exception as e:
                raise ProcessingError(f"Failed to initialize document converter: {e}")
        return self._converter

    @property
    def chunker(self):
        """Lazy loading of chunker"""
        if self._chunker is None:
            try:
                self._chunker = HybridChunker()
                logger.info("Hybrid chunker initialized")
            except Exception as e:
                raise ProcessingError(f"Failed to initialize chunker: {e}")
        return self._chunker

    def convert_document(self, source_path: str):
        """Convert document with validation and error handling, including page extraction and page markers"""
        try:
            # Validate input
            source_path = InputValidator.validate_file_path(source_path)
            
            # Check memory before processing
            file_size_mb = os.path.getsize(source_path) / (1024 * 1024)
            if not self.memory_monitor.is_memory_available(file_size_mb * 3):  # 3x safety margin
                raise MemoryError(f"Insufficient memory to process file ({file_size_mb:.1f}MB)")

            # Use your converter to get the markdown document
            result = self.converter.convert(source_path)
            if not result or not result.document:
                raise ProcessingError(f"Document conversion failed: {source_path}")

            # Now extract page texts from the PDF
            from PyPDF2 import PdfReader
            reader = PdfReader(source_path)
            pages = []
            page_markdown = ""
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                pages.append({"page_number": page_number, "text": page_text})
                page_markdown += f"\n[[PAGE:{page_number}]]\n{page_text}"

            # Optionally, you can combine your converter's markdown and the page markers.
            # For best accuracy, you may want to align the converter's output with the page markers.
            # Here, we simply prepend the page markers to the converted document:
            document_with_markers = page_markdown + "\n" + result.document

            logger.info(f"Document converted successfully: {source_path}")
            return {"pages": pages, "document": document_with_markers}

        except Exception as e:
            logger.error(f"Error converting document {source_path}: {e}")
            raise ProcessingError(f"Document conversion failed: {e}")

    def create_hybrid_chunks_with_citations(self, 
                                           document, 
                                           max_tokens: int = None, 
                                           overlap_ratio: float = None) -> List[Dict]:
        """Create optimized chunks with citations and memory management"""
        try:
            # Use config defaults if not provided
            max_tokens = max_tokens or self.config.max_tokens_per_chunk
            overlap_ratio = overlap_ratio or self.config.chunk_overlap_ratio
            
            logger.info(f"Creating chunks with max_tokens={max_tokens}, overlap_ratio={overlap_ratio}")
            
            # Check memory before chunking
            memory_info = self.memory_monitor.get_memory_info()
            if memory_info['usage_percent'] > 80:
                logger.warning("High memory usage detected, forcing garbage collection")
                gc.collect()
            
            # Create chunks
            chunks = self.chunker.chunk(
                document, 
                tokenizer=None, 
                max_tokens=max_tokens, 
                overlap_ratio=overlap_ratio
            )

            if not chunks:
                raise ProcessingError("No chunks created from document")

            # Assign page numbers to each chunk
            chunks = self.assign_page_numbers_to_chunks(chunks)

            logger.info(f"Created {len(chunks)} raw chunks")
            
            # Process chunks in batches to manage memory
            batch_size = self.memory_monitor.get_optimal_batch_size(
                base_size=100,
                min_size=10,
                max_size=500
            )
            
            processed_chunks = []
            progress_tracker = ProgressTracker(len(chunks), self.config.progress_update_interval)
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_processed = self._process_chunk_batch(batch, i)
                processed_chunks.extend(batch_processed)
                
                progress_tracker.update(len(batch))
                
                # Memory management
                if i % (batch_size * 5) == 0:  # Every 5 batches
                    gc.collect()
            
            logger.info(f"✅ Created {len(processed_chunks)} processed chunks with citations")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            raise ProcessingError(f"Chunk creation failed: {e}")

    def _process_chunk_batch(self, chunks: List, start_index: int) -> List[Dict]:
        """Process a batch of chunks efficiently"""
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            try:
                chunk_index = start_index + i
                page_number = self._get_page_number(chunk)
                heading = self._get_chunk_heading(chunk)
                
                # Sanitize text
                clean_text = InputValidator.sanitize_text(chunk.text)
                
                # Use cached hash calculation
                text_hash = self.hash_cache.get_hash(clean_text)
                
                chunk_dict = {
                    'id': str(uuid.uuid4()),
                    'text': clean_text,
                    'chunk_id': chunk_index,
                    'level': self._get_chunk_level(chunk),
                    'heading': heading,
                    'page_number': page_number,
                    'element_type': self._get_element_type(chunk),
                    'length': len(clean_text),
                    'paragraph_id': f"p{page_number}_{chunk_index}_{text_hash}",
                    'citation_text': self._create_citation_text(clean_text, page_number, heading),
                    'content_type': self._get_detailed_content_type(chunk),
                    'section_info': self._get_section_info(chunk)
                }
                
                processed_chunks.append(chunk_dict)
                
            except Exception as e:
                logger.warning(f"Error processing chunk {start_index + i}: {e}")
                continue
        
        return processed_chunks

    @lru_cache(maxsize=1000)
    def _get_chunk_level(self, chunk) -> int:
        """Get chunk level with caching"""
        if hasattr(chunk, 'doc_items') and chunk.doc_items:
            for item in chunk.doc_items:
                if hasattr(item, 'label') and item.label and item.label.startswith('#'):
                    return item.label.count('#')
        return 0

    @lru_cache(maxsize=1000)
    def _get_chunk_heading(self, chunk) -> str:
        """Get chunk heading with caching"""
        if hasattr(chunk, 'doc_items') and chunk.doc_items:
            for item in chunk.doc_items:
                if hasattr(item, 'label') and item.label and item.label.startswith('#'):
                    return item.label.replace('#', '').strip()
        return ""

    @lru_cache(maxsize=1000)
    def _get_page_number(self, chunk) -> int:
        """Return the page number for the given chunk, if available."""
        if isinstance(chunk, dict) and 'page_number' in chunk:
            return int(chunk['page_number'])
        if hasattr(chunk, 'page_number'):
            return int(chunk.page_number)
        logger.warning("Page number not found for chunk, defaulting to 1")
        return 1

    @lru_cache(maxsize=1000)
    def _get_element_type(self, chunk) -> str:
        """Get element type with caching"""
        if hasattr(chunk, 'doc_items') and chunk.doc_items:
            for item in chunk.doc_items:
                if hasattr(item, 'label'):
                    label = item.label.lower()
                    if label.startswith('#'):
                        return 'heading'
                    elif 'table' in label:
                        return 'table'
                    elif 'figure' in label:
                        return 'figure'
        return 'text'

    def _create_citation_text(self, text: str, page_number: int, heading: str = "") -> str:
        """Create citation text (hash already calculated)"""
        preview = text[:100].strip()
        if len(text) > 100:
            preview += "..."
        
        if heading:
            return f"Page {page_number}, {heading}: \"{preview}\""
        else:
            return f"Page {page_number}: \"{preview}\""

    def _get_section_info(self, chunk) -> Dict:
        """Get section info with caching"""
        section_info = {
            'section_title': '',
            'section_number': '',
            'section_level': 0
        }
        
        if hasattr(chunk, 'doc_items') and chunk.doc_items:
            for item in chunk.doc_items:
                if hasattr(item, 'label') and item.label and item.label.startswith('#'):
                    heading_text = item.label.replace('#', '').strip()
                    section_info['section_title'] = heading_text
                    section_info['section_level'] = item.label.count('#')
                    
                    # Extract section number
                    import re
                    section_match = re.search(r'^(\d+\.?\d*)', heading_text)
                    if section_match:
                        section_info['section_number'] = section_match.group(1)
        
        return section_info

    def _get_detailed_content_type(self, chunk) -> str:
        """Get detailed content type with caching"""
        if hasattr(chunk, 'doc_items') and chunk.doc_items:
            for item in chunk.doc_items:
                if hasattr(item, 'label') and item.label:
                    label = item.label.lower()
                    if label.startswith('#'):
                        return 'heading'
                    elif 'table' in label:
                        return 'table'
                    elif 'figure' in label:
                        return 'figure'
                    elif 'list' in label:
                        return 'list'
        return 'paragraph'

    def embed_and_store(self, chunks: List[Dict], document_name: str = "document"):
        """Optimized embedding and storage with advanced error handling"""
        try:
            # Validate inputs
            chunks = InputValidator.validate_chunks(chunks)
            document_name = InputValidator.sanitize_text(document_name)
            
            if not chunks:
                logger.warning("No chunks to process")
                return
            
            logger.info(f"Processing {len(chunks)} chunks for storage")
            
            # Dynamic batch sizing based on memory
            base_batch_size = self.memory_monitor.get_optimal_batch_size(
                base_size=64,
                min_size=self.config.min_batch_size,
                max_size=self.config.max_batch_size
            )
            
            progress_tracker = ProgressTracker(len(chunks), self.config.progress_update_interval)
            stored_count = 0
            
            # Process in batches
            for i in range(0, len(chunks), base_batch_size):
                batch = chunks[i:i + base_batch_size]
                
                # Adjust batch size based on current memory usage
                current_batch_size = self.memory_monitor.get_optimal_batch_size(
                    base_size=len(batch),
                    min_size=self.config.min_batch_size,
                    max_size=min(base_batch_size, len(batch))
                )
                
                if current_batch_size < len(batch):
                    batch = batch[:current_batch_size]
                
                # Process batch with retry logic
                batch_stored = self._process_embedding_batch(batch, document_name)
                stored_count += batch_stored
                
                progress_tracker.update(batch_stored)
                
                # Memory management
                if i % (base_batch_size * 3) == 0:  # Every 3 batches
                    gc.collect()
            
            logger.info(f"✅ Successfully stored {stored_count} chunks with citations")
            
        except Exception as e:
            logger.error(f"Error in embed_and_store: {e}")
            raise DatabaseError(f"Storage operation failed: {e}")

    def _process_embedding_batch(self, batch: List[Dict], document_name: str) -> int:
        """Process a batch of embeddings with retry logic"""
        for attempt in range(self.config.retry_attempts):
            try:
                # Extract texts
                texts = [chunk['text'] for chunk in batch]
                
                # Generate embeddings with dynamic batch size
                embeddings = self.embedding_model.encode(
                    texts, 
                    show_progress_bar=False, 
                    batch_size=min(len(texts), 64)
                )
                
                # Prepare data for storage
                ids = [chunk['id'] for chunk in batch]
                metadatas = self._prepare_metadata_batch(batch, document_name)
                
                # Store with connection pooling and transaction handling
                with self.connection_pool.get_connection() as client:
                    collection = self._get_or_create_collection(client)
                    
                    # Store with transaction-like behavior
                    try:
                        collection.add(
                            embeddings=embeddings.tolist(),
                            documents=texts,
                            metadatas=metadatas,
                            ids=ids
                        )
                        
                        logger.debug(f"Stored batch of {len(batch)} chunks")
                        return len(batch)
                        
                    except Exception as e:
                        # Rollback strategy: try to remove any partially added data
                        try:
                            existing_ids = collection.get(ids=ids)['ids']
                            if existing_ids:
                                collection.delete(ids=existing_ids)
                                logger.warning(f"Rolled back {len(existing_ids)} partially stored chunks")
                        except:
                            pass  # Rollback failed, but main operation already failed
                        
                        raise e
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise DatabaseError(f"Failed to store batch after {self.config.retry_attempts} attempts: {e}")
        
        return 0

    def _prepare_metadata_batch(self, batch: List[Dict], document_name: str) -> List[Dict]:
        """Prepare metadata for a batch efficiently"""
        metadatas = []
        
        for chunk in batch:
            try:
                metadata = {
                    'document_name': str(document_name),
                    'chunk_id': str(chunk['chunk_id']),
                    'level': int(chunk['level']),
                    'heading': str(chunk['heading']),
                    'page_number': int(chunk['page_number']),
                    'element_type': str(chunk['element_type']),
                    'length': int(chunk['length']),
                    'paragraph_id': str(chunk['paragraph_id']),
                    'citation_text': str(chunk['citation_text']),
                    'section_title': str(chunk['section_info']['section_title']),
                    'section_number': str(chunk['section_info']['section_number']),
                    'section_level': int(chunk['section_info']['section_level']),
                    'content_type': str(chunk['content_type'])
                }
                metadatas.append(metadata)
            except Exception as e:
                logger.warning(f"Error preparing metadata for chunk {chunk.get('id', 'unknown')}: {e}")
                # Skip this chunk rather than failing the entire batch
                continue
        
        return metadatas

    def _get_or_create_collection(self, client):
        """Get or create collection with error handling"""
        try:
            return client.get_collection(self.collection_name)
        except Exception:
            try:
                return client.create_collection(name=self.collection_name)
            except Exception as e:
                raise DatabaseError(f"Failed to get or create collection: {e}")

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        memory_info = self.memory_monitor.get_memory_info()
        
        return {
            'memory_usage_mb': memory_info['used_mb'],
            'memory_available_mb': memory_info['available_mb'],
            'memory_usage_percent': memory_info['usage_percent'],
            'hash_cache_size': len(self.hash_cache.cache),
            'connection_pool_size': len(self.connection_pool.pool),
            'optimal_batch_size': self.memory_monitor.get_optimal_batch_size(64, 16, 256)
        }

    def __del__(self):
        """Destructor with cleanup"""
        self.cleanup()

    def assign_page_numbers_to_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Assigns page numbers to each chunk based on [[PAGE:X]] markers in the text.
        Modifies each chunk in-place to add a 'page_number' field.
        """
        page_marker_pattern = re.compile(r"\[\[PAGE:(\d+)\]\]")
        current_page = 1
        for chunk in chunks:
            # Search for a page marker in the chunk text
            match = page_marker_pattern.search(chunk['text'])
            if match:
                current_page = int(match.group(1))
            chunk['page_number'] = current_page
        return chunks