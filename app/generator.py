"""
RAG Response Generator - Optimized generator with performance improvements
"""

import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime, timedelta

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError
from .citation_generator import CitationLinkGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom exceptions for better error handling
class GeneratorError(Exception):
    """Base exception for generator errors"""
    pass

class ConnectionError(GeneratorError):
    """Raised when Ollama connection fails"""
    pass

class TimeoutError(GeneratorError):
    """Raised when request times out"""
    pass

class ModelError(GeneratorError):
    """Raised when model encounters an error"""
    pass

class ContextTooLargeError(GeneratorError):
    """Raised when context exceeds maximum size"""
    pass

class ResponseCache:
    """Simple in-memory cache for responses with TTL support"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[str, datetime]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.access_count = 0
        self.hit_count = 0
    
    def _generate_key(self, query: str, context_hash: str) -> str:
        """Generate cache key from query and context"""
        combined = f"{query}:{context_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _is_expired(self, timestamp: datetime) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() - timestamp > timedelta(seconds=self.ttl_seconds)
    
    def _cleanup_expired(self):
        """Remove expired entries from cache"""
        current_time = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if self._is_expired(timestamp)
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def get(self, query: str, context_hash: str) -> Optional[str]:
        """Get cached response if exists and not expired"""
        self.access_count += 1
        key = self._generate_key(query, context_hash)
        
        if key in self.cache:
            response, timestamp = self.cache[key]
            if not self._is_expired(timestamp):
                self.hit_count += 1
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return response
            else:
                # Remove expired entry
                del self.cache[key]
        
        return None
    
    def set(self, query: str, context_hash: str, response: str):
        """Cache response with current timestamp"""
        # Cleanup expired entries periodically
        if len(self.cache) > self.max_size * 0.8:
            self._cleanup_expired()
        
        # If still too large, remove oldest entries
        if len(self.cache) >= self.max_size:
            # Remove 20% of oldest entries
            sorted_items = sorted(self.cache.items(), key=lambda x: x[1][1])
            num_to_remove = max(1, len(sorted_items) // 5)
            for key, _ in sorted_items[:num_to_remove]:
                del self.cache[key]
        
        key = self._generate_key(query, context_hash)
        self.cache[key] = (response, datetime.now())
        logger.debug(f"Cached response for query: {query[:50]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = (self.hit_count / self.access_count) if self.access_count > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "access_count": self.access_count,
            "hit_count": self.hit_count,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds
        }

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0,
            "total_response_time": 0,
            "cache_hits": 0,
            "context_truncations": 0,
            "retry_attempts": 0
        }
    
    def record_request(self, success: bool, response_time: float, cache_hit: bool = False, 
                      context_truncated: bool = False, retry_count: int = 0):
        """Record request metrics"""
        self.metrics["total_requests"] += 1
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        self.metrics["total_response_time"] += response_time
        self.metrics["average_response_time"] = (
            self.metrics["total_response_time"] / self.metrics["total_requests"]
        )
        
        if cache_hit:
            self.metrics["cache_hits"] += 1
        
        if context_truncated:
            self.metrics["context_truncations"] += 1
        
        self.metrics["retry_attempts"] += retry_count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics.copy()
    
    def log_metrics(self):
        """Log current metrics"""
        logger.info(f"Performance Metrics: {self.metrics}")

class Generator:
    
    def __init__(self, 
                 ollama_model: str = "qwen3",
                 ollama_url: str = None,
                 temperature: float = 0.3,
                 pdf_path: str = None,
                 base_url: str = "",
                 request_timeout: int = 30,
                 max_retries: int = 3,
                 max_context_size: int = 8000,
                 cache_size: int = 100,
                 cache_ttl: int = 3600):
        """
        Initialize the Response Generator with performance optimizations
        
        Args:
            ollama_model: Ollama model name for generation
            ollama_url: URL for remote Ollama instance (optional)
            temperature: Model temperature for response generation
            pdf_path: Path to the constitution PDF for citations
            base_url: Base URL for PDF links (optional)
            request_timeout: Timeout for API requests in seconds
            max_retries: Maximum number of retry attempts
            max_context_size: Maximum context size in characters
            cache_size: Maximum number of cached responses
            cache_ttl: Cache time-to-live in seconds
        """
        self.ollama_model = ollama_model
        self.temperature = temperature
        self.ollama_url = ollama_url
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.max_context_size = max_context_size
        
        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.cache = ResponseCache(max_size=cache_size, ttl_seconds=cache_ttl)
        
        # Initialize Ollama LLM with timeout
        self._initialize_llm()
        
        # Create prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Create LangChain chain
        self.chain = (
            self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        # Validate connection on initialization
        self._validate_connection()
        
        logger.info(f"Generator initialized with model: {ollama_model}")
        logger.info(f"Performance features: timeout={request_timeout}s, retries={max_retries}, "
                   f"context_limit={max_context_size}, cache_size={cache_size}")

    def _initialize_llm(self):
        """Initialize Ollama LLM with proper configuration"""
        try:
            llm_kwargs = {
                "model": self.ollama_model,
                "temperature": self.temperature,
            }
            
            if self.ollama_url:
                llm_kwargs["base_url"] = self.ollama_url
            
            self.llm = OllamaLLM(**llm_kwargs)
            logger.info("Ollama LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama LLM: {str(e)}")
            raise ModelError(f"Failed to initialize model: {str(e)}")

    def _validate_connection(self):
        """Validate Ollama connection before use"""
        try:
            if self.ollama_url:
                # Test HTTP connection for remote Ollama
                response = requests.get(
                    f"{self.ollama_url}/api/version",
                    timeout=5
                )
                if response.status_code != 200:
                    raise ConnectionError(f"Ollama server returned status {response.status_code}")
            
            # Test model invocation
            test_response = self.llm.invoke("test")
            if not test_response:
                raise ConnectionError("Model returned empty response")
                
            logger.info("Ollama connection validated successfully")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama connection validation failed: {str(e)}")
            raise ConnectionError(f"Cannot connect to Ollama: {str(e)}")
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            raise ModelError(f"Model validation failed: {str(e)}")

    def _create_prompt_template(self) -> PromptTemplate:
        """Create a prompt template for the RAG system"""
        template = """ You are a legal assistant for Pakistan's Constitution. Answer using ONLY the provided context.
    
    CONTEXT: {context}
    QUESTION: {question}
    
    RULES:
    - Use ONLY information from the context - no external knowledge
    - Quote exact Articles, chapters, and sections with citations
    - Use quotation marks for direct text excerpts
    - If information is missing, state: "Context insufficient for [specific aspect]"
    - For partial answers, specify what's available vs. missing
    - Maintain exact constitutional language and formatting
    - Structure multi-part answers with clear numbering
    - If question is not related to Pakistan's 1973 Constitution, respond: "This question is outside my domain. I only answer questions about Pakistan's 1973 Constitution."
    - Do not include any thinking process or reasoning in your response
    - Only provide the final answer with citations inline.

    RESPONSE: [Answer here following above rules]
    """
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "pdf_path"]
        )

    def _prepare_context(self, retrieval_results: List) -> Tuple[str, str]:
        """
        Prepare context from retrieval results with size limits and truncation.
        Includes PDF path and relevant page numbers.
        
        Args:
            retrieval_results: List of retrieval results.
            pdf_path: Path to the source PDF.
            page_numbers: List of relevant page numbers.
        
        Returns:
            Tuple of (context_string, context_hash)
        """
        context_parts = []
        current_size = 0
        truncated = False
        
        for i, result in enumerate(retrieval_results):
            # Add document content with citation numbering
            citation_id = f"[{i+1}]"
            context_part = f"Source {citation_id}:\n"
            
            # Add hierarchical path if available
            if hasattr(result, 'hierarchical_path') and result.hierarchical_path:
                context_part += f"Path: {result.hierarchical_path}\n"
            
            # Add heading if available
            if hasattr(result, 'heading') and result.heading:
                context_part += f"Section: {result.heading}\n"
            
            # Add page number if available
            if hasattr(result, 'page_number'):
                context_part += f"Page: {result.page_number}\n"
            
            # Add the main text content
            context_part += f"Content: {result.text}\n\n"
            
            # Check if adding this part would exceed the limit
            if current_size + len(context_part) > self.max_context_size:
                # Try to fit partial content
                remaining_space = self.max_context_size - current_size
                if remaining_space > 100:  # Only add if meaningful space remains
                    partial_text = result.text[:remaining_space - 50]
                    context_part = f"Source {citation_id}:\n"
                    if hasattr(result, 'hierarchical_path') and result.hierarchical_path:
                        context_part += f"Path: {result.hierarchical_path}\n"
                    if hasattr(result, 'heading') and result.heading:
                        context_part += f"Section: {result.heading}\n"
                    context_part += f"Content: {partial_text}...\n\n"
                    context_parts.append(context_part)
                
                truncated = True
                break
            
            context_parts.append(context_part)
            current_size += len(context_part)
        
        context_string = "".join(context_parts)
        
        if truncated:
            context_string += "\n[NOTE: Context was truncated due to size limits]"
            logger.warning(f"Context truncated to {len(context_string)} characters")
            self.performance_monitor.record_request(
                success=True, response_time=0, context_truncated=True
            )
        
        # Generate hash for caching
        context_hash = hashlib.md5(context_string.encode()).hexdigest()
        
        return context_string, context_hash

    def _add_citations_optimized(self, response: str, retrieval_results: list, pdf_path: str, page_numbers: list) -> str:
        """
        Add citations to the response using optimized processing.
        Includes PDF path and relevant page numbers.
        """
        # Use unique and sorted page numbers
        page_numbers = sorted(set(page_numbers))
        reference_links = []
        for page_num in page_numbers:
            link = f"{pdf_path}#page={page_num}"
            reference_links.append(
                f'<a href="{link}" target="_blank" class="pdf-reference">📄 Page {page_num}</a>'
            )

        references_html = f'''
        <div class="references-section">
            <div class="references-title">📚 References:</div>
            <div>PDF Path: {pdf_path}</div>
            {"".join(reference_links)}
        </div>
        '''

        enhanced_response = response.strip() + "\n\n" + references_html
        return enhanced_response

    def _invoke_with_retry(self, input_data: Dict[str, str]) -> str:
        """
        Invoke the LLM with retry mechanism and exponential backoff
        
        Args:
            input_data: Input data for the LLM
            
        Returns:
            Generated response string
            
        Raises:
            TimeoutError: If request times out after all retries
            ModelError: If model encounters an error after all retries
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                # Calculate backoff delay
                if attempt > 0:
                    delay = min(2 ** attempt, 16)  # Exponential backoff, max 16 seconds
                    logger.info(f"Retrying request after {delay}s delay (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                
                # Make the request with timeout
                start_time = time.time()
                response = self.chain.invoke(input_data)
                response_time = time.time() - start_time
                
                logger.debug(f"LLM request completed in {response_time:.2f}s")
                return response
                
            except Exception as e:
                last_exception = e
                logger.warning(f"LLM request failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                
                # Don't retry on certain error types
                if "timeout" in str(e).lower():
                    if attempt == self.max_retries - 1:
                        raise TimeoutError(f"Request timed out after {self.max_retries} attempts")
                elif "connection" in str(e).lower():
                    if attempt == self.max_retries - 1:
                        raise ConnectionError(f"Connection failed after {self.max_retries} attempts")
                else:
                    if attempt == self.max_retries - 1:
                        raise ModelError(f"Model error after {self.max_retries} attempts: {str(e)}")
        
        # This should not be reached, but just in case
        raise ModelError(f"Request failed after {self.max_retries} attempts: {str(last_exception)}")

    def generate_response(self, query: str, retrieval_results: List) -> str:
        """
        Generate a response using retrieval results with performance optimizations.
        Includes PDF path and relevant page numbers in the context.
        
        Args:
            query: User question
            retrieval_results: List of RetrievalResult objects from retriever
            pdf_path: Path to the source PDF
            page_numbers: List of relevant page numbers
            
        Returns:
            Generated answer string
        """
        start_time = time.time()
        cache_hit = False
        context_truncated = False
        retry_count = 0
        
        try:
            if not retrieval_results:
                return "I couldn't find relevant information in the Pakistan Constitution database to answer your question."
            
            # Prepare context from retrieval results
            context, context_hash = self._prepare_context(retrieval_results)
            
            # Check cache first
            cached_response = self.cache.get(query, context_hash)
            if cached_response:
                cache_hit = True
                response_time = time.time() - start_time
                self.performance_monitor.record_request(
                    success=True, response_time=response_time, cache_hit=True
                )
                return cached_response
            
            # Validate connection before making request
            self._validate_connection()
            
            # Generate response using LangChain with retry mechanism
            logger.info("Generating response using language model...")
            
            input_data = {
                "context": context,
                "question": query
            }
            
            response = self._invoke_with_retry(input_data)
            
            # Cache the response
            self.cache.set(query, context_hash, response)
            
            # Record performance metrics
            response_time = time.time() - start_time
            self.performance_monitor.record_request(
                success=True, response_time=response_time, cache_hit=cache_hit,
                context_truncated=context_truncated, retry_count=retry_count
            )
            
            return response
            
        except (TimeoutError, ConnectionError, ModelError) as e:
            response_time = time.time() - start_time
            self.performance_monitor.record_request(
                success=False, response_time=response_time, cache_hit=cache_hit,
                context_truncated=context_truncated, retry_count=retry_count
            )
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error while generating the response: {str(e)}. Please try again."
            
        except Exception as e:
            response_time = time.time() - start_time
            self.performance_monitor.record_request(
                success=False, response_time=response_time, cache_hit=cache_hit,
                context_truncated=context_truncated, retry_count=retry_count
            )
            logger.error(f"Unexpected error generating response: {str(e)}")
            return "I apologize, but I encountered an unexpected error while generating the response. Please try again."

    def test_connection(self) -> Dict[str, Any]:
        """Test the connection to Ollama with enhanced validation"""
        try:
            start_time = time.time()
            
            # Test basic connection
            if self.ollama_url:
                response = requests.get(
                    f"{self.ollama_url}/api/version",
                    timeout=5
                )
                if response.status_code != 200:
                    return {
                        "status": "error",
                        "model": self.ollama_model,
                        "error": f"HTTP {response.status_code}"
                    }
            
            # Test model invocation
            test_response = self.llm.invoke("Hello, this is a test.")
            response_time = time.time() - start_time
            
            return {
                "status": "success",
                "model": self.ollama_model,
                "response_time": round(response_time, 2),
                "test_response": test_response[:100],
                "cache_stats": self.cache.get_stats(),
                "performance_metrics": self.performance_monitor.get_metrics()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "model": self.ollama_model,
                "error": str(e)
            }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            "cache_stats": self.cache.get_stats(),
            "performance_metrics": self.performance_monitor.get_metrics(),
            "configuration": {
                "max_context_size": self.max_context_size,
                "request_timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "cache_size": self.cache.max_size,
                "cache_ttl": self.cache.ttl_seconds
            }
        }

    def clear_cache(self):
        """Clear the response cache"""
        self.cache.cache.clear()
        self.cache.access_count = 0
        self.cache.hit_count = 0
        logger.info("Response cache cleared")

    def __del__(self):
        """Cleanup and log final performance metrics"""
        try:
            self.performance_monitor.log_metrics()
        except:
            pass
