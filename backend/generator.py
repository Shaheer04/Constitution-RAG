"""
RAG Response Generator Using Ollama LLM
"""

import hashlib
import os
import time
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests

from backend.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom exceptions for better error handling
class GeneratorError(Exception):
    pass

class ConnectionError(GeneratorError):
    pass

class TimeoutError(GeneratorError):
    pass

class ModelError(GeneratorError):
    pass

class ResponseCache:
    """In-memory cache for responses with TTL support"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[str, datetime]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

    def _generate_key(self, query: str, context_hash: str) -> str:
        combined = f"{query}:{context_hash}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _is_expired(self, timestamp: datetime) -> bool:
        return datetime.now() - timestamp > timedelta(seconds=self.ttl_seconds)

    def _cleanup_expired(self):
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if self._is_expired(timestamp)
        ]
        for key in expired_keys:
            del self.cache[key]

    def get(self, query: str, context_hash: str) -> Optional[str]:
        key = self._generate_key(query, context_hash)
        if key in self.cache:
            response, timestamp = self.cache[key]
            if not self._is_expired(timestamp):
                return response
            else:
                del self.cache[key]
        return None

    def set(self, query: str, context_hash: str, response: str):
        if len(self.cache) > self.max_size * 0.8:
            self._cleanup_expired()
        if len(self.cache) >= self.max_size:
            sorted_items = sorted(self.cache.items(), key=lambda x: x[1][1])
            num_to_remove = max(1, len(sorted_items) // 5)
            for key, _ in sorted_items[:num_to_remove]:
                del self.cache[key]
        key = self._generate_key(query, context_hash)
        self.cache[key] = (response, datetime.now())

class Generator:
    def __init__(self, 
                 ollama_model: str,
                 ollama_url: str = None,
                 temperature: float = 0.3,
                 max_context_size: int = 8000,
                 cache_size: int = 100,
                 cache_ttl: int = 3600):
        
        self.ollama_model = ollama_model
        self.temperature = temperature
        self.ollama_url = ollama_url
        self.max_context_size = max_context_size
        self.cache = ResponseCache(max_size=cache_size, ttl_seconds=cache_ttl)
        self._initialize_llm()
        self.prompt_template = self._create_prompt_template()
        self.chain = (
            self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        self._validate_connection()
        logger.info(f"Generator initialized with model: {ollama_model}")

    def _initialize_llm(self):
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
        try:
            if self.ollama_url:
                print(self.ollama_url)
                response = requests.get(
                    f"{self.ollama_url}/api/version",
                    timeout=5
                )
                if response.status_code != 200:
                    raise ConnectionError(f"Ollama server returned status {response.status_code}")
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
        template ="""
        You are a concise constitutional interpreter on the 1973 Constitution of Pakistan.  
        Answer **only** from the numbered snippets below.

        Rules:
        1.  40-150 words.  
        2.  **Quote** the relevant **Article number** and **≤40 words** from the snippet.  
        3.  Cite **only** the **source index** in superscript immediately after the clause, e.g. “Article 19 guarantees freedom of speech[1]”.  
        4.  Do **not** mention “References”, “Page”, or any external context.  
        5.  If snippets lack the answer, state: Available provisions do not address this query.

        Numbered context:
        {context}

        Question: {question}

        Answer (≤150 words, cite index):
        """
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def _prepare_context(
        self, retrieval_results: List
    ) -> Tuple[str, Dict[int, int], str]:
        """
        Returns:
            context_string   : numbered text for the LLM prompt
            index_to_page    : {source_idx : page_number} for footnotes
            context_hash     : hash for caching
        """
        snippets = []
        index_to_page = {}          # 1-based index → page

        for idx, res in enumerate(retrieval_results, 1):
            index_to_page[idx] = res.page_number
            snippets.append(f"[{idx}] {res.text.strip()}")

        context_str = "\n\n".join(snippets)
        context_hash = hashlib.md5(context_str.encode()).hexdigest()
        return context_str, index_to_page, context_hash

    def _make_clickable(self, index_to_page: Dict[int, int]) -> str:
        """Make references clickable links to the PDF viewer."""
        
        # Use settings for PDF path and file name
        pdf_path = str(settings.raw_pdf_dir / "constitution-1973.pdf")
        pdf_file = os.path.basename(pdf_path)
        base_url = "http://localhost:8000"
        refs = [
            f'<a href="{base_url}/{pdf_file}#page={p}" target="_blank">{i}</a>'
            for i, p in sorted(index_to_page.items())
        ]
        return "<br><b>References:</b> " + ", ".join(refs)

    def _invoke_with_retry(self, input_data: Dict[str, str]) -> str:
        last_exception = None
        for attempt in range(3):
            try:
                if attempt > 0:
                    delay = min(2 ** attempt, 16)
                    logger.info(f"Retrying request after {delay}s delay (attempt {attempt + 1}/3)")
                    time.sleep(delay)
                response = self.chain.invoke(input_data)
                return response
            except Exception as e:
                last_exception = e
                logger.warning(f"LLM request failed (attempt {attempt + 1}/3): {str(e)}")
                if "timeout" in str(e).lower():
                    if attempt == 2:
                        raise TimeoutError(f"Request timed out after 3 attempts")
                elif "connection" in str(e).lower():
                    if attempt == 2:
                        raise ConnectionError(f"Connection failed after 3 attempts")
                else:
                    if attempt == 2:
                        raise ModelError(f"Model error after 3 attempts: {str(e)}")
        raise ModelError(f"Request failed after 3 attempts: {str(last_exception)}")

    def generate_response(self, query: str, retrieval_results: List) -> str:
        """
        Generate a response with numbered citations and clickable footnotes.
        """
        if not retrieval_results:
            return "I couldn't find relevant information in the Pakistan Constitution database to answer your question."

        context_str, index_to_page, context_hash = self._prepare_context(retrieval_results)

        cached_response = self.cache.get(query, context_hash)
        if cached_response:
            return cached_response
        
        input_data = {
            "context": context_str,
            "question": query
        }

        try:
            answer = self._invoke_with_retry(input_data)
        except (TimeoutError, ConnectionError, ModelError) as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error while generating the response: {str(e)}. Please try again."
        except Exception as e:
            logger.error(f"Unexpected error generating response: {str(e)}")
            return "I apologize, but I encountered an unexpected error while generating the response. Please try again."

        # 4) Append clickable footnotes
        footnotes = self._make_clickable(index_to_page)
        full_answer = answer + footnotes

        # 5) Cache
        self.cache.set(query, context_hash, full_answer)
        return full_answer
