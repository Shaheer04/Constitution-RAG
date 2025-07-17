"""
RAG Response Generator Using Ollama LLM
"""

import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests

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
                 ollama_model: str = "qwen3",
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
        template = """You are a concise legal expert on the 1973 Constitution of Pakistan. Answer only from the provided context; never add external facts.

        Format rules (strict):
        1. Answer length: 50 -- 300 words. If context is insufficient, state what is missing.
        2. If context is insufficient, state what is missing.
        2. Cite every fact with its source number in superscript immediately after the clause, e.g. … right to life¹ … right to privacy². Reference list is appended automatically; do not write it inside the answer.
        3. Prefer direct quotes (≤25 words) enclosed in double quotes.
        4. Use formal constitutional language.
        5. If the question is unrelated, reply: This question is outside the scope of the 1973 Constitution of Pakistan.

        Context:
        {context}

        Question:
        {question}

        Answer (cite with superscripts):"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def _prepare_context(self, retrieval_results: List) -> Tuple[str, str]:
        context_parts = []
        current_size = 0
        truncated = False
        for i, result in enumerate(retrieval_results):
            citation_id = f"[{i+1}]"
            context_part = f"Source {citation_id}:\n"
            if hasattr(result, 'hierarchical_path') and result.hierarchical_path:
                context_part += f"Path: {result.hierarchical_path}\n"
            if hasattr(result, 'heading') and result.heading:
                context_part += f"Section: {result.heading}\n"
            if hasattr(result, 'page_number'):
                context_part += f"Page: {result.page_number}\n"
            context_part += f"Content: {result.text}\n\n"
            if current_size + len(context_part) > self.max_context_size:
                remaining_space = self.max_context_size - current_size
                if remaining_space > 100:
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
        context_hash = hashlib.md5(context_string.encode()).hexdigest()
        return context_string, context_hash

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
        if not retrieval_results:
            return "I couldn't find relevant information in the Pakistan Constitution database to answer your question."
        context, context_hash = self._prepare_context(retrieval_results)
        cached_response = self.cache.get(query, context_hash)
        if cached_response:
            return cached_response
        input_data = {
            "context": context,
            "question": query
        }
        try:
            response = self._invoke_with_retry(input_data)
            self.cache.set(query, context_hash, response)
            return response
        except (TimeoutError, ConnectionError, ModelError) as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error while generating the response: {str(e)}. Please try again."
        except Exception as e:
            logger.error(f"Unexpected error generating response: {str(e)}")
            return "I apologize, but I encountered an unexpected error while generating the response. Please try again."
