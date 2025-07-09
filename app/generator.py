"""
RAG Response Generator - Simple generator that takes retrieved docs and generates responses
"""

from typing import List, Dict, Any
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Generator:
    
    def __init__(self, 
                 ollama_model: str = "qwen3",
                 ollama_url: str = None,
                 temperature: float = 0.3):
        """
        Initialize the Response Generator
        
        Args:
            ollama_model: Ollama model name for generation
            ollama_url: URL for remote Ollama instance (optional)
            temperature: Model temperature for response generation
        """
        self.ollama_model = ollama_model
        self.temperature = temperature
        
        # Initialize Ollama LLM
        if ollama_url:
            self.llm = OllamaLLM(
                model=ollama_model,
                base_url=ollama_url,
                temperature=temperature
            )
        else:
            self.llm = OllamaLLM(
                model=ollama_model,
                temperature=temperature
            )
        
        # Create prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Create LangChain chain
        self.chain = (
            self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        logger.info(f"Generator initialized with model: {ollama_model}")

    def _create_prompt_template(self) -> PromptTemplate:

        """Create a prompt template for the RAG system"""

        template = """ You are a legal assistant for Pakistan's Constitution. Answer using ONLY the provided context.
        CONTEXT: {context}
        QUESTION: {question}
        RULES:

        Use ONLY information from the context - no external knowledge
        Quote exact Articles, chapters, and sections with citations: [Article number, Chapter Number]
        Use quotation marks for direct text excerpts
        If information is missing, state: "Context insufficient for [specific aspect]"
        For partial answers, specify what's available vs. missing
        Maintain exact constitutional language and formatting
        Structure multi-part answers with clear numbering
        If question is not related to Pakistan's 1973 Constitution, respond: "This question is outside my domain. I only answer questions about Pakistan's 1973 Constitution."

        RESPONSE: [Answer here following above rules]
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def generate_response(self, query: str, retrieval_results: List) -> str:
        """
        Generate a response using retrieval results from your retriever
        
        Args:
            query: User question
            retrieval_results: List of RetrievalResult objects from retriever
            
        Returns:
            Generated answer string
        """
        try:
            if not retrieval_results:
                return "I couldn't find relevant information in the Pakistan Constitution database to answer your question."
            
            # Prepare context from retrieval results
            context = self._prepare_context(retrieval_results)
            
            # Generate response using LangChain
            logger.info("Generating response using language model...")
            response = self.chain.invoke({
                "context": context,
                "question": query
            })
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return 
    
    def _prepare_context(self, retrieval_results: List) -> str:
        """
        Prepare context from retrieval results
        
        Args:
            retrieval_results: List of RetrievalResult objects
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, result in enumerate(retrieval_results):
            # Add document content with numbering and metadata
            context_part = f"Document {i+1}:\n"
            
            # Add hierarchical path if available
            if hasattr(result, 'hierarchical_path') and result.hierarchical_path:
                context_part += f"Path: {result.hierarchical_path}\n"
            
            # Add heading if available
            if hasattr(result, 'heading') and result.heading:
                context_part += f"Heading: {result.heading}\n"
            
            # Add the main text content
            context_part += f"{result.text}\n\n"
            
            context_parts.append(context_part)
        
        return "".join(context_parts)
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the connection to Ollama"""
        try:
            test_response = self.llm.invoke("Hello, this is a test.")
            return {
                "status": "success",
                "model": self.ollama_model,
                "test_response": test_response[:100]
            }
        except Exception as e:
            return {
                "status": "error",
                "model": self.ollama_model,
                "error": str(e)
            }

