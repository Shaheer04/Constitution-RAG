"""
Smart Retriever for querying pre-built Chroma vectorstore
"""

from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartRetriever:
    """Handles retrieval from pre-built Chroma database - use for queries"""
    
    def __init__(self, persist_directory: str = "./vectorstore_db", collection_name: str = "pakistan_constitution"):
        """
        Initialize SmartRetriever
        
        Args:
            persist_directory: Path where the Chroma database is stored
            collection_name: Name of the Chroma collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore = None
        
        # Initialize Ollama embeddings (must match the ones used for creation)
        logger.info("Initializing Ollama embeddings (nomic-embed-text)...")
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
        )
        
        # Load the vectorstore
        self._load_vectorstore()
    
    def _load_vectorstore(self):
        """Load the pre-built Chroma vectorstore from local storage"""
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(
                f"Vectorstore not found at {self.persist_directory}. "
                f"Please create it first using DocumentProcessor."
            )
        
        try:
            logger.info(f"Loading Chroma vectorstore from {self.persist_directory}...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            # Test the connection
            collection = self.vectorstore._collection
            doc_count = collection.count()
            logger.info(f"Vectorstore loaded successfully with {doc_count} documents")
            
        except Exception as e:
            error_msg = f"Failed to load vectorstore: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def smart_retrieve(self, query: str, max_results: int = 3) -> List[Document]:
        """
        Implement intelligent retrieval based on query type
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant Document objects
        """
        if self.vectorstore is None:
            raise RuntimeError("Vectorstore not loaded. Check if database exists.")
        
        logger.info(f"Processing query: '{query}'")
        
        try:
            # Determine query type and retrieve accordingly
            if self._is_specific_article_query(query):
                logger.info("Detected specific article query")
                results = self._retrieve_specific_article(query, max_results)
                
            elif self._is_topic_based_query(query):
                logger.info("Detected topic-based query")
                results = self._retrieve_topic_based(query, max_results)
                
            else:
                logger.info("Using multi-level search")
                results = self._retrieve_multi_level(query, max_results)
            
            logger.info(f"Retrieved {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            # Fallback to basic similarity search
            logger.info("Falling back to basic similarity search")
            return self.vectorstore.similarity_search(query, k=max_results)
    
    def _retrieve_specific_article(self, query: str, max_results: int) -> List[Document]:
        """Retrieve specific articles"""
        # First try to get the specific article
        article_results = self.vectorstore.similarity_search(
            query, 
            k=max_results,
            filter={'chunk_type': 'article'}
        )
        
        # Add related sub-articles if available
        if len(article_results) < max_results:
            sub_article_results = self.vectorstore.similarity_search(
                query,
                k=max_results - len(article_results),
                filter={'chunk_type': 'sub_article'}
            )
            article_results.extend(sub_article_results)
        
        return article_results
    
    def _retrieve_topic_based(self, query: str, max_results: int) -> List[Document]:
        """Retrieve topic-based results"""
        results = []
        
        # Search at chapter level first (broader context)
        chapter_results = self.vectorstore.similarity_search(
            query, 
            k=min(3, max_results // 2),
            filter={'chunk_type': 'chapter'}
        )
        results.extend(chapter_results)
        
        # Add specific articles related to the topic
        remaining_slots = max_results - len(results)
        if remaining_slots > 0:
            article_results = self.vectorstore.similarity_search(
                query, 
                k=remaining_slots,
                filter={'chunk_type': 'article'}
            )
            results.extend(article_results)
        
        return results
    
    def _retrieve_multi_level(self, query: str, max_results: int) -> List[Document]:
        """Multi-level search across all chunk types"""
        return self.vectorstore.similarity_search(query, k=max_results)
    
    def _is_specific_article_query(self, query: str) -> bool:
        """Detect if query is asking for a specific article"""
        query_lower = query.lower()
        
        # Check for "Article X" pattern
        if re.search(r'article\s+\d+', query_lower):
            return True
        
        article_pattern = re.compile(r"^\s*(\d+)\.\s*(.+)$", re.MULTILINE)
        
        return any(re.match(article_pattern, line.strip()) for line in query_lower.splitlines())
    
    def _is_topic_based_query(self, query: str) -> bool:
        """Detect if query is topic-based"""
        topic_patterns = [
            'fundamental rights', 'basic rights', 'human rights',
            'parliamentary system', 'parliament', 'legislature',
            'judiciary', 'courts', 'supreme court', 'high court',
            'executive', 'president', 'prime minister', 'cabinet',
            'federalism', 'federal', 'provincial', 'provinces',
            'elections', 'electoral', 'voting', 'franchise',
            'citizenship', 'nationality', 'citizens',
            'emergency', 'martial law', 'proclamation',
            'equality', 'equal protection', 'discrimination',
            'freedom', 'liberty', 'speech', 'expression',
            'religion', 'religious', 'belief', 'faith'
        ]
        
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in topic_patterns)
    
    def retrieve_by_metadata(self, filter_dict: Dict[str, str], k: int = 5) -> List[Document]:
        """
        Retrieve documents based on metadata filters
        
        Args:
            filter_dict: Dictionary of metadata filters
            k: Number of results to return
            
        Returns:
            List of filtered documents
        """
        try:
            return self.vectorstore.similarity_search(
                "", 
                k=k,
                filter=filter_dict
            )
        except Exception as e:
            logger.error(f"Error in metadata retrieval: {str(e)}")
            return []
    
    def retrieve_similar_to_document(self, document: Document, k: int = 5) -> List[Document]:
        """
        Find documents similar to a given document
        
        Args:
            document: Reference document
            k: Number of similar documents to return
            
        Returns:
            List of similar documents
        """
        return self.vectorstore.similarity_search(
            document.page_content,
            k=k
        )
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the loaded database"""
        if self.vectorstore is None:
            return {"status": "not_loaded"}
        
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            # Get sample metadata to understand structure
            sample_docs = self.vectorstore.similarity_search("", k=1)
            sample_metadata = sample_docs[0].metadata if sample_docs else {}
            
            return {
                "status": "loaded",
                "document_count": count,
                "persist_directory": self.persist_directory,
                "collection_name": self.collection_name,
                "embedding_model": "nomic-embed-text",
                "sample_metadata_keys": list(sample_metadata.keys()) if sample_metadata else []
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """
        Retrieve documents with similarity scores
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            return self.vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error(f"Error in scored search: {str(e)}")
            return []


# Example usage and testing
if __name__ == "__main__":
    # Initialize retriever
    try:
        retriever = SmartRetriever(persist_directory="./pakistan_constitution_db")
        
        # Get database info
        db_info = retriever.get_database_info()
        print(f"Database info: {db_info}")
        
        # Test queries
        test_queries = [
            "What are the fundamental rights in Pakistan? and Article 25 equality",
            "Parliamentary system in Pakistan",
            "Article 25",
            "Supreme Court jurisdiction",
            "Presidential powers"
        ]
        
        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"Query: {query}")
            print('='*50)
            
            # Get results
            results = retriever.smart_retrieve(query, max_results=3)
            
            for i, doc in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"Metadata: {doc.metadata}")
                print(f"Content: {doc.page_content[:300]}...")
                print("-" * 40)
    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you've created the database first using document_processor.py")