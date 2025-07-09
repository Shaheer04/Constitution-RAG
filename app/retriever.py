"""
Retriever Component with Hybrid Retrieval for Constitution RAG System
"""

from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
import logging
from collections import defaultdict, Counter
import re
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FilterType(Enum):
    """Enumeration for filter types"""
    HEADING = "heading"
    TABLE = "table"
    TEXT = "text"


@dataclass
class RetrievalResult:
    """Data class for retrieval results"""
    text: str
    metadata: Dict[str, Any]
    similarity_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0
    distance: float = 1.0
    matched_keywords: List[str] = None
    hierarchical_path: str = "root"
    level: int = 0
    heading: str = ""
    page_number: int = 0
    element_type: str = "text"

    def __post_init__(self):
        if self.matched_keywords is None:
            self.matched_keywords = []


class KeywordExtractor:
    
    # Common stop words for filtering
    STOP_WORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 
        'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 
        'will', 'with', 'what', 'when', 'where', 'who', 'how', 'this', 'these', 'those',
        'shall', 'may', 'can', 'could', 'would', 'should', 'must', 'said', 'says',
        'any', 'all', 'each', 'every', 'some', 'such', 'no', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'just', 'now', 'here', 'there', 'then'
    }
    
    # Constitutional terms specific to Pakistani constitution
    CONSTITUTIONAL_TERMS = [
        'parliament', 'assembly', 'senate', 'federal', 'provincial', 'constitutional',
        'fundamental rights', 'judiciary', 'executive', 'legislative', 'amendment',
        'president', 'prime minister', 'governor', 'chief minister', 'supreme court',
        'high court', 'election', 'citizenship', 'federation', 'province', 'islamabad',
        'islamic', 'sharia', 'ordinance', 'proclamation', 'emergency', 'martial law'
    ]
    
    # Regex patterns for legal references
    LEGAL_PATTERNS = {
        'article': r'article\s+(\d+[a-z]?)',
        'part': r'part\s+([ivx]+|\d+)',
        'chapter': r'chapter\s+(\d+[a-z]?)',
        'section': r'section\s+(\d+[a-z]?)',
        'clause': r'clause\s+(\d+[a-z]?)'
    }

    @classmethod
    def extract_keywords(cls, text: str, top_k: int = 5) -> List[str]:
        """
        Extract important keywords from text using simple NLP techniques
        
        Args:
            text: Input text
            top_k: Number of top keywords to return
            
        Returns:
            List of extracted keywords
        """
        if not text or not text.strip():
            return []
            
        # Clean and tokenize text
        text_clean = text.lower()
        text_clean = re.sub(r'[^\w\s]', ' ', text_clean)  # Remove punctuation
        words = text_clean.split()
        
        # Filter words
        filtered_words = [
            word for word in words
            if (len(word) > 2 and 
                word not in cls.STOP_WORDS and 
                not word.isdigit() and 
                word.isalpha())
        ]
        
        # Count word frequencies and extract top keywords
        word_freq = Counter(filtered_words)
        keywords = [word for word, _ in word_freq.most_common(top_k)]
        
        return keywords

    @classmethod
    def extract_legal_terms(cls, text: str) -> List[str]:
        """
        Extract legal terms and constitutional references from text
        
        Args:
            text: Input text
            
        Returns:
            List of legal terms and references
        """
        if not text or not text.strip():
            return []
            
        legal_terms = []
        text_lower = text.lower()
        
        # Extract legal references using patterns
        for ref_type, pattern in cls.LEGAL_PATTERNS.items():
            matches = re.findall(pattern, text_lower)
            legal_terms.extend([f"{ref_type} {match}" for match in matches])
        
        # Extract constitutional terms
        for term in cls.CONSTITUTIONAL_TERMS:
            if term in text_lower:
                legal_terms.append(term)
        
        return list(set(legal_terms))  # Remove duplicates


class ChromaDBManager:
    
    def __init__(self, chroma_db_path: str, collection_name: str = "documents"):
        """Initialize ChromaDB manager"""
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """Get or create ChromaDB collection"""
        try:
            return self.chroma_client.get_collection(self.collection_name)
        except Exception:
            logger.info(f"Collection '{self.collection_name}' doesn't exist in the database")
            return None
    
    def query(self, query_embeddings: List[List[float]], n_results: int, 
              where_clause: Optional[Dict] = None) -> Optional[Dict]:
        """Query the ChromaDB collection"""
        if not self.collection:
            logger.error("No collection available for querying")
            return None
            
        search_params = {
            'query_embeddings': query_embeddings,
            'n_results': n_results,
            'include': ['documents', 'metadatas', 'distances']
        }
        
        if where_clause:
            search_params['where'] = where_clause
        
        try:
            return self.collection.query(**search_params)
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return None
    
    def get_all_documents(self) -> Optional[Dict]:
        """Get all documents from the collection"""
        if not self.collection:
            logger.error("No collection available")
            return None
            
        try:
            return self.collection.get(include=['documents', 'metadatas'])
        except Exception as e:
            logger.error(f"Error getting documents from ChromaDB: {e}")
            return None


class Retriever:
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 chroma_db_path: str = "./constitution_db"):
        """
        Initialize the Retriever
        
        Args:
            embedding_model_name: Name of the sentence transformer model
            chroma_db_path: Path to store ChromaDB
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.db_manager = ChromaDBManager(chroma_db_path)
        self.keyword_extractor = KeywordExtractor()
        
    def _build_where_clause(self, filter_by_level: Optional[int], 
                           filter_by_type: Optional[str]) -> Dict:
        """Build where clause for filtering"""
        where_clause = {}
        if filter_by_level is not None:
            where_clause['level'] = filter_by_level
        if filter_by_type is not None:
            where_clause['element_type'] = {"$contains": filter_by_type}
        return where_clause
    
    def _format_similarity_results(self, results: Dict) -> List[RetrievalResult]:
        """Format similarity search results"""
        formatted_results = []
        
        if not results or not results.get('documents') or not results['documents'][0]:
            return formatted_results
            
        for i in range(len(results['documents'][0])):
            metadata = results['metadatas'][0][i]
            result = RetrievalResult(
                text=results['documents'][0][i],
                metadata=metadata,
                distance=results['distances'][0][i],
                similarity_score=1.0 - results['distances'][0][i],
                hierarchical_path=metadata.get('hierarchical_path', 'root'),
                level=metadata.get('level', 0),
                heading=metadata.get('heading', ''),
                page_number=metadata.get('page_number', 0),
                element_type=metadata.get('element_type', 'text')
            )
            formatted_results.append(result)
        
        return formatted_results
    
    def retrieve_relevant_chunks(self, query: str, n_results: int = 10, 
                               filter_by_level: Optional[int] = None, 
                               filter_by_type: Optional[str] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks based on query with hierarchical filtering
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_by_level: Filter by hierarchical level
            filter_by_type: Filter by element type
            
        Returns:
            List of relevant chunks with hierarchical metadata
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []
            
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Build where clause for filtering
        where_clause = self._build_where_clause(filter_by_level, filter_by_type)
        
        # Query ChromaDB
        results = self.db_manager.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where_clause=where_clause if where_clause else None
        )
        
        if not results:
            return []
            
        return self._format_similarity_results(results)
    
    def keyword_search(self, keywords: List[str], n_results: int = 10) -> List[RetrievalResult]:
        """
        Perform keyword-based search in the document collection
        
        Args:
            keywords: List of keywords to search for
            n_results: Number of results to return
            
        Returns:
            List of matching chunks with keyword scores
        """
        if not keywords:
            return []
        
        all_docs = self.db_manager.get_all_documents()
        if not all_docs or not all_docs.get('documents'):
            return []
        
        keyword_matches = []
        
        for i, doc_text in enumerate(all_docs['documents']):
            doc_text_lower = doc_text.lower()
            keyword_score = 0
            matched_keywords = []
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                
                # Count exact matches (weighted higher)
                exact_matches = doc_text_lower.count(keyword_lower)
                if exact_matches > 0:
                    keyword_score += exact_matches * 2
                    matched_keywords.append(keyword)
                
                # Count word boundary matches
                boundary_matches = len(re.findall(r'\b' + re.escape(keyword_lower) + r'\b', doc_text_lower))
                if boundary_matches > 0:
                    keyword_score += boundary_matches
                    if keyword not in matched_keywords:
                        matched_keywords.append(keyword)
            
            if keyword_score > 0:
                metadata = all_docs['metadatas'][i]
                result = RetrievalResult(
                    text=doc_text,
                    metadata=metadata,
                    keyword_score=keyword_score,
                    matched_keywords=matched_keywords,
                    hierarchical_path=metadata.get('hierarchical_path', 'root'),
                    level=metadata.get('level', 0),
                    heading=metadata.get('heading', ''),
                    page_number=metadata.get('page_number', 0),
                    element_type=metadata.get('element_type', 'text')
                )
                keyword_matches.append(result)
        
        # Sort by keyword score and return top results
        keyword_matches.sort(key=lambda x: x.keyword_score, reverse=True)
        return keyword_matches[:n_results]
    
    def _combine_search_results(self, similarity_results: List[RetrievalResult], 
                               keyword_results: List[RetrievalResult]) -> Dict[str, RetrievalResult]:
        """Combine similarity and keyword search results"""
        combined_results = {}
        
        # Add similarity results
        for i, result in enumerate(similarity_results):
            doc_id = result.metadata.get('chunk_id', str(i))
            combined_results[doc_id] = result
        
        # Merge keyword results
        for result in keyword_results:
            doc_id = result.metadata.get('chunk_id', 'unknown')
            
            if doc_id in combined_results:
                # Update existing result
                combined_results[doc_id].keyword_score = result.keyword_score
                combined_results[doc_id].matched_keywords = result.matched_keywords
            else:
                # Add new result with max distance for keyword-only results
                result.distance = 1.0
                combined_results[doc_id] = result
        
        return combined_results
    
    def _calculate_combined_scores(self, combined_results: Dict[str, RetrievalResult], 
                                 similarity_weight: float, keyword_weight: float) -> None:
        """Calculate combined scores for hybrid retrieval"""
        if not combined_results:
            return
            
        # Get max scores for normalization
        max_similarity = max(r.similarity_score for r in combined_results.values())
        max_keyword = max(r.keyword_score for r in combined_results.values())
        
        # Calculate combined scores
        for result in combined_results.values():
            norm_similarity = result.similarity_score / max_similarity if max_similarity > 0 else 0
            norm_keyword = result.keyword_score / max_keyword if max_keyword > 0 else 0
            
            result.combined_score = (
                similarity_weight * norm_similarity + 
                keyword_weight * norm_keyword
            )
    
    def hybrid_retrieve(self, query: str, n_results: int = 10, 
                       similarity_weight: float = 0.7, 
                       keyword_weight: float = 0.3,
                       filter_by_level: Optional[int] = None, 
                       filter_by_type: Optional[str] = None) -> List[RetrievalResult]:
        """
        Hybrid retrieval combining similarity search and keyword matching
        
        Args:
            query: Search query
            n_results: Number of results to return
            similarity_weight: Weight for similarity search results (0-1)
            keyword_weight: Weight for keyword search results (0-1)
            filter_by_level: Filter by hierarchical level
            filter_by_type: Filter by element type
            
        Returns:
            List of relevant chunks with combined scores
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []
            
        # Normalize weights
        total_weight = similarity_weight + keyword_weight
        if total_weight == 0:
            logger.error("Both weights cannot be zero")
            return []
            
        similarity_weight = similarity_weight / total_weight
        keyword_weight = keyword_weight / total_weight
        
        # 1. Similarity search
        similarity_results = self.retrieve_relevant_chunks(
            query, n_results * 2, filter_by_level, filter_by_type
        )
        
        # 2. Extract keywords and legal terms from query
        query_keywords = self.keyword_extractor.extract_keywords(query, top_k=8)
        legal_terms = self.keyword_extractor.extract_legal_terms(query)
        all_keywords = query_keywords + legal_terms
        
        # 3. Keyword search
        keyword_results = self.keyword_search(all_keywords, n_results * 2)
        
        # 4. Combine results
        combined_results = self._combine_search_results(similarity_results, keyword_results)
        
        # 5. Calculate combined scores
        self._calculate_combined_scores(combined_results, similarity_weight, keyword_weight)
        
        # 6. Sort by combined score and return top results
        final_results = sorted(
            combined_results.values(), 
            key=lambda x: x.combined_score, 
            reverse=True
        )
        
        return final_results[:n_results]
