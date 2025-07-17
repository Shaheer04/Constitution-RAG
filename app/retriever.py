"""
Enhanced Retriever Component with Improved Accuracy for Constitution RAG System
"""

from typing import List, Dict, Any, Optional, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
import logging
from dataclasses import dataclass
from enum import Enum
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langchain.retrievers import BM25Retriever
from langchain.schema import Document
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

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
    """Enhanced data class for retrieval results"""
    text: str
    metadata: Dict[str, Any]
    similarity_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0
    distance: float = 1.0
    matched_keywords: List[str] = None
    page_number: int = 0
    element_type: str = "text"
    final_score: float = 0.0

class QueryProcessor:
    """Enhanced query processing with legal domain awareness"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.legal_terms = {
            'constitution', 'amendment', 'article', 'section', 'clause',
            'law', 'legal', 'court', 'judge', 'justice', 'right', 'liberty',
            'freedom', 'government', 'federal', 'state', 'congress', 'senate',
            'house', 'president', 'judicial', 'legislative', 'executive'
        }
        
    def extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query with legal domain awareness"""
        # Remove punctuation and convert to lowercase
        cleaned_query = re.sub(r'[^\w\s]', ' ', query.lower())
        tokens = word_tokenize(cleaned_query)
        
        # Filter out stop words and lemmatize
        key_terms = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemmatized = self.lemmatizer.lemmatize(token)
                key_terms.append(lemmatized)
        
        return key_terms
    
    def identify_query_type(self, query: str) -> str:
        """Identify the type of query (factual, procedural, interpretive, etc.)"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'who', 'when', 'where', 'which']):
            return 'factual'
        elif any(word in query_lower for word in ['how', 'process', 'procedure']):
            return 'procedural'
        elif any(word in query_lower for word in ['why', 'purpose', 'intent', 'meaning']):
            return 'interpretive'
        elif any(word in query_lower for word in ['can', 'may', 'allowed', 'permitted']):
            return 'permission'
        else:
            return 'general'
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        query_variants = [query]
        
        # Add legal synonyms
        legal_synonyms = {
            'freedom': ['liberty', 'right'],
            'law': ['statute', 'regulation', 'rule'],
            'government': ['state', 'federal', 'administration'],
            'court': ['tribunal', 'judiciary', 'judicial'],
            'congress': ['legislature', 'parliament']
        }
        
        for term, synonyms in legal_synonyms.items():
            if term in query.lower():
                for synonym in synonyms:
                    query_variants.append(query.replace(term, synonym))
        
        return query_variants

class ChromaDBManager:
    def __init__(self, chroma_db_path: str, collection_name: str = "pdf_chunks"):
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

class Reranker:
    """Enhanced reranker with multiple scoring mechanisms"""

    def __init__(self, model_name: str = "nlpaueb/legal-bert-base-uncased"):
        """Initialize Advanced reranker"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.query_processor = QueryProcessor()

    def calculate_context_score(self, query: str, result: RetrievalResult, 
                              all_results: List[RetrievalResult]) -> float:
        """Calculate contextual relevance score"""
        context_score = 0.0

        # Element type relevance
        query_type = self.query_processor.identify_query_type(query)
        if query_type == 'factual' and result.element_type == 'text':
            context_score += 0.1
        elif query_type == 'procedural' and result.element_type in ['table']:
            context_score += 0.15

        return min(context_score, 1.0)

    def calculate_diversity_penalty(self, result: RetrievalResult, 
                                  selected_results: List[RetrievalResult]) -> float:
        """Calculate diversity penalty to avoid redundant results"""
        if not selected_results:
            return 0.0

        penalties = []
        for selected in selected_results:
            # Text similarity penalty
            result_words = set(result.text.lower().split())
            selected_words = set(selected.text.lower().split())
            overlap = len(result_words & selected_words) / len(result_words | selected_words)
            if overlap > 0.7:
                penalties.append(0.4)

        return max(penalties) if penalties else 0.0

    def rerank_results(self, query: str, results: List[RetrievalResult], top_k: int = 10) -> List[RetrievalResult]:
        """Optimized reranking with batch scoring and efficient penalty calculation"""
        if not results:
            return []

        # Batch LegalBERT scoring
        texts = [r.text for r in results]
        inputs = self.tokenizer(
            [query] * len(texts),
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            relevance_scores = torch.softmax(outputs.logits, dim=-1)[:, 1].tolist()
        for result, score in zip(results, relevance_scores):
            result.reranker_score = score

        # Context scores (vectorized)
        query_type = self.query_processor.identify_query_type(query)
        for result in results:
            if query_type == 'factual' and result.element_type == 'text':
                result.context_score = 0.1
            elif query_type == 'procedural' and result.element_type == 'table':
                result.context_score = 0.15
            else:
                result.context_score = 0.0

        # Sort by combined_score once
        results.sort(key=lambda x: x.combined_score, reverse=True)

        # Diversity penalty (set-based, only for top_k)
        selected_results = []
        selected_texts = set()
        for result in results:
            penalty = 0.0
            result_words = set(result.text.lower().split())
            for selected in selected_results:
                selected_words = set(selected.text.lower().split())
                overlap = len(result_words & selected_words) / max(1, len(result_words | selected_words))
                if overlap > 0.7:
                    penalty = 0.4
                    break
            result.diversity_penalty = penalty
            result.final_score = (
                0.4 * result.combined_score +
                0.3 * result.reranker_score +
                0.2 * result.context_score -
                0.1 * result.diversity_penalty
            )
            selected_results.append(result)
            if len(selected_results) >= top_k:
                break

        # Final sort and return top_k
        selected_results.sort(key=lambda x: x.final_score, reverse=True)
        return selected_results[:top_k]

class Retriever:
    """Retriever with improved accuracy mechanisms"""

    def __init__(self,
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 chroma_db_path: str = "../data/chroma_db"):
        """Initialize the Enhanced Retriever"""
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.db_manager = ChromaDBManager(chroma_db_path)
        self.reranker = Reranker()
        self.query_processor = QueryProcessor()

        # Load all docs for BM25
        all_docs = self.db_manager.get_all_documents()
        self.bm25_docs = []
        self.bm25_metadatas = []
        if all_docs and all_docs.get('documents'):
            self.bm25_docs = all_docs['documents']
            self.bm25_metadatas = all_docs['metadatas']
        
        self.bm25_documents = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(self.bm25_docs, self.bm25_metadatas)
        ]
        self.bm25_retriever = BM25Retriever.from_documents(self.bm25_documents)

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
                page_number=max(1, metadata.get('page_number', 1)),
                element_type=metadata.get('element_type', 'text')
            )
            # Add all_pages to metadata and as attribute if present
            result.metadata['all_pages'] = metadata.get('all_pages', str(result.page_number))
            # Optionally, add as attribute (if you want direct access)
            result.all_pages = metadata.get('all_pages', str(result.page_number))
            formatted_results.append(result)
        return formatted_results

    def multi_vector_retrieve(self, query: str, n_results: int = 15) -> List[RetrievalResult]:
        """Multi-vector retrieval with query expansion"""
        all_results = []
        
        # Original query
        query_embedding = self.embedding_model.encode([query])
        results = self.db_manager.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        if results:
            all_results.extend(self._format_similarity_results(results))
        
        # Expanded queries
        expanded_queries = self.query_processor.expand_query(query)
        for expanded_query in expanded_queries[1:3]:  # Limit to avoid too many queries
            query_embedding = self.embedding_model.encode([expanded_query])
            results = self.db_manager.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results // 2
            )
            if results:
                all_results.extend(self._format_similarity_results(results))
        
        # Remove duplicates based on text content
        unique_results = {}
        for result in all_results:
            text_hash = hash(result.text)
            if text_hash not in unique_results or result.similarity_score > unique_results[text_hash].similarity_score:
                unique_results[text_hash] = result
        
        return list(unique_results.values())

    def adaptive_hybrid_retrieve(self, query: str, n_results: int = 10) -> List[RetrievalResult]:
        """Adaptive hybrid retrieval with query-aware weighting"""
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        # Determine query type for adaptive weighting
        print("Identifying query type...")
        query_type = self.query_processor.identify_query_type(query)
        
        # Adaptive weights based on query type
        if query_type == 'factual':
            similarity_weight, keyword_weight = 0.8, 0.2
        elif query_type == 'procedural':
            similarity_weight, keyword_weight = 0.6, 0.4
        elif query_type == 'interpretive':
            similarity_weight, keyword_weight = 0.7, 0.3
        else:
            similarity_weight, keyword_weight = 0.7, 0.3
        
        # Multi-vector similarity search
        print(f"Performing multi-vector retrieval for query")
        similarity_results = self.multi_vector_retrieve(query, n_results * 2)
        
        # Enhanced keyword search
        print(f"Performing enhanced keyword search for query")
        keyword_results = self.enhanced_keyword_search(query, n_results * 2)
        
        # Combine and score results
        print(f"Combining search results")
        combined_results = self._combine_search_results(similarity_results, keyword_results)
        self._calculate_combined_scores(combined_results, similarity_weight, keyword_weight)
        
        # Sort by combined score
        sorted_results = sorted(
            combined_results.values(), 
            key=lambda x: x.combined_score, 
            reverse=True
        )
        
        # Enhanced reranking
        print(f"Reranking results for query")
        reranked_results = self.reranker.rerank_results(query, sorted_results, top_k=n_results)
        
        return reranked_results

    def enhanced_keyword_search(self, query: str, n_results: int = 10) -> List[RetrievalResult]:
        """Enhanced keyword search with term boosting"""
        if not self.bm25_documents:
            return []
        key_terms = self.query_processor.extract_key_terms(query)
        boosted_query = query
        for term in key_terms:
            if term in self.query_processor.legal_terms:
                boosted_query += f" {term} {term}"
        
        results = self.bm25_retriever.get_relevant_documents(boosted_query)
        keyword_matches = []
        
        for doc in results[:n_results]:
            meta = doc.metadata
            keyword_score = getattr(doc, "score", 0)
            text_lower = doc.page_content.lower()
            matched_terms = [term for term in key_terms if term in text_lower]
            term_boost = len(matched_terms) / len(key_terms) if key_terms else 0
            
            result = RetrievalResult(
                text=doc.page_content,
                metadata=meta,
                keyword_score=keyword_score * (1 + term_boost),
                matched_keywords=matched_terms,
                page_number=max(1, meta.get('page_number', 1)),
                element_type=meta.get('element_type', 'text')
            )
            # Add all_pages to metadata and as attribute if present
            result.metadata['all_pages'] = meta.get('all_pages', str(result.page_number))
            result.all_pages = meta.get('all_pages', str(result.page_number))
            keyword_matches.append(result)
        
        return keyword_matches  

    def _combine_search_results(self, similarity_results: List[RetrievalResult], 
                               keyword_results: List[RetrievalResult]) -> Dict[str, RetrievalResult]:
        """Combine similarity and keyword search results"""
        combined_results = {}
        
        # Add similarity results
        for i, result in enumerate(similarity_results):
            doc_id = result.metadata.get('chunk_id', f"sim_{i}")
            combined_results[doc_id] = result
        
        # Add or merge keyword results
        for i, result in enumerate(keyword_results):
            doc_id = result.metadata.get('chunk_id', f"key_{i}")
            if doc_id in combined_results:
                # Merge scores
                combined_results[doc_id].keyword_score = result.keyword_score
                combined_results[doc_id].matched_keywords = result.matched_keywords
            else:
                # Add new result
                result.distance = 1.0
                result.similarity_score = 0.0
                combined_results[doc_id] = result
        
        return combined_results

    def _calculate_combined_scores(self, combined_results: Dict[str, RetrievalResult], 
                                 similarity_weight: float, keyword_weight: float) -> None:
        """Calculate combined scores for hybrid retrieval"""
        if not combined_results:
            return
        
        # Normalize scores
        similarity_scores = [r.similarity_score for r in combined_results.values()]
        keyword_scores = [r.keyword_score for r in combined_results.values()]
        
        max_similarity = max(similarity_scores) if similarity_scores else 1.0
        max_keyword = max(keyword_scores) if keyword_scores else 1.0
        
        for result in combined_results.values():
            norm_similarity = result.similarity_score / max_similarity if max_similarity > 0 else 0
            norm_keyword = result.keyword_score / max_keyword if max_keyword > 0 else 0
            
            result.combined_score = (
                similarity_weight * norm_similarity + 
                keyword_weight * norm_keyword
            )
