"""
Retriever Component with Reranker and Hybrid Retrieval for Constitution RAG System
"""

from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
import logging
from dataclasses import dataclass
from enum import Enum
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langchain.retrievers import BM25Retriever
from langchain.schema import Document

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

class LegalBERTReranker:
    def __init__(self, model_name: str = "nlpaueb/legal-bert-base-uncased"):
        """Initialize Legal BERT reranker"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def rerank_results(self, query: str, results: List[RetrievalResult], 
                      top_k: int = 10) -> List[RetrievalResult]:
        """Rerank retrieval results using Legal BERT model"""
        if not results:
            return []
        reranked_results = []
        for result in results:
            inputs = self.tokenizer(
                query, 
                result.text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                relevance_score = torch.softmax(outputs.logits, dim=-1)[0, 1].item()
            result.reranker_score = relevance_score
            result.combined_score = (
                0.6 * result.combined_score + 
                0.4 * relevance_score
            )
            reranked_results.append(result)
        reranked_results.sort(key=lambda x: x.combined_score, reverse=True)
        return reranked_results[:top_k]

class Retriever:
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 chroma_db_path: str = "./constitution_db"):
        """Initialize the Retriever"""
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.db_manager = ChromaDBManager(chroma_db_path)
        self.reranker = LegalBERTReranker()

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
                hierarchical_path=metadata.get('hierarchical_path', 'root'),
                level=metadata.get('level', 0),
                heading=metadata.get('heading', ''),
                page_number=max(1, metadata.get('page_number', 1)),
                element_type=metadata.get('element_type', 'text')
            )
            formatted_results.append(result)
        return formatted_results

    def retrieve_relevant_chunks(self, query: str, n_results: int = 10, 
                               filter_by_level: Optional[int] = None, 
                               filter_by_type: Optional[str] = None) -> List[RetrievalResult]:
        """Retrieve relevant chunks based on query with hierarchical filtering"""
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        query_embedding = self.embedding_model.encode([query])
        where_clause = self._build_where_clause(filter_by_level, filter_by_type)
        results = self.db_manager.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where_clause=where_clause if where_clause else None
        )
        if not results:
            return []
        return self._format_similarity_results(results)

    def keyword_search(self, query: str, n_results: int = 10) -> List[RetrievalResult]:
        """Perform optimized BM25 keyword-based search using LangChain."""
        if not self.bm25_documents:
            return []
        results = self.bm25_retriever.get_relevant_documents(query)
        keyword_matches = []
        for doc in results[:n_results]:
            meta = doc.metadata
            result = RetrievalResult(
                text=doc.page_content,
                metadata=meta,
                keyword_score=getattr(doc, "score", 0),
                matched_keywords=[],
                hierarchical_path=meta.get('hierarchical_path', 'root'),
                level=meta.get('level', 0),
                heading=meta.get('heading', ''),
                page_number=max(1, meta.get('page_number', 1)),
                element_type=meta.get('element_type', 'text')
            )
            keyword_matches.append(result)
        return keyword_matches

    def _combine_search_results(self, similarity_results: List[RetrievalResult], 
                               keyword_results: List[RetrievalResult]) -> Dict[str, RetrievalResult]:
        """Combine similarity and keyword search results"""
        combined_results = {}
        for i, result in enumerate(similarity_results):
            doc_id = result.metadata.get('chunk_id', str(i))
            combined_results[doc_id] = result
        for result in keyword_results:
            doc_id = result.metadata.get('chunk_id', 'unknown')
            if doc_id in combined_results:
                combined_results[doc_id].keyword_score = result.keyword_score
                combined_results[doc_id].matched_keywords = result.matched_keywords
            else:
                result.distance = 1.0
                combined_results[doc_id] = result
        return combined_results

    def _calculate_combined_scores(self, combined_results: Dict[str, RetrievalResult], 
                                 similarity_weight: float, keyword_weight: float) -> None:
        """Calculate combined scores for hybrid retrieval"""
        if not combined_results:
            return
        max_similarity = max(r.similarity_score for r in combined_results.values())
        max_keyword = max(r.keyword_score for r in combined_results.values())
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
        """Hybrid retrieval combining similarity search, keyword matching, and reranking"""
        if not query.strip():
            logger.warning("Empty query provided")
            return []
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
        # 2. Keyword search (BM25)
        keyword_results = self.keyword_search(query, n_results * 2)
        # 3. Combine results
        combined_results = self._combine_search_results(similarity_results, keyword_results)
        # 4. Calculate combined scores
        self._calculate_combined_scores(combined_results, similarity_weight, keyword_weight)
        # 5. Sort by combined score
        sorted_results = sorted(
            combined_results.values(), 
            key=lambda x: x.combined_score, 
            reverse=True
        )
        # 6. Rerank with LegalBERTReranker (if enough results)
        reranked_results = self.reranker.rerank_results(query, sorted_results, top_k=n_results)
        return reranked_results