import re
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.retrievers import ParentDocumentRetriever
import pymupdf
import json

class Chunker:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vectorstore = Chroma(
            collection_name="constitution_chunks_collection",
            embedding_function=self.embeddings,
            persist_directory="./chroma_langchain_db"
        )
        
    def parse_constitution_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Parse PDF and extract hierarchical structure"""
        doc = pymupdf.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        
        # Parse structure using regex patterns
        structure = self._parse_hierarchical_structure(text)
        return structure
    
    def _parse_hierarchical_structure(self, text: str) -> Dict[str, Any]:
        """Extract Parts, Chapters, and Articles from constitution text"""
        
        # Matches "PART" followed by a Roman numeral (e.g., "PART I", "PART II. SOME TITLE")
        part_pattern = re.compile(r"^\s*PART\s+([IVXLCDM]+)(?:\.\s*(.+))?$", re.MULTILINE | re.IGNORECASE)

        # Matches "CHAPTER" followed by a number (e.g., "CHAPTER 1", "Chapter 2.--LOCAL GOVERNMENTS")
        chapter_pattern = re.compile(r"^\s*CHAPTER\s+(\d+)(?:\.?\s*(.+))?$", re.MULTILINE | re.IGNORECASE)

        # Captures: 1: Article number, 2: Content/title following the number and period
        article_pattern = re.compile(r"^\s*(\d+)\.\s*(.+)$", re.MULTILINE)

        parts = []
        current_part = None
        current_chapter = None
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            # Check for Part
            part_match = re.match(part_pattern, line.strip())
            if part_match:
                if current_part:
                    parts.append(current_part)
                
                current_part = {
                    'number': part_match.group(1),
                    'title': part_match.group(2),
                    'full_text': '',
                    'chapters': [],
                    'start_line': i
                }
                current_chapter = None
                continue
            
            # Check for Chapter
            chapter_match = re.match(chapter_pattern, line.strip())
            if chapter_match and current_part:
                if current_chapter:
                    current_part['chapters'].append(current_chapter)
                
                current_chapter = {
                    'number': chapter_match.group(1),
                    'title': chapter_match.group(2).strip(),
                    'full_text': '',
                    'articles': [],
                    'start_line': i
                }
                continue
            
            # Check for Article
            article_match = re.match(article_pattern, line.strip())
            if article_match and current_chapter:
                article = {
                    'number': article_match.group(1),
                    'title': article_match.group(2).strip(),
                    'full_text': '',
                    'clauses': [],
                    'start_line': i
                }
                current_chapter['articles'].append(article)
                continue
            
            # Add text to current containers
            if current_part:
                current_part['full_text'] += line + '\n'
                if current_chapter:
                    current_chapter['full_text'] += line + '\n'
                    if current_chapter['articles']:
                        current_chapter['articles'][-1]['full_text'] += line + '\n'
        
        # Add last part
        if current_part:
            if current_chapter:
                current_part['chapters'].append(current_chapter)
            parts.append(current_part)
        
        return {'parts': parts}
    
    def create_hierarchical_chunks(self, structure: Dict[str, Any]) -> List[Document]:
        """Create hierarchical chunks with metadata"""
        documents = []
        
        for part in structure['parts']:
            # Create part-level chunks
            part_doc = Document(
                page_content=f"PART {part['number']}: {part['title']}\n\n{part['full_text'][:2000]}...",
                metadata={
                    'chunk_type': 'part',
                    'part_number': part['number'],
                    'part_title': part['title'],
                    'chunk_id': f"part_{part['number']}",
                    'level': 0
                }
            )
            documents.append(part_doc)
            
            for chapter in part['chapters']:
                # Create chapter-level chunks
                chapter_context = f"PART {part['number']}: {part['title']}\n\n"
                chapter_doc = Document(
                    page_content=chapter_context + f"Chapter {chapter['number']}: {chapter['title']}\n\n{chapter['full_text']}",
                    metadata={
                        'chunk_type': 'chapter',
                        'part_number': part['number'],
                        'part_title': part['title'],
                        'chapter_number': chapter['number'],
                        'chapter_title': chapter['title'],
                        'chunk_id': f"part_{part['number']}_chapter_{chapter['number']}",
                        'parent_id': f"part_{part['number']}",
                        'level': 1
                    }
                )
                documents.append(chapter_doc)
                
                for article in chapter['articles']:
                    # Create article-level chunks (most granular)
                    article_context = f"PART {part['number']}: {part['title']} > Chapter {chapter['number']}: {chapter['title']}\n\n"
                    article_doc = Document(
                        page_content=article_context + f"Article {article['number']}: {article['title']}\n\n{article['full_text']}",
                        metadata={
                            'chunk_type': 'article',
                            'part_number': part['number'],
                            'part_title': part['title'],
                            'chapter_number': chapter['number'],
                            'chapter_title': chapter['title'],
                            'article_number': article['number'],
                            'article_title': article['title'],
                            'chunk_id': f"part_{part['number']}_chapter_{chapter['number']}_article_{article['number']}",
                            'parent_id': f"part_{part['number']}_chapter_{chapter['number']}",
                            'level': 2
                        }
                    )
                    documents.append(article_doc)
        
        return documents
    
    def setup_retriever(self) -> ParentDocumentRetriever:
        """Setup parent document retriever for hierarchical search"""
        
        # Child splitter for fine-grained chunks
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        
        # Parent splitter for broader context
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        
        retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        
        return retriever
    
    def smart_retrieve(self, query: str, structure: Dict[str, Any]) -> List[Document]:
        """Implement intelligent retrieval based on query type"""
        documents = self.create_hierarchical_chunks(structure)
        
        # Add documents to vectorstore
        self.vectorstore.add_documents(documents)
        
        # Determine query type and retrieve accordingly
        if self._is_specific_article_query(query):
            # Search at article level
            results = self.vectorstore.similarity_search(
                query, 
                k=5,
                filter={'chunk_type': 'article'}
            )
        elif self._is_topic_based_query(query):
            # Search at chapter level first, then expand
            results = self.vectorstore.similarity_search(
                query, 
                k=3,
                filter={'chunk_type': 'chapter'}
            )
            # Add related articles
            article_results = self.vectorstore.similarity_search(
                query, 
                k=5,
                filter={'chunk_type': 'article'}
            )
            results.extend(article_results)
        else:
            # Multi-level search
            results = self.vectorstore.similarity_search(query, k=8)
        
        return results
    
    def _is_specific_article_query(self, query: str) -> bool:
        """Check if query asks for specific article"""
        article_patterns = [r'article\s+\d+', r'section\s+\d+', r'clause\s+\d+']
        return any(re.search(pattern, query.lower()) for pattern in article_patterns)
    
    def _is_topic_based_query(self, query: str) -> bool:
        """Check if query is topic-based"""
        topic_keywords = ['fundamental rights', 'parliament', 'judiciary', 'federal', 'provincial']
        return any(keyword in query.lower() for keyword in topic_keywords)
    

    def view_sample_chunks(self, documents: List[Document], sample_size: int = 5):
        """View sample chunks from each hierarchy level"""
        
        # Group by type
        grouped = {}
        for doc in documents:
            chunk_type = doc.metadata.get('chunk_type', 'unknown')
            if chunk_type not in grouped:
                grouped[chunk_type] = []
            grouped[chunk_type].append(doc)
        
        for chunk_type, docs in grouped.items():
            print(f"\n{'='*60}")
            print(f"SAMPLE {chunk_type.upper()} CHUNKS")
            print(f"{'='*60}")
            
            # Show random sample
            import random
            sample_docs = random.sample(docs, min(sample_size, len(docs)))
            
            for i, doc in enumerate(sample_docs):
                print(f"\n--- SAMPLE {i+1} ---")
                print(f"ID: {doc.metadata.get('chunk_id')}")
                print(f"Content Length: {len(doc.page_content)}")
                print(f"Content Preview:")
                print(doc.page_content[:500] + "..." if len(doc.page_content) > 300 else doc.page_content)
                print(f"Metadata: {json.dumps(doc.metadata, indent=2)}")
                
                if i < len(sample_docs) - 1:
                    input("Press Enter for next sample...")

# Usage Example
def main():
    chunker = Chunker()
    
    # Parse constitution PDF
    structure = chunker.parse_constitution_pdf("data/Constitution of pakistan 1973.pdf")

    # Create hierarchical chunks
    documents = chunker.create_hierarchical_chunks(structure)
    
    print(f"Created {len(documents)} hierarchical chunks")
    
    # Setup retriever
    #retriever = chunker.setup_retriever()

    #chunker.view_sample_chunks(documents, sample_size=5)

if __name__ == "__main__":
    main()