"""
Document Processor for creating and persisting constitutional database
Run this once to create your vectorstore database
"""
import os
os.environ["CHROMA_TELEMETRY"] = "FALSE"

from typing import Dict, Any, List
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.schema import Document
import os
import logging
import re
import pymupdf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document parsing and database creation - run once"""
    
    def __init__(self, persist_directory: str = "./constitution_vectorstore"):
        """
        Initialize DocumentProcessor
        
        Args:
            persist_directory: Directory to persist the Chroma database
        """
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
        )
        self.vectorstore = None
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
    
    def create_and_persist_database(self, text_input, input_type: str = "structure") -> Chroma:
        """
        Create hierarchical chunks and persist to local Chroma database
        
        Args:
            text_input: Either raw constitutional text or parsed structure
            input_type: "text" for raw text, "structure" for pre-parsed structure
            
        Returns:
            Chroma vectorstore instance
        """
        try:
            logger.info("Starting document processing...")
            
            # Parse text if needed, otherwise use provided structure
            if input_type == "text":
                logger.info("Parsing constitutional text...")
                structure = self.parse_constitutional_text(text_input)
            else:
                structure = text_input
            
            # Create hierarchical chunks
            documents = self.create_hierarchical_chunks(structure)
            logger.info(f"Created {len(documents)} hierarchical chunks")
            
            if not documents:
                raise ValueError("No documents were created from the structure")
            
            # Initialize Chroma vectorstore
            logger.info("Initializing Chroma vectorstore...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="pakistan_constitution"
            )
            
            # Add documents to vectorstore in batches
            batch_size = 100
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{total_batches} ({len(batch)} documents)")
                
                filtered_batch = filter_complex_metadata(batch)
                self.vectorstore.add_documents(filtered_batch)
            
            logger.info(f"Database successfully created and saved to {self.persist_directory}")
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"Error creating database: {str(e)}")
            raise
    
    def create_hierarchical_chunks(self, structure: Dict[str, Any]) -> List[Document]:
        """
        Create hierarchical chunks from parsed constitution structure
        
        Args:
            structure: Parsed constitution structure using regex patterns
            
        Returns:
            List of Document objects with metadata
        """
        documents = []
        
        try:
            # Process each part of the constitution
            for part_key, part_data in structure.items():
                if isinstance(part_data, dict):
                    # Create part-level document
                    part_content = f"{part_data.get('title', part_key)}"
                    documents.append(Document(
                        page_content=part_content,
                        metadata={
                            'chunk_type': 'part',
                            'part': part_key,
                            'part_title': part_data.get('title', ''),
                            'level': 'part'
                        }
                    ))
                    
                    # Process chapters within the part
                    chapters = part_data.get('chapters', {})
                    for chapter_key, chapter_data in chapters.items():
                        # Create chapter-level document
                        chapter_content = f"{chapter_data.get('title', chapter_key)}"
                        documents.append(Document(
                            page_content=chapter_content,
                            metadata={
                                'chunk_type': 'chapter',
                                'part': part_key,
                                'chapter': chapter_key,
                                'chapter_title': chapter_data.get('title', ''),
                                'level': 'chapter'
                            }
                        ))
                        
                        # Process articles within the chapter
                        articles = chapter_data.get('articles', {})
                        for article_key, article_data in articles.items():
                            article_content = self._build_article_content(article_data)
                            
                            documents.append(Document(
                                page_content=article_content,
                                metadata={
                                    'chunk_type': 'article',
                                    'part': part_key,
                                    'chapter': chapter_key,
                                    'article': article_key,
                                    'article_number': article_data.get('number', ''),
                                    'level': 'article'
                                }
                            ))
                    
        except Exception as e:
            logger.error(f"Error in hierarchical chunking: {str(e)}")
            raise
            
        return documents
    
    def _build_article_content(self, article_data: Dict[str, Any]) -> str:
        """Build complete article content from parsed data"""
        content_parts = []
        
        # Add article number and main content
        if 'number' in article_data:
            content_parts.append(f"Article {article_data['number']}")
        
        if 'content' in article_data:
            content_parts.append(article_data['content'])
        
        if 'full_text' in article_data:
            content_parts.append(article_data['full_text'])
        
        # Add any additional content
        if 'additional_content' in article_data:
            content_parts.extend(article_data['additional_content'])
        
        return "\n".join(content_parts)
    
    def _process_part(self, part_key: str, part_data: Dict[str, Any]) -> List[Document]:
        """Process a constitutional part (like Part I, Part II, etc.)"""
        documents = []
        
        # Create part-level document
        part_content = self._extract_part_content(part_data)
        if part_content:
            documents.append(Document(
                page_content=part_content,
                metadata={
                    'chunk_type': 'part',
                    'part': part_key,
                    'level': 'part'
                }
            ))
        
        # Process chapters within the part
        for chapter_key, chapter_data in part_data.items():
            if isinstance(chapter_data, dict):
                documents.extend(self._process_chapter(part_key, chapter_key, chapter_data))
        
        return documents
    
    def _process_chapter(self, part_key: str, chapter_key: str, chapter_data: Dict[str, Any]) -> List[Document]:
        """Process a constitutional chapter"""
        documents = []
        
        # Create chapter-level document
        chapter_content = self._extract_chapter_content(chapter_data)
        if chapter_content:
            documents.append(Document(
                page_content=chapter_content,
                metadata={
                    'chunk_type': 'chapter',
                    'part': part_key,
                    'chapter': chapter_key,
                    'level': 'chapter'
                }
            ))
        
        # Process articles within the chapter
        for article_key, article_data in chapter_data.items():
            if self._is_article(article_key):
                documents.extend(self._process_article(part_key, chapter_key, article_key, article_data))
        
        return documents
    
    def _process_article(self, part_key: str, chapter_key: str, article_key: str, article_data: Any) -> List[Document]:
        """Process a constitutional article"""
        documents = []
        
        article_content = self._extract_article_content(article_data)
        if article_content:
            # Extract article number if present
            article_number = self._extract_article_number(article_key)
            
            documents.append(Document(
                page_content=article_content,
                metadata={
                    'chunk_type': 'article',
                    'part': part_key,
                    'chapter': chapter_key,
                    'article': article_key,
                    'article_number': article_number,
                    'level': 'article'
                }
            ))
        
        return documents
    
    def _extract_part_content(self, part_data: Dict[str, Any]) -> str:
        """Extract content from part data"""
        # Implement based on your data structure
        content_parts = []
        
        # Add part title if available
        if 'title' in part_data:
            content_parts.append(f"PART: {part_data['title']}")
        
        # Add part description if available
        if 'description' in part_data:
            content_parts.append(part_data['description'])
            
        return "\n".join(content_parts)
    
    def _extract_chapter_content(self, chapter_data: Dict[str, Any]) -> str:
        """Extract content from chapter data"""
        content_parts = []
        
        # Add chapter title if available
        if 'title' in chapter_data:
            content_parts.append(f"CHAPTER: {chapter_data['title']}")
        
        # Add chapter description if available
        if 'description' in chapter_data:
            content_parts.append(chapter_data['description'])
            
        return "\n".join(content_parts)
    
    def _extract_article_content(self, article_data: Any) -> str:
        """Extract content from article data"""
        if isinstance(article_data, str):
            return article_data
        elif isinstance(article_data, dict):
            content_parts = []
            
            # Add article title if available
            if 'title' in article_data:
                content_parts.append(f"ARTICLE: {article_data['title']}")
            
            # Add article text
            if 'text' in article_data:
                content_parts.append(article_data['text'])
            elif 'content' in article_data:
                content_parts.append(article_data['content'])
            
            # Add subsections if present
            for key, value in article_data.items():
                if key.startswith('subsection') or key.startswith('clause'):
                    content_parts.append(f"{key}: {value}")
            
            return "\n".join(content_parts)
        else:
            return str(article_data)
    
    def parse_constitutional_text(self, text: str) -> Dict[str, Any]:
        """
        Parse constitutional text using proven regex patterns
        Handles nested structure: PART -> CHAPTER -> ARTICLE -> subsections -> sub-subsections
        
        Args:
            text: Raw constitutional text
            
        Returns:
            Structured dictionary of parts, chapters, and articles
        """
        # Enhanced regex patterns for parsing
        part_pattern = re.compile(r"^\s*PART\s+([IVXLCDM]+)(?:\s+(.+))?$", re.MULTILINE | re.IGNORECASE)
        chapter_pattern = re.compile(r"^\s*CHAPTER\s+(\d+)(?:\.?\s*—?\s*(.+))?$", re.MULTILINE | re.IGNORECASE)
        article_pattern = re.compile(r"^\s*(\d+)\.\s*(.+)$", re.MULTILINE)
        subsection_pattern = re.compile(r"^\s*\((\d+)\)\s*(.+)$", re.MULTILINE)
        sub_subsection_pattern = re.compile(r"^\s*\(([a-z])\)\s*(.+)$", re.MULTILINE)
        
        structure = {}
        current_part = None
        current_chapter = None
        current_article = None
        current_article_content = []
        
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check for PART
            part_match = part_pattern.match(line)
            if part_match:
                # Save previous article content
                self._save_article_content(structure, current_part, current_chapter, current_article, current_article_content)
                current_article_content = []
                
                part_num = part_match.group(1)
                part_title = part_match.group(2).strip() if part_match.group(2) else ""
                current_part = f"PART_{part_num}"
                current_chapter = None
                current_article = None
                
                structure[current_part] = {
                    'number': part_num,
                    'title': part_title,
                    'full_title': f"PART {part_num} {part_title}".strip(),
                    'chapters': {}
                }
                continue
            
            # Check for CHAPTER
            chapter_match = chapter_pattern.match(line)
            if chapter_match:
                # Save previous article content
                self._save_article_content(structure, current_part, current_chapter, current_article, current_article_content)
                current_article_content = []
                
                chapter_num = chapter_match.group(1)
                chapter_title = chapter_match.group(2).strip() if chapter_match.group(2) else ""
                current_chapter = f"CHAPTER_{chapter_num}"
                current_article = None
                
                if current_part and current_part in structure:
                    structure[current_part]['chapters'][current_chapter] = {
                        'number': chapter_num,
                        'title': chapter_title,
                        'full_title': f"CHAPTER {chapter_num}—{chapter_title}".strip(),
                        'articles': {}
                    }
                continue
            
            # Check for ARTICLE
            article_match = article_pattern.match(line)
            if article_match:
                # Save previous article content
                self._save_article_content(structure, current_part, current_chapter, current_article, current_article_content)
                current_article_content = []
                
                article_num = article_match.group(1)
                article_title = article_match.group(2).strip()
                current_article = f"ARTICLE_{article_num}"
                
                if current_part and current_chapter and current_part in structure:
                    if current_chapter in structure[current_part]['chapters']:
                        structure[current_part]['chapters'][current_chapter]['articles'][current_article] = {
                            'number': article_num,
                            'title': article_title,
                            'full_text': line,
                            'subsections': [],
                            'content': []
                        }
                        # Add the main article line to content
                        current_article_content.append(line)
                continue
            
            # If we're inside an article, collect all content (subsections, sub-subsections, etc.)
            if current_article and line:
                current_article_content.append(line)
        
        # Save any remaining article content
        self._save_article_content(structure, current_part, current_chapter, current_article, current_article_content)
        
        return structure
    
    def _save_article_content(self, structure: Dict, part: str, chapter: str, article: str, content: List[str]):
        """Save accumulated content to the current article"""
        if (part and chapter and article and 
            part in structure and 
            chapter in structure[part]['chapters'] and 
            article in structure[part]['chapters'][chapter]['articles']):
            
            # Join all content for the article
            full_content = '\n'.join(content)
            structure[part]['chapters'][chapter]['articles'][article]['content'] = content
            structure[part]['chapters'][chapter]['articles'][article]['full_content'] = full_content
    
    def _build_article_content(self, article_data: Dict[str, Any]) -> str:
        """Build complete article content from parsed data"""
        content_parts = []
        
        # Add article header (number and title)
        if 'number' in article_data and 'title' in article_data:
            content_parts.append(f"{article_data['number']}. {article_data['title']}")
        
        # Add all content (subsections, etc.)
        if 'full_content' in article_data:
            content_parts.append(article_data['full_content'])
        elif 'content' in article_data and isinstance(article_data['content'], list):
            content_parts.extend(article_data['content'])
        elif 'content' in article_data:
            content_parts.append(str(article_data['content']))
        
        return "\n".join(content_parts)
    
    def create_hierarchical_chunks(self, structure: Dict[str, Any]) -> List[Document]:
        """
        Create hierarchical chunks from parsed constitution structure
        Optimized for Pakistani constitution format with detailed articles
        
        Args:
            structure: Parsed constitution structure using regex patterns
            
        Returns:
            List of Document objects with metadata
        """
        documents = []
        
        try:
            # Process each part of the constitution
            for part_key, part_data in structure.items():
                if not isinstance(part_data, dict):
                    continue
                
                # Create part-level document
                part_content = part_data.get('full_title', part_data.get('title', part_key))
                documents.append(Document(
                    page_content=part_content,
                    metadata={
                        'chunk_type': 'part',
                        'part': part_key,
                        'part_number': part_data.get('number', ''),
                        'part_title': part_data.get('title', ''),
                        'level': 'part'
                    }
                ))
                
                # Process chapters within the part
                chapters = part_data.get('chapters', {})
                for chapter_key, chapter_data in chapters.items():
                    # Create chapter-level document
                    chapter_content = chapter_data.get('full_title', chapter_data.get('title', chapter_key))
                    documents.append(Document(
                        page_content=chapter_content,
                        metadata={
                            'chunk_type': 'chapter',
                            'part': part_key,
                            'chapter': chapter_key,
                            'chapter_number': chapter_data.get('number', ''),
                            'chapter_title': chapter_data.get('title', ''),
                            'level': 'chapter'
                        }
                    ))
                    
                    # Process articles within the chapter
                    articles = chapter_data.get('articles', {})
                    for article_key, article_data in articles.items():
                        article_content = self._build_article_content(article_data)
                        
                        # Create comprehensive metadata for articles
                        metadata = {
                            'chunk_type': 'article',
                            'part': part_key,
                            'chapter': chapter_key,
                            'article': article_key,
                            'article_number': article_data.get('number', ''),
                            'article_title': article_data.get('title', ''),
                            'level': 'article'
                        }
                        
                        # Add searchable keywords based on content
                        keywords = self._extract_keywords(article_content)
                        if keywords:
                            metadata['keywords'] = keywords
                        
                        documents.append(Document(
                            page_content=article_content,
                            metadata=metadata
                        ))
            
            logger.info(f"Created {len(documents)} documents: "
                       f"{len([d for d in documents if d.metadata['chunk_type'] == 'part'])} parts, "
                       f"{len([d for d in documents if d.metadata['chunk_type'] == 'chapter'])} chapters, "
                       f"{len([d for d in documents if d.metadata['chunk_type'] == 'article'])} articles")
                    
        except Exception as e:
            logger.error(f"Error in hierarchical chunking: {str(e)}")
            raise
            
        return documents
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract relevant keywords from article content for better searchability"""
        keywords = []
        content_lower = content.lower()
        
        # Constitutional concepts
        constitutional_terms = [
            'fundamental rights', 'equality', 'freedom', 'liberty', 'justice',
            'president', 'prime minister', 'parliament', 'assembly', 'senate',
            'judiciary', 'supreme court', 'high court', 'election', 'voting',
            'citizenship', 'provincial', 'federal', 'emergency', 'amendment'
        ]
        
        for term in constitutional_terms:
            if term in content_lower:
                keywords.append(term)
        
        return keywords
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the created database"""
        if not self.vectorstore:
            return {"status": "not_created"}
        
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "status": "created",
                "document_count": count,
                "persist_directory": self.persist_directory,
                "embedding_model": "nomic-embed-text"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor(persist_directory="./pakistan_constitution_db")
    
    # Extract text from PDF
    pdf_path = "data/Constitution of pakistan 1973.pdf"
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Parse the text into structure
    structure = processor.parse_constitutional_text(pdf_text)
    
    # Create and persist the database
    vectorstore = processor.create_and_persist_database(structure, input_type="structure")
    
    # Get statistics
    stats = processor.get_database_stats()
    print(f"Database stats: {stats}")
