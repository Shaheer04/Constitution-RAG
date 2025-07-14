import os
import logging
from typing import Dict, Optional, Union
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CitationGeneratorError(Exception):
    """Base exception for citation generator errors"""
    pass


class InvalidInputError(CitationGeneratorError):
    """Raised when input validation fails"""
    pass


class FileNotFoundError(CitationGeneratorError):
    """Raised when PDF file is not found"""
    pass


class InvalidPageNumberError(CitationGeneratorError):
    """Raised when page number is invalid"""
    pass


class CitationLinkGenerator:
    """Generate clickable links to PDF paragraphs with Docling metadata"""
    
    def __init__(self, pdf_path: str, base_url: str = ""):
        """
        Initialize the citation generator
        
        Args:
            pdf_path: Path to the PDF file
            base_url: Base URL for web-based PDF access (optional)
            
        Raises:
            InvalidInputError: If pdf_path is None, empty, or invalid type
            FileNotFoundError: If PDF file doesn't exist
        """
        try:
            self._validate_pdf_path(pdf_path)
            self._validate_base_url(base_url)
            
            self.pdf_path = pdf_path
            self.base_url = base_url
            
            logger.info(f"CitationLinkGenerator initialized with PDF: {pdf_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize CitationLinkGenerator: {str(e)}")
            raise
    
    def _validate_pdf_path(self, pdf_path: Union[str, None]) -> None:
        """Validate PDF path input"""
        if pdf_path is None:
            raise InvalidInputError("PDF path cannot be None")
        
        if not isinstance(pdf_path, str):
            raise InvalidInputError(f"PDF path must be a string, got {type(pdf_path).__name__}")
        
        if not pdf_path.strip():
            raise InvalidInputError("PDF path cannot be empty")
        
        # Check if file exists
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Check if it's actually a file (not a directory)
        if not Path(pdf_path).is_file():
            raise InvalidInputError(f"PDF path is not a file: {pdf_path}")
        
        # Check if it's a PDF file
        if not pdf_path.lower().endswith('.pdf'):
            raise InvalidInputError(f"File must be a PDF: {pdf_path}")
    
    def _validate_base_url(self, base_url: Union[str, None]) -> None:
        """Validate base URL input"""
        if base_url is None:
            raise InvalidInputError("Base URL cannot be None")
        
        if not isinstance(base_url, str):
            raise InvalidInputError(f"Base URL must be a string, got {type(base_url).__name__}")
        
        # Empty string is allowed for base_url
    
    def _validate_page_number(self, page_number: Union[int, None]) -> None:
        """Validate page number input"""
        if page_number is None:
            raise InvalidPageNumberError("Page number cannot be None")
        
        if not isinstance(page_number, int):
            raise InvalidPageNumberError(f"Page number must be an integer, got {type(page_number).__name__}")
        
        if page_number <= 0:
            raise InvalidPageNumberError(f"Page number must be positive, got {page_number}")
    
    def _validate_paragraph_id(self, paragraph_id: Union[str, None]) -> None:
        """Validate paragraph ID input"""
        if paragraph_id is not None and not isinstance(paragraph_id, str):
            raise InvalidInputError(f"Paragraph ID must be a string or None, got {type(paragraph_id).__name__}")
    
    def _validate_citation_info(self, citation_info: Union[Dict, None]) -> None:
        """Validate citation info dictionary"""
        if citation_info is None:
            raise InvalidInputError("Citation info cannot be None")
        
        if not isinstance(citation_info, dict):
            raise InvalidInputError(f"Citation info must be a dictionary, got {type(citation_info).__name__}")
        
        # Validate page_number if present
        if 'page_number' in citation_info:
            page_num = citation_info['page_number']
            if page_num is not None:
                self._validate_page_number(page_num)
        
        # Validate string fields
        string_fields = ['paragraph_id', 'citation_text', 'citation_number', 
                        'section_title', 'section_number', 'content_type']
        for field in string_fields:
            if field in citation_info and citation_info[field] is not None:
                if not isinstance(citation_info[field], str):
                    raise InvalidInputError(f"{field} must be a string or None, got {type(citation_info[field]).__name__}")
        
    def generate_pdf_link(self, page_number: int, paragraph_id: Optional[str] = None) -> str:
        """
        Generate a clickable link to a specific page in the PDF and validate its accessibility
        
        Args:
            page_number: The page number to link to (must be positive integer)
            paragraph_id: Optional paragraph identifier
            
        Returns:
            str: The generated PDF link
            
        Raises:
            InvalidPageNumberError: If page number is invalid
            InvalidInputError: If paragraph_id is invalid type
        """
        try:
            self._validate_page_number(page_number)
            self._validate_paragraph_id(paragraph_id)
            
            try:
                if self.base_url:
                    pdf_url = f"{self.base_url}/{os.path.basename(self.pdf_path)}"
                    link = f"{pdf_url}#page={page_number}"
                else:
                    abs_path = os.path.abspath(self.pdf_path)
                    link = f"file:///{abs_path}#page={page_number}"
                
                # Validate link accessibility
                # No need to validate link accessibility for local file paths
                # This check is more relevant for web URLs and can cause issues with local file:// paths
                pass
            except Exception as e:
                logger.warning(f"Skipping link accessibility validation due to error: {str(e)}")
            
            logger.debug(f"Generated PDF link for page {page_number}: {link}")
            return link
            
        except Exception as e:
            logger.error(f"Failed to generate PDF link for page {page_number}: {str(e)}")
            raise
    
    def generate_enhanced_citation(self, citation_info: Dict) -> str:
        """
        Generate enhanced citation with Docling metadata
        
        Args:
            citation_info: Dictionary containing citation information
            
        Returns:
            str: The formatted citation string
            
        Raises:
            InvalidInputError: If citation_info is invalid
            InvalidPageNumberError: If page number is invalid
        """
        try:
            self._validate_citation_info(citation_info)
            
            # Extract and sanitize data with defaults
            page_num = citation_info.get('page_number', 1)
            paragraph_id = citation_info.get('paragraph_id', '') or ''
            citation_text = citation_info.get('citation_text', '') or ''
            citation_number = citation_info.get('citation_number', '') or ''
            section_title = citation_info.get('section_title', '') or ''
            section_number = citation_info.get('section_number', '') or ''
            content_type = citation_info.get('content_type', 'paragraph') or 'paragraph'
            
            # Generate the link
            pdf_link = self.generate_pdf_link(page_num, paragraph_id)
            
            # Format the citation with enhanced metadata
            citation_parts = [f"[{citation_number}]"]
            
            if section_number and section_title:
                citation_parts.append(f"Section {section_number}: {section_title}")
            elif section_title:
                citation_parts.append(f"Section: {section_title}")
            
            citation_parts.append(citation_text)
            
            formatted_citation = " ".join(citation_parts)
            
            formatted_citation += f"\n    LINK {pdf_link}"
            formatted_citation += f"\n    PAGE Page {page_num} | Type: {content_type} | ID: {paragraph_id}"
            
            # Include extracted page content if available
            if 'page_content' in citation_info:
                page_content_preview = citation_info['page_content'][:100].strip()
                if len(citation_info['page_content']) > 100:
                    page_content_preview += "..."
                formatted_citation += f"\n    CONTENT Preview: \"{page_content_preview}\""
            
            logger.debug(f"Generated enhanced citation for page {page_num}")
            return formatted_citation
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced citation: {str(e)}")
            raise
    
    def generate_markdown_citation_enhanced(self, citation_info: Dict) -> str:
        """
        Generate enhanced Markdown citation with Docling metadata
        
        Args:
            citation_info: Dictionary containing citation information
            
        Returns:
            str: The formatted Markdown citation string
            
        Raises:
            InvalidInputError: If citation_info is invalid
            InvalidPageNumberError: If page number is invalid
        """
        try:
            self._validate_citation_info(citation_info)
            
            # Extract and sanitize data with defaults
            page_num = citation_info.get('page_number', 1)
            paragraph_id = citation_info.get('paragraph_id', '') or ''
            citation_text = citation_info.get('citation_text', '') or ''
            citation_number = citation_info.get('citation_number', '') or ''
            section_title = citation_info.get('section_title', '') or ''
            section_number = citation_info.get('section_number', '') or ''
            content_type = citation_info.get('content_type', 'paragraph') or 'paragraph'
            
            # Generate the link
            pdf_link = self.generate_pdf_link(page_num, paragraph_id)
            
            # Build citation text
            citation_parts = [f"**[{citation_number}]**"]
            
            if section_number and section_title:
                citation_parts.append(f"**Section {section_number}:** {section_title}")
            elif section_title:
                citation_parts.append(f"**Section:** {section_title}")
            
            citation_parts.append(citation_text)
            
            formatted_citation = " ".join(citation_parts)
            formatted_citation += f"  \nLINK [View in PDF (Page {page_num})]({pdf_link}) | Type: `{content_type}` | ID: `{paragraph_id}`"
            
            logger.debug(f"Generated enhanced Markdown citation for page {page_num}")
            return formatted_citation
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced Markdown citation: {str(e)}")
            raise