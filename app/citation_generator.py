import os
from typing import Dict

class CitationLinkGenerator:
    """Generate clickable links to PDF paragraphs with Docling metadata"""
    
    def __init__(self, pdf_path: str, base_url: str = ""):
        self.pdf_path = pdf_path
        self.base_url = base_url
        
    def generate_pdf_link(self, page_number: int, paragraph_id: str = None) -> str:
        """Generate a clickable link to a specific page in the PDF"""
        if self.base_url:
            pdf_url = f"{self.base_url}/{os.path.basename(self.pdf_path)}"
            link = f"{pdf_url}#page={page_number}"
        else:
            abs_path = os.path.abspath(self.pdf_path)
            link = f"file:///{abs_path}#page={page_number}"
        
        return link
    
    def generate_enhanced_citation(self, citation_info: Dict) -> str:
        """Generate enhanced citation with Docling metadata"""
        page_num = citation_info.get('page_number', 1)
        paragraph_id = citation_info.get('paragraph_id', '')
        citation_text = citation_info.get('citation_text', '')
        citation_number = citation_info.get('citation_number', '')
        section_title = citation_info.get('section_title', '')
        section_number = citation_info.get('section_number', '')
        content_type = citation_info.get('content_type', 'paragraph')
        
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
        
        formatted_citation += f"\n    🔗 {pdf_link}"
        formatted_citation += f"\n    📄 Page {page_num} | Type: {content_type} | ID: {paragraph_id}"
        
        return formatted_citation
    
    def generate_markdown_citation_enhanced(self, citation_info: Dict) -> str:
        """Generate enhanced Markdown citation with Docling metadata"""
        page_num = citation_info.get('page_number', 1)
        paragraph_id = citation_info.get('paragraph_id', '')
        citation_text = citation_info.get('citation_text', '')
        citation_number = citation_info.get('citation_number', '')
        section_title = citation_info.get('section_title', '')
        section_number = citation_info.get('section_number', '')
        content_type = citation_info.get('content_type', 'paragraph')
        
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
        formatted_citation += f"  \n🔗 [View in PDF (Page {page_num})]({pdf_link}) | Type: `{content_type}` | ID: `{paragraph_id}`"
        
        return formatted_citation