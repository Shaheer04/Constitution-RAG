"""
FrontEnd for RAG System using Streamlit
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import os
import logging
from datetime import datetime
from typing import List, Dict, Any
import uuid
import re

# LangChain imports for conversation memory
from langchain.memory import ConversationBufferWindowMemory

# Import your existing RAG components
from backend.retriever import Retriever
from backend.generator import Generator
from backend.query_rewriter import QueryRewriter
from backend.config import settings

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Constitution RAG Chatbot",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for references
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}

.stChatMessage {
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}

.pdf-reference {
    display: inline-block;
    background-color: #e3f2fd;
    color: #1976d2;
    padding: 6px 12px;
    margin: 4px;
    border-radius: 5px;
    text-decoration: none;
    border: 1px solid #1976d2;
    font-size: 0.9em;
    font-weight: 500;
}

.pdf-reference:hover {
    background-color: #bbdefb;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.references-section {
    margin-top: 15px;
    padding: 12px;
    background-color: #f8f9fa;
    border-radius: 8px;
    border-left: 4px solid #28a745;
}

.references-title {
    font-weight: bold;
    margin-bottom: 8px;
    color: #2c3e50;
}

.metric-card {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
}

.status-healthy {
    color: #4caf50;
}

.status-unhealthy {
    color: #f44336;
}

.chat-stats {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
h1 {
    color: #fff; /* or another color that stands out on dark background */
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 1rem;
}

</style>
""", unsafe_allow_html=True)

class ConstitutionRAGApp:
    def __init__(self):
        self.initialize_session_state()
        self.setup_rag_system()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        if 'conversation_memory' not in st.session_state:
            st.session_state.conversation_memory = ConversationBufferWindowMemory(
                k=10,
                return_messages=True
            )
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
        
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
        
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = {}
        
        if 'current_sources' not in st.session_state:
            st.session_state.current_sources = []
    

    @st.cache_resource
    def load_rag_system(_self):
        """Load RAG system (cached to avoid reloading)"""
        try:
            logger.info("Loading RAG system...")

            pdf_path = str(settings.raw_pdf_dir / "constitution-1973.pdf")
            db_path = str(settings.chroma_dir)
            embedding_model = settings.embed_model
            ollama_model = settings.ollama_model 
            ollama_url = os.getenv("OLLAMA_URL")
            
            print("OLLAMA_URL being used:", ollama_url)

            # Initialize components
            retriever = Retriever(
                embedding_model_name=embedding_model,
                chroma_db_path=db_path
            )

            generator = Generator(
                ollama_model=ollama_model,
                ollama_url=ollama_url,
            )

            query_rewriter = QueryRewriter(ollama_url, ollama_model)

            return {
                'retriever': retriever,
                'generator': generator,
                'pdf_path': pdf_path,
                'db_path': db_path,
                'query_rewriter': query_rewriter
            }

        except Exception as e:
            logger.error(f"Error loading RAG system: {e}")
            st.error(f"Failed to load RAG system: {e}")
            return None
    
    def setup_rag_system(self):
        """Setup the RAG system"""
        if not st.session_state.system_initialized:
            with st.spinner("🔧 Initializing RAG system..."):
                rag_components = self.load_rag_system()
                if rag_components:
                    st.session_state.rag_system = rag_components
                    st.session_state.system_initialized = True
                else:
                    st.error("Failed to initialize RAG system")
                    st.stop()
    
    def needs_database_setup(self) -> bool:
        """Check if database needs to be set up"""
        try:
            rag_system = st.session_state.rag_system
            if not rag_system:
                return True
            
            # Try to get documents from database
            retriever = rag_system['retriever']
            if not retriever.db_manager.collection:
                return True
            
            all_docs = retriever.db_manager.get_all_documents()
            return not (all_docs and all_docs.get('documents') and len(all_docs['documents']) > 0)
            
        except Exception as e:
            logger.error(f"Error checking database: {e}")
            return True

    def _make_clickable(self, refs):
        """refs: List[RetrievalResult] or similar objects"""

        if not refs:
            return ""

        # Use settings for PDF path and file name
        pdf_path = str(settings.raw_pdf_dir / "constitution-1973.pdf")
        pdf_file = os.path.basename(pdf_path)
        base_url = "http://localhost:8000"

        # Check if PDF exists
        if not os.path.exists(pdf_path):
            return "<br><b>References:</b> PDF file not found."

        # Build unique pages once, keep order
        unique_pages = sorted({
            int(p)
            for r in refs
            for p in str(r.all_pages or r.page_number).split(",")
        })

        # Superscript footnote numbers inside the answer
        footnotes = []
        for idx, page in enumerate(unique_pages, 1):
            link = f'<a href="{base_url}/{pdf_file}#page={page}" target="_blank">{idx}</a>'
            footnotes.append(link)
        # Return the footer line
        return "<br><b>References:</b> " + ", ".join(footnotes)

    def ask_question(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        """Ask question to RAG system"""
        try:
            rag_system = st.session_state.rag_system
            if not rag_system:
                return {"error": "RAG system not initialized"}
            
            retriever = rag_system['retriever']
            generator = rag_system['generator']
            query_rewriter = rag_system['query_rewriter']
            
            # Query rewriting step (raises error if fails)
            print(f"Rewriting question")
            rewritten_question = query_rewriter.rewrite(question)
            
            # Retrieve relevant documents
            print(f"Retrieving documents for question: {rewritten_question}")
            reranked_results = retriever.adaptive_hybrid_retrieve(
                query=rewritten_question,
                n_results=n_results
            )
            if not reranked_results:
                return {
                    "answer": "No relevant information found in the constitution database.",
                    "sources": []
                }
            
            # Generate response
            print(f"Generating response for question")
            answer = generator.generate_response(rewritten_question, reranked_results)
            print("making clickable line")
            clickable_line = self._make_clickable(reranked_results)
            answer_html = answer + clickable_line

            return {
                "answer": answer_html,
                "sources": reranked_results
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {"error": str(e)}
    
    def display_conversation_controls(self):
        """Display minimal conversation controls"""
        st.sidebar.markdown("## 💬 Chat Controls")
        
        # Main controls in a single row
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🔄 New Chat", help="Start new conversation"):
                self.start_new_conversation()
        
        with col2:
            if st.button("🗑️ Clear", help="Clear current chat"):
                self.clear_conversation()

    def display_settings(self):
        """Display essential settings only"""
        st.sidebar.markdown("## ⚙️ Settings")
        
        # Retrieval settings
        n_results = st.sidebar.slider("📊 Source documents", 1, 10, 5, help="Number of sources to retrieve")
        st.session_state.n_results = n_results
        
        # PDF info
        rag_system = st.session_state.rag_system
        if rag_system:
            pdf_path = rag_system['pdf_path']
            if os.path.exists(pdf_path):
                st.sidebar.success(f"📄 PDF: {os.path.basename(pdf_path)}")
            else:
                st.sidebar.error(f"📄 PDF not found: {pdf_path}")
    
    def display_main_interface(self):
        """Display main chat interface"""
        
        st.title("🏛️ Constitution RAG Chatbot")
        st.markdown("---")  
        
        # Display conversation
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)

        # Chat input
        if prompt := st.chat_input("Ask a question about the Constitution of Pakistan..."):
            self.process_user_input(prompt)
    
    def start_new_conversation(self):
        """Start a new conversation"""
        st.session_state.messages = []
        st.session_state.current_sources = []
        st.session_state.conversation_memory.clear()
        st.rerun()
    
    def clear_conversation(self):
        """Clear current conversation"""
        st.session_state.messages = []
        st.session_state.current_sources = []
        st.session_state.conversation_memory.clear()
        st.rerun()
    
    def load_conversation_history(self):
        """Load conversation history (placeholder)"""
        pass
    
    @st.fragment
    def process_user_input(self, user_input: str):
        """Process user input and generate response"""
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                st.session_state.conversation_memory.chat_memory.add_user_message(user_input)
                n_results = getattr(st.session_state, 'n_results', 5)
                print("calling ask question")
                response = self.ask_question(user_input, n_results)
            
            if "error" in response:
                error_msg = f"❌ Error: {response['error']}"
                st.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now()
                })
                return
            
            # Display answer with clickable PDF page references
            answer = response.get("answer", "No answer received")
            sources = response.get("sources", [])
            generator = st.session_state.rag_system['generator']

            # --- Remove unwanted tags like <think> ---
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

            st.markdown(answer, unsafe_allow_html=True) 
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "timestamp": datetime.now()
            })
            st.session_state.conversation_memory.chat_memory.add_ai_message(answer)
    
    def run(self):
        """Run the main application"""
        # Load conversation history on startup
        if 'history_loaded' not in st.session_state:
            self.load_conversation_history()
            st.session_state.history_loaded = True
        
        # Display minimal controls
        self.display_conversation_controls()
        self.display_settings()
        
        # Main interface
        self.display_main_interface()

# Run the application
if __name__ == "__main__":
    app = ConstitutionRAGApp()
    app.run()