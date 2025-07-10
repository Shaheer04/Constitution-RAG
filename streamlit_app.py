"""
Streamlit-only Constitution RAG System
Integrates everything directly without separate backend
"""

import streamlit as st
import os
import logging
from datetime import datetime
from typing import List, Dict, Any
import uuid

# LangChain imports for conversation memory
from langchain.memory import ConversationBufferWindowMemory

# Import your existing RAG components
from app.pre_processor import PreProcessor
from app.retriever import Retriever
from app.generator import Generator

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

# Custom CSS
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

.source-box {
    background-color: #f5f5f5;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 4px solid #ff9800;
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
                k=10,  # Keep last 10 exchanges
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
            
            # Configuration
            pdf_path = "./data/constitution-1973.pdf"
            db_path = "./constitution_db"
            embedding_model = "all-MiniLM-L6-v2"
            ollama_model = "qwen3"
            ollama_url = "https://e1d4c021133d.ngrok-free.app/"
            
            # Initialize components
            preprocessor = PreProcessor(
                embedding_model_name=embedding_model,
                chroma_db_path=db_path,
                ollama_model=ollama_model,
                ollama_url=ollama_url
            )
            
            retriever = Retriever(
                embedding_model_name=embedding_model,
                chroma_db_path=db_path
            )
            
            generator = Generator(
                ollama_model=ollama_model,
                ollama_url=ollama_url,
                pdf_path=pdf_path, 
                base_url=""         
            )
            
            return {
                'preprocessor': preprocessor,
                'retriever': retriever,
                'generator': generator,
                'pdf_path': pdf_path,
                'db_path': db_path
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
                    
                    # Check if database needs setup
                    if self.needs_database_setup():
                        st.warning("Database is empty. Setting up database...")
                        self.setup_database()
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
    
    def setup_database(self):
        """Setup the database"""
        try:
            rag_system = st.session_state.rag_system
            pdf_path = rag_system['pdf_path']
            
            if not os.path.exists(pdf_path):
                st.error(f"PDF file not found: {pdf_path}")
                return False
            
            with st.spinner("📖 Processing constitution PDF..."):
                # Process document
                preprocessor = rag_system['preprocessor']
                
                # Convert document
                document = preprocessor.convert_document(pdf_path)
                st.success("✅ Document converted successfully")
                
                # Create chunks
                chunks = preprocessor.create_hybrid_chunks(document, max_tokens=1000, overlap_ratio=0.1)
                st.success(f"✅ Created {len(chunks)} chunks")
                
                # Embed and store
                preprocessor.embed_and_store(chunks, document_name="constitution_of_pakistan")
                st.success("✅ Database setup completed!")
                
            return True
            
        except Exception as e:
            st.error(f"Error setting up database: {e}")
            logger.error(f"Database setup error: {e}")
            return False
    
    def test_system_health(self) -> Dict[str, Any]:
        """Test system health"""
        health_status = {
            "overall": False,
            "generator": False,
            "retriever": False,
            "database": False,
            "document_count": 0,
            "errors": []
        }
        
        try:
            rag_system = st.session_state.rag_system
            if not rag_system:
                health_status["errors"].append("RAG system not initialized")
                return health_status
            
            # Test generator
            try:
                generator = rag_system['generator']
                test_result = generator.test_connection()
                health_status["generator"] = test_result.get("status") == "success"
                if not health_status["generator"]:
                    health_status["errors"].append(f"Generator: {test_result.get('error', 'Unknown error')}")
            except Exception as e:
                health_status["errors"].append(f"Generator test failed: {e}")
            
            # Test retriever and database
            try:
                retriever = rag_system['retriever']
                test_results = retriever.retrieve_relevant_chunks("test", n_results=1)
                health_status["retriever"] = isinstance(test_results, list)
                
                if health_status["retriever"]:
                    # Check database content
                    all_docs = retriever.db_manager.get_all_documents()
                    if all_docs and all_docs.get('documents'):
                        health_status["document_count"] = len(all_docs['documents'])
                        health_status["database"] = health_status["document_count"] > 0
                    else:
                        health_status["errors"].append("Database is empty")
                else:
                    health_status["errors"].append("Retriever test failed")
                    
            except Exception as e:
                health_status["errors"].append(f"Retriever test failed: {e}")
            
            # Overall health
            health_status["overall"] = all([
                health_status["generator"],
                health_status["retriever"],
                health_status["database"]
            ])
            
        except Exception as e:
            health_status["errors"].append(f"Health check failed: {e}")
        
        return health_status
    
    def ask_question(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        """Ask question to RAG system"""
        try:
            rag_system = st.session_state.rag_system
            if not rag_system:
                return {"error": "RAG system not initialized"}
            
            retriever = rag_system['retriever']
            generator = rag_system['generator']
            
            # Retrieve relevant documents
            results = retriever.hybrid_retrieve(question)
            reranked_results = retriever.reranker.rerank_results(question, results, n_results)
            
            if not reranked_results:
                return {
                    "answer": "No relevant information found in the constitution database.",
                    "sources": []
                }
            
            # Generate response (this will now include citations!)
            answer = generator.generate_response(question, reranked_results)
            
            # Format sources - Fix the attribute access here
            sources = []
            for i, result in enumerate(reranked_results):
                sources.append({
                    "id": i + 1,
                    "text": result.text[:300] + "..." if len(result.text) > 300 else result.text,  # Use .text instead of .get("text")
                    "metadata": getattr(result, 'metadata', {}),  # Use getattr for safety
                    "score": getattr(result, 'score', 0.0)  # Use getattr for safety
                })
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {"error": str(e)}
    
    def display_system_status(self):
        """Display system status in sidebar"""
        st.sidebar.markdown("## 🔧 System Status")
        
        # Get health status
        health = self.test_system_health()
        
        # Display overall status
        if health["overall"]:
            st.sidebar.success("✅ System Healthy")
        else:
            st.sidebar.error("❌ System Issues")
        
        # Display errors if any (only if there are issues)
        if health["errors"]:
            with st.sidebar.expander("⚠️ Issues"):
                for error in health["errors"]:
                    st.error(error)
        
        return health["overall"]
    
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
        
        # Database management (collapsed by default)
        with st.sidebar.expander("🗄️ Database Management"):
            if st.button("🔄 Rebuild Database", help="Rebuild document database"):
                with st.spinner("Rebuilding database..."):
                    if self.setup_database():
                        st.success("Database rebuilt!")
                    else:
                        st.error("Rebuild failed!")

    def display_sources(self, sources: List[Dict]):
        """Display source documents"""
        if not sources:
            return
        
        with st.expander(f"📚 Source Documents ({len(sources)})", expanded=False):
            for source in sources:
                st.markdown(f"""
                <div class="source-box">
                    <strong>📄 Source {source['id']} (Relevance: {source['score']:.3f})</strong><br>
                    <p>{source['text']}</p>
                    <small><em>Metadata: {source.get('metadata', {})}</em></small>
                </div>
                """, unsafe_allow_html=True)
    
    def display_chat_stats(self):
        """Display chat statistics"""
        if st.session_state.messages:
            total_messages = len(st.session_state.messages)
            user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
            assistant_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
            
            st.markdown(f"""
            <div class="chat-stats">
                <h4>📊 Chat Statistics</h4>
                <p><strong>Total Messages:</strong> {total_messages}</p>
                <p><strong>Your Questions:</strong> {user_messages}</p>
                <p><strong>AI Responses:</strong> {assistant_messages}</p>
                <p><strong>Session Duration:</strong> {datetime.now().strftime('%H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def display_main_interface(self):
        """Display main chat interface"""
        st.title("🏛️ Constitution of Pakistan RAG Chatbot")
        st.markdown("Ask questions about the Constitution of Pakistan. This chatbot uses advanced retrieval and generation to provide accurate answers.")
        
        # Display chat statistics
        if st.session_state.messages:
            self.display_chat_stats()
        
        # Display conversation
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources for assistant messages
                if message["role"] == "assistant" and "sources" in message:
                    self.display_sources(message["sources"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the Constitution of Pakistan..."):
            self.process_user_input(prompt)
    
    def process_user_input(self, user_input: str):
        """Process user input and generate response"""
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Add to LangChain memory
        st.session_state.conversation_memory.chat_memory.add_user_message(user_input)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            # Show thinking indicator
            with st.spinner("🤔 Thinking..."):
                n_results = getattr(st.session_state, 'n_results', 5)
                response = self.ask_question(user_input, n_results)
            
            if "error" in response:
                error_msg = f"❌ Error: {response['error']}"
                st.markdown(error_msg)
                
                # Add error to messages
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now()
                })
                return
            
            # Display answer
            answer = response.get("answer", "No answer received")
            st.markdown(answer)
            
            # Display sources
            sources = response.get("sources", [])
            if sources:
                self.display_sources(sources)
                st.session_state.current_sources = sources
            
            # Add to messages
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "timestamp": datetime.now()
            })
            
            # Add to LangChain memory
            st.session_state.conversation_memory.chat_memory.add_ai_message(answer)
    
    def run(self):
        """Run the main application"""
        # Load conversation history on startup
        if 'history_loaded' not in st.session_state:
            self.load_conversation_history()
            st.session_state.history_loaded = True
        
        # Check system health
        is_healthy = self.display_system_status()
        
        if not is_healthy:
            st.error("⚠️ System has issues. Some features may not work properly.")
        
        # Display minimal controls
        self.display_conversation_controls()
        self.display_settings()
        
        # Main interface
        self.display_main_interface()