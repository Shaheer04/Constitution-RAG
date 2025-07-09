"""
Main script to run the Constitution RAG system
Integrates PreProcessor, Retriever, and Generator components
"""

import os
import sys
import logging

# Import our components
from app.pre_processor import PreProcessor
from app.retriever import Retriever
from app.generator import Generator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConstitutionRAG:
    """Main RAG system that orchestrates all components"""
    
    def __init__(self, 
                 pdf_path: str = "./data/constitution.pdf",
                 db_path: str = "./constitution_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 ollama_model: str = "qwen2.5:3b",
                 ollama_url: str = "https://f997ec1eaec5.ngrok-free.app/"):
        """
        Initialize the Constitution RAG system
        
        Args:
            pdf_path: Path to the constitution PDF
            db_path: Path for ChromaDB storage
            embedding_model: Sentence transformer model for embeddings
            ollama_model: Ollama model for generation
            ollama_url: URL for remote Ollama instance (optional)
        """
        self.pdf_path = pdf_path
        self.db_path = db_path
        
        print("🔧 Initializing Constitution RAG System...")
        
        # Initialize components
        try:
            self.preprocessor = PreProcessor(
                embedding_model_name=embedding_model,
                chroma_db_path=db_path,
                ollama_model=ollama_model,
                ollama_url=ollama_url
            )
            self.retriever = Retriever(
                embedding_model_name=embedding_model,
                chroma_db_path=db_path
            )
            self.generator = Generator(
                ollama_model=ollama_model,
                ollama_url=ollama_url
            )
            print("✅ All components initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            sys.exit(1)
    
    def setup_database(self) -> bool:
        """Setup the database by processing the constitution PDF"""
        try:
            if not os.path.exists(self.pdf_path):
                print(f"❌ PDF file not found: {self.pdf_path}")
                return False
            
            print(f"📖 Processing PDF: {self.pdf_path}")
            
            # Step 1: Convert document
            print("🔄 Converting document...")
            document = self.preprocessor.convert_document(self.pdf_path)
            
            # Step 2: Create hybrid chunks
            print("🔧 Creating hybrid chunks...")
            chunks = self.preprocessor.create_hybrid_chunks(document, max_tokens=1000, overlap_ratio=0.1)
            print(f"📊 Created {len(chunks)} chunks")
            
            # Step 3: Embed and store
            print("💾 Embedding and storing chunks...")
            self.preprocessor.embed_and_store(chunks, document_name="constitution_of_pakistan")
            
            print("✅ Database setup completed successfully!")
            return True
                
        except Exception as e:
            print(f"❌ Error setting up database: {e}")
            return False
    
    def test_connections(self) -> bool:
        """Test all component connections"""
        print("🔍 Testing component connections...")
        
        # Test generator (Ollama) connection
        generator_test = self.generator.test_connection()
        if generator_test["status"] == "success":
            print(f"✅ Generator connected: {generator_test['model']}")
        else:
            print(f"❌ Generator connection failed: {generator_test.get('error', 'Unknown error')}")
            return False
        
        # Test retriever (database) connection
        try:
            test_results = self.retriever.retrieve_relevant_chunks("test", n_results=1)
            if isinstance(test_results, list):
                print("✅ Retriever connected to database")
                if len(test_results) > 0:
                    print(f"📊 Database contains documents")
                else:
                    print("⚠️ Database appears to be empty")
            else:
                print("❌ Retriever connection failed")
                return False
        except Exception as e:
            print(f"❌ Retriever connection failed: {e}")
            return False
        
        return True
    
    def ask_question(self, question: str, n_results: int = 5) -> str:
        """
        Ask a question to the RAG system using hybrid retrieval
        
        Args:
            question: User question
            n_results: Number of documents to retrieve
            
        Returns:
            Generated answer
        """
        try:
            print(f"🔍 Searching for relevant information using hybrid search...")
            
            # Always use hybrid retrieval
            retrieval_results = self.retriever.hybrid_retrieve(question, n_results=n_results)
            
            if not retrieval_results:
                return "❌ No relevant information found in the constitution database."
            
            print(f"📚 Found {len(retrieval_results)} relevant documents")
            
            # Generate response
            print("🤖 Generating response...")
            answer = self.generator.generate_response(question, retrieval_results)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return
    
    def interactive_mode(self):
        """Run the system in interactive mode"""
        print("\n" + "="*60)
        print("🏛️  CONSTITUTION OF PAKISTAN - RAG SYSTEM")
        print("="*60)
        print("Ask questions about the Constitution of Pakistan!")
        print("  - Type 'quit' or 'exit' to end")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("❓ Your question: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                else:
                    # Process question using hybrid retrieval
                    answer = self.ask_question(user_input)
                
                # Display answer
                print(f"\n💡 Answer:\n{answer}\n")
                print("-" * 60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")
    
    def debug_database_status(self):
        """Debug method to check database status"""
        try:
            print("🔍 Checking database status...")
            
            # Check if collection exists
            if not self.retriever.db_manager.collection:
                print("❌ No collection found in database")
                return False
                
            # Try to get documents
            all_docs = self.retriever.db_manager.get_all_documents()
            if all_docs and all_docs.get('documents'):
                doc_count = len(all_docs['documents'])
                print(f"✅ Database contains {doc_count} documents")
                
                # Show sample content
                if doc_count > 0:
                    sample_text = all_docs['documents'][0][:200]
                    print(f"📄 Sample document: {sample_text}...")
                    
                    # Check metadata
                    if all_docs.get('metadatas') and len(all_docs['metadatas']) > 0:
                        sample_metadata = all_docs['metadatas'][0]
                        print(f"📊 Sample metadata: {list(sample_metadata.keys())}")
                
                return True
            else:
                print("❌ Database collection exists but contains no documents")
                return False
                
        except Exception as e:
            print(f"❌ Error checking database: {e}")
            return False

def main():
    """Main function to run the Constitution RAG system"""
    
    # Configuration
    PDF_PATH = "./data/constitution-1973.pdf"
    DB_PATH = "./constitution_db"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    OLLAMA_MODEL = "qwen3" 
    OLLAMA_URL = "https://356e2f6fc9a3.ngrok-free.app/"  
    
    print("🚀 Starting Constitution RAG System...")
    
    # Initialize RAG system
    rag_system = ConstitutionRAG(
        pdf_path=PDF_PATH,
        db_path=DB_PATH,
        embedding_model=EMBEDDING_MODEL,
        ollama_model=OLLAMA_MODEL,
        ollama_url=OLLAMA_URL
    )
    
    # Check if database exists AND has actual content
    def database_has_content():
        """Check if database actually contains documents"""
        try:
            if not rag_system.retriever.db_manager.collection:
                return False
            
            # Try to get documents from the collection
            all_docs = rag_system.retriever.db_manager.get_all_documents()
            return all_docs and all_docs.get('documents') and len(all_docs['documents']) > 0
        except Exception:
            return False
    
    database_exists = database_has_content()
    
    if not database_exists:
        print("📊 Database is empty or doesn't contain valid data. Setting up database...")
        if not rag_system.setup_database():
            print("❌ Failed to setup database. Exiting...")
            return
    else:
        print("📊 Database found with content. Checking connections...")
        # Test connections to make sure everything works
        if not rag_system.test_connections():
            print("❌ Component connection tests failed.")
            choice = input("🔄 Do you want to rebuild the database? (y/N): ").strip().lower()
            if choice in ['y', 'yes']:
                if not rag_system.setup_database():
                    print("❌ Database rebuild failed. Exiting...")
                    return
            else:
                print("❌ Cannot proceed without working connections. Exiting...")
                return
        else:
            print("✅ All systems operational!")
    
    # Final connection test
    print("🔍 Final system check...")
    if not rag_system.test_connections():
        print("❌ System check failed. Please verify your setup.")
        return
    
    # Run interactive mode
    try:
        rag_system.interactive_mode()
    except Exception as e:
        print(f"❌ System error: {e}")

if __name__ == "__main__":
    main()
