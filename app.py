import os
import atexit
import threading
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
from dotenv import load_dotenv

# LangChain imports
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Text processing imports
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredMarkdownLoader

# Vector store imports
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams

# For hybrid search result re-ranking
from langchain.retrievers.multi_query import MultiQueryRetriever

# Global client instance
_qdrant_client = None
_qdrant_client_lock = threading.Lock()

def get_qdrant_client(persist_dir=None):
    """Thread-safe singleton pattern for Qdrant client"""
    global _qdrant_client
    
    if _qdrant_client is None:
        with _qdrant_client_lock:
            if _qdrant_client is None:
                if persist_dir:
                    # Make sure the directory exists
                    os.makedirs(persist_dir, exist_ok=True)
                    _qdrant_client = QdrantClient(path=persist_dir)
                else:
                    _qdrant_client = QdrantClient(":memory:")
                
                # Register cleanup function
                atexit.register(lambda: _qdrant_client.close() if _qdrant_client else None)
    
    return _qdrant_client


class HybridRagPipeline:
    def __init__(
        self,
        markdown_dir: str,
        collection_name: str = "hybrid_markdown_docs",
        dense_embedding_model: str = "text-embedding-3-large",
        sparse_embedding_model: str = "Qdrant/bm25",
        llm_model: str = "gpt-4.1",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 4,
        persist_dir: Optional[str] = None,
        auto_index: bool = False,
    ):
        """
        Initialize the Hybrid RAG pipeline with markdown documents.
        
        Args:
            markdown_dir: Directory containing markdown files
            collection_name: Name for the Qdrant collection
            dense_embedding_model: OpenAI embedding model to use
            sparse_embedding_model: Sparse embedding model to use
            llm_model: LLM model to use for the generation
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            k: Number of documents to retrieve
            persist_dir: Directory to persist Qdrant data (None for in-memory)
            auto_index: Whether to auto-index documents on startup
        """
        self.markdown_dir = markdown_dir
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.persist_dir = persist_dir
        
        # Initialize embeddings
        self.dense_embeddings = OpenAIEmbeddings(model=dense_embedding_model)
        self.sparse_embeddings = FastEmbedSparse(model_name=sparse_embedding_model)
        
        # Initialize LLM
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        
        # Initialize Qdrant client using the singleton pattern
        self.client = get_qdrant_client(persist_dir)
        
        # Initialize vector store
        self._setup_vector_store()

        if auto_index:
            self.load_and_index_documents()
        
        # Load and process documents
        self.documents = self._load_and_process_documents()
        
        # Index documents if needed
        if self.documents:
            self._index_documents()
        
        # Create retrieval chain
        self.retrieval_chain = self._create_retrieval_chain()
    
    def _setup_vector_store(self):
        """Set up the Qdrant vector store with hybrid search capabilities."""
        # Check if collection exists, if not create it
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            # Create a new collection with dense and sparse vectors
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=3072, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
                },
            )
        
        # Initialize vector store with hybrid search
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.dense_embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        
        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k}
        )
    
    def load_and_index_documents(self, force_reindex: bool = False):
        """
        Load documents and index them if they're not already indexed or if force_reindex is True.
        
        Args:
            force_reindex: If True, reindex documents even if they already exist
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                print(f"Collection {self.collection_name} doesn't exist yet. Creating...")
                # Collection will be created in _setup_vector_store
                count = 0
            else:
                # Get collection info
                collection_info = self.client.get_collection(self.collection_name)
                # Safely get vectors count, defaulting to 0 if not available
                count = getattr(collection_info, 'vectors_count', 0) or 0
            
            # If count is still None, set it to 0 to avoid comparison errors
            if count is None:
                count = 0
                
            if count > 0 and not force_reindex:
                print(f"Collection already contains {count} vectors. Skipping indexing.")
                return
            
            # Load and process documents
            self.documents = self._load_and_process_documents()
            
            # Index documents if needed
            if self.documents:
                self._index_documents()
                
        except Exception as e:
            import traceback
            print(f"Error in load_and_index_documents: {e}")
            print(traceback.format_exc())
            # If there's an error, try to proceed with empty documents
            self.documents = []

    def _load_and_process_documents(self) -> List[Document]:
        """Load markdown documents and split them into chunks."""
        # Check if directory exists
        if not os.path.exists(self.markdown_dir):
            print(f"Directory {self.markdown_dir} not found, creating it")
            os.makedirs(self.markdown_dir, exist_ok=True)
            return []
        
        # Create document loader for markdown files
        # Using UnstructuredMarkdownLoader for better processing of markdown structure
        loader = DirectoryLoader(
            self.markdown_dir,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader
        )
        
        # Load documents
        documents = loader.load()
        
        if not documents:
            print(f"No markdown documents found in {self.markdown_dir}")
            return []
        
        print(f"Loaded {len(documents)} markdown documents")
        
        # Create markdown-specific text splitter
        splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            # Keep headers with content for better context
            keep_separator=True,
        )
        
        # Split documents
        split_documents = splitter.split_documents(documents)
        
        print(f"Split into {len(split_documents)} chunks")
        return split_documents
    
    def _index_documents(self):
        """Index the documents in the vector store."""
        if not self.documents:
            print("No documents to index.")
            return
            
        try:
            # Generate UUIDs for documents
            uuids = [str(uuid.uuid4()) for _ in range(len(self.documents))]
            
            # Add documents to vector store
            self.vector_store.add_documents(documents=self.documents, ids=uuids)
            print(f"Indexed {len(self.documents)} document chunks in Qdrant")
        except Exception as e:
            import traceback
            print(f"Error indexing documents: {e}")
            print(traceback.format_exc())
    
    def get_context_for_query(self, input_dict):
        """Retrieve relevant documents for a question."""
        # Extract query from input dictionary
        query = input_dict.get("question", "")
        
        try:
            # Get relevant documents
            docs = self.retriever.get_relevant_documents(query)
            
            # Format documents as string
            return "\n\n".join(doc.page_content for doc in docs)
        except Exception as e:
            import traceback
            print(f"Error getting context for query: {e}")
            print(traceback.format_exc())
            return "No relevant context found."
    
    def _create_retrieval_chain(self):
        """Create the retrieval chain for the RAG pipeline."""
        # Define the prompt template with better context integration and chat history
        template = """
            You are an AI assistant designed to help users by answering their questions using the information provided in the context and chat history.

            Your tone should be natural, friendly, and conversational—like a helpful and knowledgeable companion. Keep answers clear and detailed when the information is available, and be polite and transparent when it's not.

            {chat_history}

            Here’s some information you can use:
            {context}

            User question: {question}

            Instructions:
            1. Use the context and chat history to provide a helpful and relevant answer, but don’t invent or rely on knowledge outside of what's provided.
            2. If the user greets you (e.g., says "hi" or "hello"), respond with a friendly greeting and ask about what they need to know.
            3. If the information isn't available in the context, let the user know politely and offer to help further if possible.
            4. If the context includes conflicting details, mention this and explain the different possibilities clearly.
            5. If the user asks about something unrelated to the courses offered by the Australian International College of Business and Technology (AICBT), gently redirect them and let them know you're here to help specifically with AICBT course information.
            6. Keep the response easy to follow. Use the same technical or precise language as in the context when needed, but avoid sounding robotic.
            7. Maintain continuity with previous conversation when appropriate.

            Your reply:
            """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the chain with simple structure
        chain = (
            {
                "context": RunnableLambda(self.get_context_for_query),
                "question": lambda x: x["question"],
                "chat_history": lambda x: x.get("chat_history", "")
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def query(self, question: str, chat_history: List[Dict[str, Any]] = None) -> str:
        """
        Query the hybrid RAG pipeline with a question and optional chat history.
        
        Args:
            question: The question to ask
            chat_history: Optional list of previous messages 
                
        Returns:
            The generated answer
        """
        # Initialize chat history if None
        if chat_history is None:
            chat_history = []
        
        try:
            # Print chat history for debugging
            print(f"Processing query: {question}")
            print(f"Chat history (length {len(chat_history)}): {chat_history[:2]}...")
            
            # Format chat history for the prompt
            formatted_history = ""
            if len(chat_history) > 0:
                formatted_history = "Chat history:\n"
                for message in chat_history:
                    # Handle different chat history formats
                    if isinstance(message, dict):
                        if "role" in message and "content" in message:
                            # Standard format
                            role = message.get("role", "")
                            content = message.get("content", "")
                            formatted_history += f"{role.capitalize()}: {content}\n"
                        elif "sender" in message and "message" in message:
                            # Custom format
                            sender = message.get("sender", "")
                            content = message.get("message", "")
                            formatted_history += f"{sender.capitalize()}: {content}\n"
                        else:
                            # Unknown format - log it
                            print(f"Unknown message format: {message}")
                    else:
                        print(f"Non-dict message in chat history: {type(message)}")
            
            # Use the retrieval chain with chat history
            return self.retrieval_chain.invoke({
                "question": question,
                "chat_history": formatted_history
            })
        except Exception as e:
            import traceback
            print(f"Error in query: {e}")
            print(traceback.format_exc())
            return "I'm experiencing technical difficulties. Please try again later."
    
    def add_documents(self, documents: List[Document]):
        """
        Add additional documents to the vector store.
        
        Args:
            documents: List of Document objects to add
        """
        if not documents:
            print("No documents to add.")
            return
            
        try:
            uuids = [str(uuid.uuid4()) for _ in range(len(documents))]
            self.vector_store.add_documents(documents=documents, ids=uuids)
            print(f"Added {len(documents)} additional documents to Qdrant")
        except Exception as e:
            import traceback
            print(f"Error adding documents: {e}")
            print(traceback.format_exc())
    
    def get_raw_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Get raw retrieved documents for a query (for debugging/analysis).
        
        Args:
            query: The query string
            k: Optional number of documents to retrieve (defaults to self.k)
            
        Returns:
            List of retrieved documents
        """
        try:
            k = k or self.k
            return self.retriever.get_relevant_documents(query)
        except Exception as e:
            import traceback
            print(f"Error getting raw documents: {e}")
            print(traceback.format_exc())
            return []