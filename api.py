from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import uvicorn
from dotenv import load_dotenv
import threading
import traceback
import json

# Import the HybridRagPipeline and the Qdrant client singleton from your module
from app import HybridRagPipeline, get_qdrant_client

# Load environment variables
load_dotenv()

# Create a lock for thread-safe access to the RAG pipeline
rag_lock = threading.Lock()

# Global variable for the RAG pipeline
rag = None

# Create FastAPI app
app = FastAPI(
    title="Hybrid RAG API",
    description="API for querying a hybrid retrieval-augmented generation system with markdown documents",
    version="1.0.0"
)

# Singleton pattern for the RAG pipeline
def get_rag_pipeline():
    global rag
    if rag is None:
        with rag_lock:
            if rag is None:
                # Initialize without auto-indexing
                rag = HybridRagPipeline(
                    markdown_dir="./output_markdown",
                    collection_name="hybrid_markdown_collection",
                    chunk_size=1000,
                    chunk_overlap=200,
                    persist_dir="./qdrant_data",
                    k=4,
                    auto_index=False  # Don't auto-index on startup
                )
                
                # Manually load and index documents (checks if already indexed)
                rag.load_and_index_documents()
    return rag

# Define request and response models
class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask the RAG system")
    chat_history: List[Dict[str, Any]] = Field(default=[], description="Conversation history as a list of message objects")

class ConversationQueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask the RAG system")
    history: List[Dict[str, Any]] = Field(default=[], description="Conversation history as a list of message objects")

class RAGResponse(BaseModel):
    answer: str = Field(..., description="The generated answer")
    retrieved_docs: Optional[List[Dict[str, Any]]] = Field(None, description="Optional retrieved documents for debugging")

# Helper function to format retrieved documents
def format_retrieved_docs(docs):
    return [
        {
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "metadata": doc.metadata
        }
        for doc in docs
    ]

# Exception handler for internal server errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_detail = f"An error occurred: {str(exc)}\n{traceback.format_exc()}"
    print(error_detail)  # Log the error
    return JSONResponse(
        status_code=500,
        content={"detail": f"Error processing request: {str(exc)}"}
    )

# Simple RAG endpoint
@app.post("/query", response_model=RAGResponse, tags=["RAG"])
async def query_rag(request: QueryRequest = Body(...)):
    """
    Query the RAG system with a question and chat history.
    """
    try:
        # Debug logging
        print(f"Received query: {request.query}")
        print(f"Chat history length: {len(request.chat_history)}")
        
        # Get the RAG pipeline
        pipeline = get_rag_pipeline()
        
        # Validate chat history format
        if request.chat_history:
            for i, msg in enumerate(request.chat_history):
                if not isinstance(msg, dict):
                    print(f"Warning: chat_history item {i} is not a dict: {type(msg)}")
                    continue
                
                # Check for required keys
                if "sender" not in msg or "message" not in msg:
                    if "role" not in msg or "content" not in msg:
                        print(f"Warning: chat_history item {i} missing required keys: {msg.keys()}")
        
        # Get answer from RAG pipeline with chat history
        answer = pipeline.query(request.query, request.chat_history)
        
        # Get retrieved documents for debugging (optional)
        retrieved_docs = pipeline.get_raw_documents(request.query)
        formatted_docs = format_retrieved_docs(retrieved_docs)
        
        return {
            "answer": answer,
            "retrieved_docs": formatted_docs
        }
    except Exception as e:
        error_detail = f"Error processing query: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # Log the error
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Conversation endpoint (alternative format for chat history)
@app.post("/conversation", response_model=RAGResponse, tags=["RAG"])
async def conversation_rag(request: ConversationQueryRequest = Body(...)):
    """
    Query the RAG system with a question and chat history in a different format.
    Converts the history format to the expected chat_history format.
    """
    try:
        # Get the RAG pipeline
        pipeline = get_rag_pipeline()
        
        # Convert history to chat_history format if needed
        chat_history = request.history
        
        # Get answer from RAG pipeline with chat history
        answer = pipeline.query(request.query, chat_history)
        
        # Get retrieved documents for debugging (optional)
        retrieved_docs = pipeline.get_raw_documents(request.query)
        formatted_docs = format_retrieved_docs(retrieved_docs)
        
        return {
            "answer": answer,
            "retrieved_docs": formatted_docs
        }
    except Exception as e:
        error_detail = f"Error processing query: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # Log the error
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "healthy"}

# Run the application
if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
    
    # Run the FastAPI app - IMPORTANT: disable reload for local Qdrant
    # Enable debug mode
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, log_level="debug")