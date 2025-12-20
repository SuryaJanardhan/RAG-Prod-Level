"""
Vector database initialization and management.
Supports Qdrant (prod) and Chroma (dev) based on configuration.
"""
from typing import Optional
from abc import ABC, abstractmethod
from langchain_community.vectorstores import Qdrant, Chroma
from langchain_core.vectorstores import VectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import settings


class VectorDBClient(ABC):
    """Abstract base class for vector database clients."""
    
    @abstractmethod
    def get_vectorstore(self, embeddings) -> VectorStore:
        """Get the vector store instance."""
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the vector database."""
        pass


class QdrantVectorDB(VectorDBClient):
    """Qdrant vector database client for production."""
    
    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self.collection_name = settings.qdrant_collection_name
        
    def initialize(self) -> None:
        """Initialize Qdrant client and create collection if needed."""
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        
        # Check if collection exists, create if not
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            print(f"Created Qdrant collection: {self.collection_name}")
        else:
            print(f"Using existing Qdrant collection: {self.collection_name}")
    
    def get_vectorstore(self, embeddings) -> VectorStore:
        """Get Qdrant vector store instance."""
        if not self.client:
            self.initialize()
        
        return Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=embeddings,
        )


class ChromaVectorDB(VectorDBClient):
    """Chroma vector database client for development."""
    
    def __init__(self):
        self.persist_directory = settings.chroma_persist_directory
        self.client: Optional[chromadb.Client] = None
        
    def initialize(self) -> None:
        """Initialize Chroma client with persistent storage."""
        self.client = chromadb.Client(
            ChromaSettings(
                persist_directory=self.persist_directory,
                anonymized_telemetry=False,
            )
        )
        print(f"Initialized Chroma DB at: {self.persist_directory}")
    
    def get_vectorstore(self, embeddings) -> VectorStore:
        """Get Chroma vector store instance."""
        if not self.client:
            self.initialize()
        
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings,
            client=self.client,
        )


def get_vector_db() -> VectorDBClient:
    """
    Factory function to get the appropriate vector database client
    based on environment configuration.
    
    Returns:
        VectorDBClient: Configured vector database client
    """
    if settings.vector_db == "qdrant":
        print("Using Qdrant vector database (production)")
        return QdrantVectorDB()
    else:
        print("Using Chroma vector database (development)")
        return ChromaVectorDB()


# Global vector DB instance
vector_db_client: Optional[VectorDBClient] = None


def initialize_vector_db() -> VectorDBClient:
    """Initialize and return the global vector database client."""
    global vector_db_client
    if vector_db_client is None:
        vector_db_client = get_vector_db()
        vector_db_client.initialize()
    return vector_db_client
