"""
Gemini LLM client wrapper with configuration.
Provides embeddings and chat models using Google's Gemini API.
"""
from typing import Optional, List
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from ..config import settings


class GeminiClient:
    """Wrapper for Gemini API client."""
    
    def __init__(self):
        self._chat_model: Optional[ChatGoogleGenerativeAI] = None
        self._embeddings: Optional[GoogleGenerativeAIEmbeddings] = None
        self._configure()
    
    def _configure(self) -> None:
        """Configure Gemini API with credentials."""
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=settings.gemini_api_key)
        print(f"Configured Gemini API with model: {settings.gemini_model}")
    
    @property
    def chat_model(self) -> ChatGoogleGenerativeAI:
        """
        Get the Gemini chat model instance.
        
        Returns:
            ChatGoogleGenerativeAI: Configured chat model
        """
        if self._chat_model is None:
            self._chat_model = ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                temperature=settings.gemini_temperature,
                max_output_tokens=settings.gemini_max_output_tokens,
                google_api_key=settings.gemini_api_key,
            )
        return self._chat_model
    
    @property
    def embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """
        Get the Gemini embeddings instance.
        
        Returns:
            GoogleGenerativeAIEmbeddings: Configured embeddings model
        """
        if self._embeddings is None:
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.embedding_model,
                google_api_key=settings.gemini_api_key,
            )
        return self._embeddings
    
    def generate_text(self, prompt: str) -> str:
        """
        Generate text using Gemini chat model.
        
        Args:
            prompt: The input prompt
            
        Returns:
            str: Generated text response
        """
        response = self.chat_model.invoke(prompt)
        return response.content
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: The input text
            
        Returns:
            List[float]: Embedding vector
        """
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)


# Global Gemini client instance
gemini_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """
    Get or create the global Gemini client instance.
    
    Returns:
        GeminiClient: Configured Gemini client
    """
    global gemini_client
    if gemini_client is None:
        gemini_client = GeminiClient()
    return gemini_client
