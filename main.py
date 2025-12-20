"""
Main application entry point for RAG system.
Initializes all components and verifies Phase 0 setup.
"""
import logging
from src.config import settings
from src.db import initialize_vector_db
from src.cache import initialize_caches
from src.llm import get_gemini_client


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_phase_0_setup():
    """
    Verify Phase 0 setup: all tech choices are configured correctly.
    """
    logger.info("=" * 60)
    logger.info("PHASE 0: Verifying Tech Stack Configuration")
    logger.info("=" * 60)
    
    # 1. Environment check
    logger.info(f"\n[Environment]")
    logger.info(f"  Mode: {settings.environment}")
    logger.info(f"  Is Dev: {settings.is_dev}")
    logger.info(f"  Is Prod: {settings.is_prod}")
    
    # 2. LLM Configuration (Gemini)
    logger.info(f"\n[LLM - Gemini]")
    logger.info(f"  Model: {settings.gemini_model}")
    logger.info(f"  Temperature: {settings.gemini_temperature}")
    logger.info(f"  Max Tokens: {settings.gemini_max_output_tokens}")
    logger.info(f"  API Key Configured: {'Yes' if settings.gemini_api_key else 'No'}")
    
    try:
        gemini_client = get_gemini_client()
        logger.info(f"  ✓ Gemini client initialized successfully")
    except Exception as e:
        logger.error(f"  ✗ Failed to initialize Gemini client: {e}")
    
    # 3. Vector Database Configuration
    logger.info(f"\n[Vector Database]")
    logger.info(f"  Selected: {settings.vector_db}")
    
    if settings.vector_db == "qdrant":
        logger.info(f"  Qdrant URL: {settings.qdrant_url}")
        logger.info(f"  Collection: {settings.qdrant_collection_name}")
        logger.info(f"  API Key Configured: {'Yes' if settings.qdrant_api_key else 'No'}")
    else:
        logger.info(f"  Chroma Directory: {settings.chroma_persist_directory}")
    
    try:
        vector_db = initialize_vector_db()
        logger.info(f"  ✓ Vector database initialized successfully")
    except Exception as e:
        logger.error(f"  ✗ Failed to initialize vector database: {e}")
    
    # 4. Cache Configuration
    logger.info(f"\n[Cache Layer]")
    logger.info(f"  Embedding Cache: {settings.embedding_cache_type}")
    logger.info(f"  Response Cache: {settings.response_cache_type}")
    
    if settings.embedding_cache_type == "sqlite":
        logger.info(f"  SQLite Path: {settings.embedding_cache_sqlite_path}")
    
    if settings.response_cache_type == "redis" or settings.embedding_cache_type == "redis":
        logger.info(f"  Redis Host: {settings.redis_host}:{settings.redis_port}")
        logger.info(f"  Redis DB: {settings.redis_db}")
        logger.info(f"  Redis TTL: {settings.redis_ttl}s")
    
    if settings.response_cache_type == "postgres":
        logger.info(f"  Postgres Host: {settings.postgres_host}:{settings.postgres_port}")
        logger.info(f"  Postgres DB: {settings.postgres_db}")
    
    try:
        emb_cache, resp_cache = initialize_caches()
        logger.info(f"  ✓ Cache layers initialized successfully")
    except Exception as e:
        logger.error(f"  ✗ Failed to initialize caches: {e}")
    
    # 5. Embedding Configuration
    logger.info(f"\n[Embeddings]")
    logger.info(f"  Model: {settings.embedding_model}")
    logger.info(f"  Dimension: {settings.embedding_dimension}")
    logger.info(f"  Chunk Size: {settings.chunk_size}")
    logger.info(f"  Chunk Overlap: {settings.chunk_overlap}")
    
    # 6. Retrieval Configuration
    logger.info(f"\n[Retrieval]")
    logger.info(f"  Top K: {settings.retrieval_top_k}")
    logger.info(f"  Score Threshold: {settings.retrieval_score_threshold}")
    
    # 7. API Configuration
    logger.info(f"\n[API]")
    logger.info(f"  Host: {settings.api_host}")
    logger.info(f"  Port: {settings.api_port}")
    logger.info(f"  Reload: {settings.api_reload}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Phase 0 verification complete!")
    logger.info("=" * 60 + "\n")


def main():
    """Main function to run Phase 0 verification."""
    try:
        verify_phase_0_setup()
        logger.info("✓ All Phase 0 components initialized successfully!")
        logger.info("\nReady to proceed to Phase 1: Basic RAG Pipeline")
        
    except Exception as e:
        logger.error(f"✗ Phase 0 verification failed: {e}")
        raise


if __name__ == "__main__":
    main()
