"""
Configuration settings for SHL Recommendation System
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_TITLE: str = "SHL Assessment Recommendation API"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Model Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-1.5-flash-001"
    
    # Data Paths
    ASSESSMENTS_JSON: str = "data/shl_assessments.json"
    EMBEDDINGS_FILE: str = "data/assessment_embeddings.npy"
    TRAIN_DATA: str = "data/train_labeled.csv"
    TEST_DATA: str = "data/test_unlabeled.csv"

    
    # Recommendation Settings
    MIN_RECOMMENDATIONS: int = 5
    MAX_RECOMMENDATIONS: int = 10
    SEMANTIC_WEIGHT: float = 0.6
    KEYWORD_WEIGHT: float = 0.4
    
    # Search Settings
    SIMILARITY_THRESHOLD: float = 0.3
    CANDIDATE_POOL_SIZE: int = 30
    
    # Performance Settings
    ENABLE_CACHING: bool = True
    CACHE_TTL: int = 3600  # 1 hour
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()
