"""Configuration module for the clinical trials pipeline.

This module provides a centralized configuration using Pydantic settings 
management for the clinical trials data pipeline.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file if it exists
load_dotenv()


class APIKeys(BaseModel):
    """API keys for external services."""

    gemini: Optional[str] = Field(None, description="Google Gemini API key")


class Paths(BaseModel):
    """Paths for data storage and outputs."""

    raw_data: Path = Field(Path("data/raw"), description="Path to raw data")
    processed_data: Path = Field(Path("data/processed"), description="Path to processed data")
    cache: Path = Field(Path("data/cache"), description="Path to cache data")
    figures: Path = Field(Path("data/figures"), description="Path to figures")
    release: Path = Field(Path("release"), description="Path to release artifacts")

    @validator("*")
    def create_directories(cls, v: Path) -> Path:
        """Create directories if they don't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


class CtGovSettings(BaseModel):
    """ClinicalTrials.gov API settings."""

    base_url: HttpUrl = Field(
        "https://clinicaltrials.gov/api/v2/studies",
        description="ClinicalTrials.gov API v2 base URL",
    )
    metadata_url: HttpUrl = Field(
        "https://clinicaltrials.gov/api/v2/studies/metadata", 
        description="URL for API metadata and schema information"
    )
    page_size: int = Field(100, description="Number of results per page (max 1000)")
    max_retries: int = Field(3, description="Maximum number of retry attempts")
    retry_delay: int = Field(1, description="Delay between retries in seconds")
    rate_limit: float = Field(10.0, description="Soft rate limit in requests per second")


env_path = os.path.join(os.getcwd(), ".env")
class Settings(BaseSettings):
    """Main configuration settings."""


    model_config = SettingsConfigDict(
        env_file=env_path, 
        env_file_encoding="utf-8", 
        case_sensitive=False,
        extra="ignore"  # Allow extra fields from environment variables
    )
    

    # Study extraction parameters
    disease: str = Field(
        "Familial Hypercholesterolemia", description="Disease to search for"
    )
    year_start: int = Field(
        datetime.now().year - 15, description="Start year for study search"
    )
    year_end: int = Field(datetime.now().year, description="End year for study search")

    # API keys
    api_keys: APIKeys = Field(default_factory=lambda: APIKeys(
        gemini=os.environ.get("GEMINI_API_KEY"),
    ))


    # Paths
    paths: Paths = Field(default_factory=Paths)

    # API settings
    ctgov: CtGovSettings = Field(default_factory=CtGovSettings)

    # Logging
    log_level: str = Field("INFO", description="Logging level")

    # Pipeline settings
    concurrency_limit: int = Field(5, description="Maximum number of concurrent API requests")
    cache_db_path: str = Field(
        "data/drug_cache.sqlite", description="Path to drug cache database"
    )
    gemini_model: str = Field(
        "gemini-2.0-flash", description="Google Gemini model to use for enrichment"
    )

    # New settings
    max_studies: Optional[int] = Field(
        default=None,
        description="Maximum number of studies to fetch from the API (None for unlimited). Can be set via .env as MAX_STUDIES."
    )
    max_pages: Optional[int] = Field(
        default=None,
        description="Maximum number of pages to fetch from the API (None for unlimited). Can be set via .env as MAX_PAGES."
    )


# Create singleton instance
settings = Settings() 
print("settings.disease:", settings.disease)
print("settings.year_start:", settings.year_start)
print("settings.year_end:", settings.year_end)