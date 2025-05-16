"""Configuration module for the clinical trials pipeline.

This module provides a centralized configuration using Pydantic settings 
management for the clinical trials data pipeline.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel, Field, HttpUrl, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file if it exists



class APIKeys(BaseModel):
    """API keys for external services."""

    gemini: Optional[str] = Field(None, description="Google Gemini API key")

class Paths(BaseModel):
    base: Path
    raw_data: Path
    processed_data: Path
    cache: Path
    figures: Path
    release: Path

    @validator("*", pre=True)
    def create_directories(cls, v: Path) -> Path:
        # Skip directory creation during Prefect deploy builds
        if os.getenv("PREFECT_DEPLOYMENT_BUILD") != "true":
            try:
                v.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                pass  # Skip if mkdir is not permitted (like read-only fs)
        return v

    @classmethod
    def from_base(cls, base_path: str):
        base = Path(base_path)
        return cls(
            base=base,
            raw_data=base / "data" / "raw",
            processed_data=base / "data" / "processed",
            cache=base / "data" / "cache",
            figures=base / "data" / "figures",
            release=base / "release",
        )


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

    base_path: str = Field(".", env="BASE_PATH")

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


    # API settings
    ctgov: CtGovSettings = Field(default_factory=CtGovSettings)

    # Logging
    log_level: str = Field("INFO", description="Logging level")

    # Pipeline settings
    concurrency_limit: int = Field(5, description="Maximum number of concurrent API requests")
    cache_db_path: str = Field(
        "/tmp/drug_cache.sqlite", description="Path to drug cache database"
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

    @property
    def paths(self):
        return Paths.from_base(self.base_path)


# Create singleton instance
settings = Settings() 