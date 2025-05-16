"""Utility functions for the clinical trials pipeline.

This module provides helper functions for HTTP requests, retries, 
logging, and other utility operations.
"""

import asyncio
import json
import logging
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, cast
import os

import aiohttp
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from google.cloud import storage

from src.pipeline.config import settings

# Set up logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("clinical_trials")

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")


def get_timestamp() -> str:
    """Get a timestamp string for file naming (date and time)."""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def get_raw_data_path(timestamp: Optional[str] = None) -> Path:
    """Get the path to store raw data for a specific timestamp.
    
    Args:
        timestamp: Optional timestamp string (defaults to current date)
        
    Returns:
        Path object for the raw data directory
    """
    if timestamp is None:
        timestamp = get_timestamp()
    
    path = settings.paths.raw_data / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, filepath: Path) -> None:
    """Save data as JSON to the specified filepath.
    
    Args:
        data: Data to save as JSON
        filepath: Path where to save the file
    """
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug(f"Saved JSON data to {filepath}")


def load_json(filepath: Path) -> Any:
    """Load JSON data from the specified filepath.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Loaded JSON data
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    logger.debug(f"Loaded JSON data from {filepath}")
    return data


async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    **kwargs: Any,
) -> Any:
    """Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        *args: Arguments to pass to the function
        max_attempts: Maximum number of retry attempts
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function call
    """
    def is_retryable_error(exception):
        """Determine if the exception is retryable based on API docs."""
        if isinstance(exception, aiohttp.ClientResponseError):
            # Retry on rate limits (429) and server errors (5xx)
            return exception.status == 429 or exception.status >= 500
        # Retry on network errors and timeouts
        return isinstance(exception, (aiohttp.ClientError, asyncio.TimeoutError))
    
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=60),  # Longer max wait for rate limits
        retry=retry_if_exception_type(
            (aiohttp.ClientError, aiohttp.ClientResponseError, asyncio.TimeoutError)
        ),
        reraise=True,
    ):
        with attempt:
            attempt_num = attempt.retry_state.attempt_number
            if attempt_num > 1:
                logger.info(f"Retry attempt {attempt_num}/{max_attempts} for {func.__name__}")
            return await func(*args, **kwargs)


def log_execution_time(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to log the execution time of a function.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = datetime.now()
        logger.info(f"Starting {func.__name__}")
        
        result = await func(*args, **kwargs)
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Completed {func.__name__} in {duration}")
        
        return result
    
    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = datetime.now()
        logger.info(f"Starting {func.__name__}")
        
        result = func(*args, **kwargs)
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Completed {func.__name__} in {duration}")
        
        return result
    
    if asyncio.iscoroutinefunction(func):
        return cast(Callable[..., T], async_wrapper)
    return cast(Callable[..., T], sync_wrapper)


async def fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Fetch data from a URL with retry logic.
    
    Args:
        session: aiohttp client session
        url: URL to fetch data from
        params: Optional query parameters
        headers: Optional request headers
        
    Returns:
        JSON response data
    """
    params = params or {}
    headers = headers or {}
    
    # Rate limiting per API docs - approximately 10 req/s soft limit
    await asyncio.sleep(1.0 / settings.ctgov.rate_limit)
    
    async def _fetch() -> Dict[str, Any]:
        try:
            async with session.get(url, params=params, headers=headers) as response:
                response.raise_for_status()
                
                # Check content type to handle potential error responses
                content_type = response.headers.get('Content-Type', '')
                if 'application/json' in content_type:
                    return await response.json()
                else:
                    # Handle non-JSON responses (should be rare)
                    text = await response.text()
                    logger.warning(f"Non-JSON response received: {text[:200]}...")
                    return {"error": "Non-JSON response", "studies": []}
                    
        except aiohttp.ClientResponseError as e:
            # For 429 Too Many Requests, we want to retry with exponential backoff
            if e.status == 429:
                logger.warning(f"Rate limit exceeded (429): {e}. Backing off...")
                # The tenacity retry will handle backing off
            # For 5xx server errors, also retry
            elif e.status >= 500:
                logger.warning(f"Server error ({e.status}): {e}. Retrying...")
            raise
    
    # Use tenacity for retries with exponential backoff
    return await retry_async(
        _fetch, 
        max_attempts=settings.ctgov.max_retries
    )


def upload_to_gcs(local_path: str, gcs_path: str, bucket_name: str = None):
    """
    Upload a file to Google Cloud Storage.

    Args:
        local_path (str): Path to the local file.
        gcs_path (str): Path in the bucket (e.g., 'runs/{timestamp}/file.csv').
        bucket_name (str, optional): GCS bucket name. If None, uses GCS_BUCKET env variable.
    """
    if bucket_name is None:
        bucket_name = os.environ.get("GCS_BUCKET")
        if not bucket_name:
            raise ValueError("GCS_BUCKET environment variable not set.")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    logger.info(f"Uploaded {local_path} to gs://{bucket_name}/{gcs_path}")


def download_from_gcs(gcs_path: str, local_path: str, bucket_name: str = None):
    """
    Download a file from Google Cloud Storage to a local path.

    Args:
        gcs_path (str): Path in the bucket (e.g., 'cache/drug_cache.sqlite').
        local_path (str): Local file path to save the downloaded file.
        bucket_name (str, optional): GCS bucket name. If None, uses GCS_BUCKET env variable.
    """
    if bucket_name is None:
        bucket_name = os.environ.get("GCS_BUCKET")
        if not bucket_name:
            raise ValueError("GCS_BUCKET environment variable not set.")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    if blob.exists():
        blob.download_to_filename(local_path)
        logger.info(f"Downloaded gs://{bucket_name}/{gcs_path} to {local_path}")
    else:
        logger.info(f"GCS file gs://{bucket_name}/{gcs_path} does not exist. Skipping download.") 