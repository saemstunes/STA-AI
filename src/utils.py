import json
import logging
import os
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
import aiohttp
import asyncio
import functools
import time

def json_serializer(obj: Any) -> str:
    """
    JSON serializer for objects not serializable by default json code.
    
    Args:
        obj: Object to serialize
        
    Returns:
        Serialized string
        
    Raises:
        TypeError: If object type is not supported
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    elif isinstance(obj, set):
        return list(obj)
    else:
        raise TypeError(f"Type {type(obj)} not serializable")

def format_response(response: Dict) -> str:
    """
    Format response for consistent output.
    
    Args:
        response: Response dictionary to format
        
    Returns:
        Formatted JSON string
    """
    return json.dumps(response, default=json_serializer, indent=2)

def get_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    Returns:
        Current timestamp as ISO string
    """
    return datetime.now().isoformat()

def validate_environment_variables(required_vars: List[str]) -> bool:
    """
    Validate that required environment variables are set.
    
    Args:
        required_vars: List of required environment variable names
        
    Returns:
        True if all variables are set, False otherwise
    """
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logging.error(f"Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    return True

def calculate_md5_hash(text: str) -> str:
    """
    Calculate MD5 hash of text.
    
    Args:
        text: Text to hash
        
    Returns:
        MD5 hash string
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to remove potentially dangerous characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove directory traversal attempts
    filename = filename.replace('../', '').replace('..\\', '')
    
    # Remove dangerous characters
    dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    return filename

async def async_get_request(url: str, headers: Optional[Dict] = None, timeout: int = 30) -> Dict[str, Any]:
    """
    Make asynchronous GET request.
    
    Args:
        url: URL to request
        headers: Optional headers
        timeout: Request timeout in seconds
        
    Returns:
        Response dictionary
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=timeout) as response:
                return {
                    'status': response.status,
                    'headers': dict(response.headers),
                    'content': await response.text(),
                    'url': str(response.url)
                }
    except Exception as e:
        return {
            'status': 0,
            'error': str(e),
            'content': '',
            'url': url
        }

async def async_post_request(url: str, data: Any, headers: Optional[Dict] = None, timeout: int = 30) -> Dict[str, Any]:
    """
    Make asynchronous POST request.
    
    Args:
        url: URL to request
        data: Data to send
        headers: Optional headers
        timeout: Request timeout in seconds
        
    Returns:
        Response dictionary
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers, timeout=timeout) as response:
                return {
                    'status': response.status,
                    'headers': dict(response.headers),
                    'content': await response.text(),
                    'url': str(response.url)
                }
    except Exception as e:
        return {
            'status': 0,
            'error': str(e),
            'content': '',
            'url': url
        }

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def get_file_size(filepath: str) -> Optional[int]:
    """
    Get file size in bytes.
    
    Args:
        filepath: Path to file
        
    Returns:
        File size in bytes or None if file doesn't exist
    """
    try:
        return os.path.getsize(filepath)
    except OSError:
        return None

def create_directory_if_not_exists(directory: str):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def is_valid_url(url: str) -> bool:
    """
    Check if string is a valid URL.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid URL, False otherwise
    """
    try:
        from urllib.parse import urlparse
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def get_memory_usage_mb() -> float:
    """
    Get current process memory usage in MB.
    
    Returns:
        Memory usage in MB
    """
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def retry_on_exception(max_retries: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """
    Decorator for retrying function on exception.
    
    Args:
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
        exceptions: Exceptions to catch
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            return None
        return wrapper
    return decorator

def batch_process(items: List, batch_size: int = 10):
    """
    Process items in batches.
    
    Args:
        items: List of items to process
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def safe_get(dictionary: Dict, keys: List[str], default: Any = None) -> Any:
    """
    Safely get nested dictionary value.
    
    Args:
        dictionary: Dictionary to search
        keys: List of keys to traverse
        default: Default value if key not found
        
    Returns:
        Value or default
    """
    current = dictionary
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def format_bytes(size: float) -> str:
    """
    Format bytes to human readable string.
    
    Args:
        size: Size in bytes
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def generate_conversation_id() -> str:
    """
    Generate a unique conversation ID.
    
    Returns:
        Unique conversation ID string
    """
    import uuid
    return f"conv_{uuid.uuid4().hex[:16]}"

def validate_email(email: str) -> bool:
    """
    Validate email format.
    
    Args:
        email: Email to validate
        
    Returns:
        True if valid email, False otherwise
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def get_platform_info() -> Dict[str, Any]:
    """
    Get platform information.
    
    Returns:
        Platform information dictionary
    """
    import platform
    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version()
    }