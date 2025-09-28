"""
Saem's Tunes AI System - Production Ready
AI-powered assistant for Saem's Tunes music education and streaming platform.
"""

__version__ = "2.0.0"
__author__ = "Saem's Tunes Development Team"
__description__ = "Production-ready AI assistant for music education and streaming"
__license__ = "Proprietary"

from .ai_system import SaemsTunesAISystem
from .supabase_integration import AdvancedSupabaseIntegration
from .security_system import AdvancedSecuritySystem
from .monitoring_system import ComprehensiveMonitor
from .utils import (
    json_serializer, 
    format_response, 
    get_timestamp, 
    validate_environment_variables,
    calculate_md5_hash,
    sanitize_filename,
    async_get_request,
    async_post_request,
    format_duration,
    get_file_size,
    create_directory_if_not_exists,
    setup_logging,
    truncate_text,
    is_valid_url,
    get_memory_usage_mb,
    retry_on_exception
)

__all__ = [
    "SaemsTunesAISystem",
    "AdvancedSupabaseIntegration", 
    "AdvancedSecuritySystem",
    "ComprehensiveMonitor",
    "json_serializer",
    "format_response", 
    "get_timestamp",
    "validate_environment_variables",
    "calculate_md5_hash",
    "sanitize_filename",
    "async_get_request",
    "async_post_request",
    "format_duration",
    "get_file_size",
    "create_directory_if_not_exists",
    "setup_logging",
    "truncate_text",
    "is_valid_url",
    "get_memory_usage_mb",
    "retry_on_exception"
]