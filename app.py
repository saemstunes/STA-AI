import os
import gradio as gr
import json
import time
import logging
import psutil
import asyncio
import threading
import requests
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import numpy as np
from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
import secrets

load_dotenv()

# Enhanced Configuration with comprehensive settings
class Config:
    # Supabase Configuration
    SUPABASE_URL = os.getenv("SUPABASE_URL", "https://your-project.supabase.co")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "your-anon-key")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "your-service-key")
    
    # Model Configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    MODEL_REPO = os.getenv("MODEL_REPO", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    MODEL_FILE = os.getenv("MODEL_FILE", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    
    # Server Configuration
    HF_SPACE = os.getenv("HF_SPACE", "saemstunes/STA-AI")
    PORT = int(os.getenv("PORT", "8000"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # AI Configuration
    MAX_RESPONSE_LENGTH = int(os.getenv("MAX_RESPONSE_LENGTH", "500"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_P = float(os.getenv("TOP_P", "0.9"))
    CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "2048"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
    
    # System Configuration
    ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "true").lower() == "true"
    ENABLE_SECURITY = os.getenv("ENABLE_SECURITY", "true").lower() == "true"
    ENABLE_RATE_LIMITING = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"
    
    # Security Configuration
    API_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "100"))
    MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))
    JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
    
    # Music Platform Specific Configuration
    MUSIC_GENRES = [
        "pop", "rock", "jazz", "classical", "hiphop", "electronic", 
        "r&b", "country", "reggae", "metal", "folk", "blues", "kpop", "latin"
    ]
    
    PLATFORM_FEATURES = [
        "music_streaming", "playlist_creation", "artist_profiles", 
        "music_courses", "live_streaming", "collaborative_playlists",
        "lyrics_display", "offline_listening", "high_quality_audio"
    ]

# Enhanced logging configuration
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('saems_ai_system.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("SaemsTunesAI")

# Enhanced Data Models
class UserRole(Enum):
    FREE_USER = "free_user"
    PREMIUM_USER = "premium_user"
    ARTIST = "artist"
    ADMIN = "admin"
    CONTENT_CREATOR = "content_creator"

class MusicContentType(Enum):
    SONG = "song"
    ALBUM = "album"
    PLAYLIST = "playlist"
    PODCAST = "podcast"
    COURSE = "course"
    LIVE_STREAM = "live_stream"

class ConversationState(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"

@dataclass
class UserProfile:
    user_id: str
    username: str
    email: str
    role: UserRole
    subscription_tier: str
    favorite_genres: List[str]
    created_at: datetime
    last_active: datetime
    preferences: Dict[str, Any]

@dataclass
class MusicContent:
    content_id: str
    title: str
    artist: str
    genre: str
    content_type: MusicContentType
    duration: int
    release_date: datetime
    popularity_score: float
    metadata: Dict[str, Any]

@dataclass
class ConversationContext:
    conversation_id: str
    user_id: str
    messages: List[Dict[str, str]]
    context_window: List[str]
    user_preferences: Dict[str, Any]
    conversation_topic: str
    created_at: datetime
    last_updated: datetime
    state: ConversationState

# Enhanced Security System Implementation
class AdvancedSecuritySystem:
    def __init__(self):
        self.suspicious_patterns = [
            r"(?i)(password|credit.card|social.security|ssn|bank.account)",
            r"(?i)(login.credentials|account.details)",
            r"(?i)(malicious|hack|exploit|vulnerability)",
            r"(?i)(bitcoin|crypto.currency|ransomware)",
            r"(?i)(phishing|scam|fraud)",
            r".*[<>].*",  # HTML/script tags
            r".*[{}].*",  # Template injection
        ]
        self.rate_limits = {}
        self.blocked_ips = set()
        self.suspicious_activities = []
        self.api_keys = {}
        
        logger.info("🔒 Advanced Security System initialized")

    def check_request(self, message: str, user_id: str, ip_address: str = "unknown") -> Dict[str, Any]:
        """Comprehensive security check for incoming requests"""
        security_result = {
            "is_suspicious": False,
            "risk_level": "low",
            "reasons": [],
            "action_taken": "none"
        }
        
        # Check rate limiting
        if not self._check_rate_limit(user_id, ip_address):
            security_result.update({
                "is_suspicious": True,
                "risk_level": "high",
                "reasons": ["Rate limit exceeded"],
                "action_taken": "block"
            })
            return security_result
        
        # Check for blocked IPs
        if ip_address in self.blocked_ips:
            security_result.update({
                "is_suspicious": True,
                "risk_level": "high",
                "reasons": ["IP address blocked"],
                "action_taken": "block"
            })
            return security_result
        
        # Check message content for suspicious patterns
        content_analysis = self._analyze_message_content(message)
        if content_analysis["suspicious"]:
            security_result["is_suspicious"] = True
            security_result["risk_level"] = content_analysis["risk_level"]
            security_result["reasons"].extend(content_analysis["reasons"])
            
            if content_analysis["risk_level"] == "high":
                security_result["action_taken"] = "block"
                self.blocked_ips.add(ip_address)
        
        # Log suspicious activity
        if security_result["is_suspicious"]:
            self._log_suspicious_activity(user_id, ip_address, message, security_result)
        
        return security_result

    def _check_rate_limit(self, user_id: str, ip_address: str) -> bool:
        """Enhanced rate limiting with user and IP tracking"""
        current_time = time.time()
        user_key = f"user_{user_id}"
        ip_key = f"ip_{ip_address}"
        
        # Clean old entries
        self._clean_rate_limits(current_time)
        
        # Check user rate limit
        if user_key not in self.rate_limits:
            self.rate_limits[user_key] = []
        
        user_requests = self.rate_limits[user_key]
        user_requests = [req_time for req_time in user_requests if current_time - req_time < 60]
        
        if len(user_requests) >= Config.MAX_REQUESTS_PER_MINUTE:
            return False
        
        user_requests.append(current_time)
        self.rate_limits[user_key] = user_requests
        
        # Check IP rate limit
        if ip_key not in self.rate_limits:
            self.rate_limits[ip_key] = []
        
        ip_requests = self.rate_limits[ip_key]
        ip_requests = [req_time for req_time in ip_requests if current_time - req_time < 60]
        
        if len(ip_requests) >= Config.MAX_REQUESTS_PER_MINUTE * 2:  # Higher limit for IP
            return False
        
        ip_requests.append(current_time)
        self.rate_limits[ip_key] = ip_requests
        
        return True

    def _analyze_message_content(self, message: str) -> Dict[str, Any]:
        """Advanced content analysis for security threats"""
        import re
        
        analysis = {
            "suspicious": False,
            "risk_level": "low",
            "reasons": []
        }
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, message):
                analysis["suspicious"] = True
                analysis["risk_level"] = "medium"
                analysis["reasons"].append(f"Matched suspicious pattern: {pattern}")
        
        # Check message length for potential flooding
        if len(message) > 1000:
            analysis["suspicious"] = True
            analysis["risk_level"] = "medium"
            analysis["reasons"].append("Message length exceeds safe limits")
        
        # Check for excessive special characters
        special_chars = len(re.findall(r'[^\w\s]', message))
        if special_chars > len(message) * 0.3:  # More than 30% special chars
            analysis["suspicious"] = True
            analysis["risk_level"] = "high"
            analysis["reasons"].append("Excessive special characters detected")
        
        return analysis

    def _clean_rate_limits(self, current_time: float):
        """Clean old rate limit entries"""
        for key in list(self.rate_limits.keys()):
            self.rate_limits[key] = [
                req_time for req_time in self.rate_limits[key] 
                if current_time - req_time < 60
            ]
            if not self.rate_limits[key]:
                del self.rate_limits[key]

    def _log_suspicious_activity(self, user_id: str, ip_address: str, message: str, result: Dict[str, Any]):
        """Log suspicious activities for monitoring"""
        activity = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "ip_address": ip_address,
            "message": message[:100],  # Truncate for logging
            "security_result": result
        }
        self.suspicious_activities.append(activity)
        logger.warning(f"Suspicious activity detected: {activity}")

    def generate_api_key(self, user_id: str, permissions: List[str]) -> str:
        """Generate API key for authenticated users"""
        api_key = secrets.token_urlsafe(32)
        self.api_keys[api_key] = {
            "user_id": user_id,
            "permissions": permissions,
            "created_at": datetime.now(),
            "last_used": None
        }
        return api_key

    def validate_api_key(self, api_key: str, required_permission: str = None) -> bool:
        """Validate API key and check permissions"""
        if api_key not in self.api_keys:
            return False
        
        key_data = self.api_keys[api_key]
        key_data["last_used"] = datetime.now()
        
        if required_permission and required_permission not in key_data["permissions"]:
            return False
        
        return True

# Enhanced Supabase Integration
class AdvancedSupabaseIntegration:
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.client = None
        self.is_connected = False
        self.connection_attempts = 0
        self.max_retries = 3
        
        self._initialize_client()
        
        # Cache for frequently accessed data
        self.user_cache = {}
        self.music_cache = {}
        self.cache_ttl = 300  # 5 minutes

    def _initialize_client(self):
        """Initialize Supabase client with retry logic"""
        try:
            from supabase import create_client
            self.client = create_client(self.supabase_url, self.supabase_key)
            
            # Test connection
            test_response = self.client.from_('users').select('count', count='exact').limit(1).execute()
            self.is_connected = True
            logger.info("✅ Supabase connection established successfully")
            
        except Exception as e:
            self.is_connected = False
            logger.error(f"❌ Supabase connection failed: {e}")
            
            if self.connection_attempts < self.max_retries:
                self.connection_attempts += 1
                logger.info(f"Retrying connection (attempt {self.connection_attempts})...")
                time.sleep(2)
                self._initialize_client()

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile with caching"""
        cache_key = f"user_{user_id}"
        
        # Check cache first
        if cache_key in self.user_cache:
            cached_data, timestamp = self.user_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
        
        try:
            response = self.client.from_('users').select('*').eq('user_id', user_id).execute()
            
            if response.data:
                user_data = response.data[0]
                profile = UserProfile(
                    user_id=user_data.get('user_id'),
                    username=user_data.get('username'),
                    email=user_data.get('email'),
                    role=UserRole(user_data.get('role', 'free_user')),
                    subscription_tier=user_data.get('subscription_tier', 'free'),
                    favorite_genres=user_data.get('favorite_genres', []),
                    created_at=datetime.fromisoformat(user_data.get('created_at')),
                    last_active=datetime.fromisoformat(user_data.get('last_active')),
                    preferences=user_data.get('preferences', {})
                )
                
                # Update cache
                self.user_cache[cache_key] = (profile, time.time())
                return profile
                
        except Exception as e:
            logger.error(f"Error fetching user profile for {user_id}: {e}")
            
        return None

    def get_music_content(self, content_id: str = None, genre: str = None, 
                         content_type: str = None, limit: int = 10) -> List[MusicContent]:
        """Get music content with filtering and caching"""
        cache_key = f"music_{content_id or genre or content_type or 'all'}_{limit}"
        
        # Check cache first
        if cache_key in self.music_cache:
            cached_data, timestamp = self.music_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
        
        try:
            query = self.client.from_('music_content').select('*')
            
            if content_id:
                query = query.eq('content_id', content_id)
            if genre:
                query = query.eq('genre', genre)
            if content_type:
                query = query.eq('content_type', content_type)
                
            query = query.limit(limit)
            response = query.execute()
            
            music_content = []
            for item in response.data:
                content = MusicContent(
                    content_id=item.get('content_id'),
                    title=item.get('title'),
                    artist=item.get('artist'),
                    genre=item.get('genre'),
                    content_type=MusicContentType(item.get('content_type')),
                    duration=item.get('duration', 0),
                    release_date=datetime.fromisoformat(item.get('release_date')),
                    popularity_score=item.get('popularity_score', 0.0),
                    metadata=item.get('metadata', {})
                )
                music_content.append(content)
            
            # Update cache
            self.music_cache[cache_key] = (music_content, time.time())
            return music_content
            
        except Exception as e:
            logger.error(f"Error fetching music content: {e}")
            
        return []

    def save_conversation(self, conversation: ConversationContext) -> bool:
        """Save conversation context to database"""
        try:
            conversation_data = {
                'conversation_id': conversation.conversation_id,
                'user_id': conversation.user_id,
                'messages': conversation.messages,
                'context_window': conversation.context_window,
                'user_preferences': conversation.user_preferences,
                'conversation_topic': conversation.conversation_topic,
                'created_at': conversation.created_at.isoformat(),
                'last_updated': conversation.last_updated.isoformat(),
                'state': conversation.state.value
            }
            
            response = self.client.from_('conversations').upsert(conversation_data).execute()
            return bool(response.data)
            
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return False

    def get_user_conversations(self, user_id: str, limit: int = 5) -> List[ConversationContext]:
        """Get user's recent conversations"""
        try:
            response = self.client.from_('conversations')\
                .select('*')\
                .eq('user_id', user_id)\
                .order('last_updated', desc=True)\
                .limit(limit)\
                .execute()
            
            conversations = []
            for item in response.data:
                conversation = ConversationContext(
                    conversation_id=item.get('conversation_id'),
                    user_id=item.get('user_id'),
                    messages=item.get('messages', []),
                    context_window=item.get('context_window', []),
                    user_preferences=item.get('user_preferences', {}),
                    conversation_topic=item.get('conversation_topic', 'general'),
                    created_at=datetime.fromisoformat(item.get('created_at')),
                    last_updated=datetime.fromisoformat(item.get('last_updated')),
                    state=ConversationState(item.get('state', 'active'))
                )
                conversations.append(conversation)
                
            return conversations
            
        except Exception as e:
            logger.error(f"Error fetching user conversations: {e}")
            return []

    def update_user_activity(self, user_id: str) -> bool:
        """Update user's last active timestamp"""
        try:
            response = self.client.from_('users')\
                .update({'last_active': datetime.now().isoformat()})\
                .eq('user_id', user_id)\
                .execute()
            return bool(response.data)
        except Exception as e:
            logger.error(f"Error updating user activity: {e}")
            return False

    def search_music_content(self, query: str, genre: str = None, 
                           content_type: str = None) -> List[MusicContent]:
        """Search music content with full-text search"""
        try:
            search_query = self.client.from_('music_content').select('*')
            
            # Full-text search on title and artist
            search_query = search_query.or_(f"title.ilike.%{query}%,artist.ilike.%{query}%")
            
            if genre:
                search_query = search_query.eq('genre', genre)
            if content_type:
                search_query = search_query.eq('content_type', content_type)
                
            response = search_query.limit(20).execute()
            
            music_content = []
            for item in response.data:
                content = MusicContent(
                    content_id=item.get('content_id'),
                    title=item.get('title'),
                    artist=item.get('artist'),
                    genre=item.get('genre'),
                    content_type=MusicContentType(item.get('content_type')),
                    duration=item.get('duration', 0),
                    release_date=datetime.fromisoformat(item.get('release_date')),
                    popularity_score=item.get('popularity_score', 0.0),
                    metadata=item.get('metadata', {})
                )
                music_content.append(content)
                
            return music_content
            
        except Exception as e:
            logger.error(f"Error searching music content: {e}")
            return []

# Enhanced Monitoring System
class ComprehensiveMonitor:
    def __init__(self, prometheus_port: int = 8001):
        self.inference_metrics = []
        self.system_metrics = []
        self.error_logs = []
        self.performance_data = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'peak_memory_usage': 0.0,
            'start_time': datetime.now()
        }
        self.alert_thresholds = {
            'response_time': 10.0,  # seconds
            'memory_usage': 85.0,   # percentage
            'error_rate': 5.0,      # percentage
            'cpu_usage': 90.0       # percentage
        }
        self.alerts = []
        
        logger.info("📊 Comprehensive Monitor initialized")

    def log_inference(self, user_id: str, input_text: str, response: str, 
                     processing_time: float, tokens_used: int, success: bool = True):
        """Log inference metrics with detailed tracking"""
        metric = {
            'timestamp': datetime.now(),
            'user_id': user_id,
            'input_length': len(input_text),
            'response_length': len(response),
            'processing_time': processing_time,
            'tokens_used': tokens_used,
            'success': success,
            'memory_usage': psutil.virtual_memory().percent,
            'cpu_usage': psutil.cpu_percent()
        }
        
        self.inference_metrics.append(metric)
        self.performance_data['total_requests'] += 1
        
        if success:
            self.performance_data['successful_requests'] += 1
        else:
            self.performance_data['failed_requests'] += 1
            
        # Update average response time
        total_time = self.performance_data['average_response_time'] * (self.performance_data['total_requests'] - 1)
        self.performance_data['average_response_time'] = (total_time + processing_time) / self.performance_data['total_requests']
        
        # Check for alerts
        self._check_alerts(metric)

    def log_error(self, error_type: str, error_message: str, user_id: str = None, 
                 context: Dict[str, Any] = None):
        """Log errors with context for debugging"""
        error_log = {
            'timestamp': datetime.now(),
            'error_type': error_type,
            'error_message': error_message,
            'user_id': user_id,
            'context': context or {},
            'system_state': {
                'memory_usage': psutil.virtual_memory().percent,
                'cpu_usage': psutil.cpu_percent(),
                'disk_usage': psutil.disk_usage('/').percent
            }
        }
        
        self.error_logs.append(error_log)
        logger.error(f"Error {error_type}: {error_message}")

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return {
            'timestamp': datetime.now(),
            'performance': self.performance_data,
            'current_usage': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_gb': psutil.virtual_memory().used / (1024**3),
                'disk_percent': psutil.disk_usage('/').percent,
                'disk_used_gb': psutil.disk_usage('/').used / (1024**3)
            },
            'process_metrics': {
                'thread_count': threading.active_count(),
                'active_inferences': len([m for m in self.inference_metrics 
                                        if (datetime.now() - m['timestamp']).seconds < 60])
            },
            'recent_errors': len([e for e in self.error_logs 
                                if (datetime.now() - e['timestamp']).seconds < 300])
        }

    def get_average_response_time(self) -> float:
        """Calculate average response time for recent requests"""
        recent_metrics = [m for m in self.inference_metrics 
                         if (datetime.now() - m['timestamp']).seconds < 300]  # Last 5 minutes
        
        if not recent_metrics:
            return 0.0
            
        return sum(m['processing_time'] for m in recent_metrics) / len(recent_metrics)

    def get_error_rate(self) -> float:
        """Calculate error rate as percentage"""
        total_requests = self.performance_data['total_requests']
        if total_requests == 0:
            return 0.0
            
        return (self.performance_data['failed_requests'] / total_requests) * 100

    def get_uptime(self) -> str:
        """Get system uptime in human-readable format"""
        uptime_seconds = (datetime.now() - self.performance_data['start_time']).total_seconds()
        
        if uptime_seconds < 60:
            return f"{int(uptime_seconds)} seconds"
        elif uptime_seconds < 3600:
            return f"{int(uptime_seconds // 60)} minutes"
        elif uptime_seconds < 86400:
            return f"{int(uptime_seconds // 3600)} hours"
        else:
            return f"{int(uptime_seconds // 86400)} days"

    def _check_alerts(self, metric: Dict[str, Any]):
        """Check metrics against alert thresholds"""
        # Response time alert
        if metric['processing_time'] > self.alert_thresholds['response_time']:
            self._trigger_alert('high_response_time', 
                               f"Response time {metric['processing_time']:.2f}s exceeded threshold")
        
        # Memory usage alert
        if metric['memory_usage'] > self.alert_thresholds['memory_usage']:
            self._trigger_alert('high_memory_usage',
                               f"Memory usage {metric['memory_usage']:.1f}% exceeded threshold")
        
        # CPU usage alert
        if metric['cpu_usage'] > self.alert_thresholds['cpu_usage']:
            self._trigger_alert('high_cpu_usage',
                               f"CPU usage {metric['cpu_usage']:.1f}% exceeded threshold")

    def _trigger_alert(self, alert_type: str, message: str):
        """Trigger and log alert"""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message,
            'severity': 'warning'
        }
        
        self.alerts.append(alert)
        logger.warning(f"ALERT: {alert_type} - {message}")

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        recent_metrics = [m for m in self.inference_metrics 
                         if (datetime.now() - m['timestamp']).seconds < 3600]  # Last hour
        
        return {
            'report_time': datetime.now(),
            'time_period': 'last_hour',
            'total_requests': len(recent_metrics),
            'success_rate': (len([m for m in recent_metrics if m['success']]) / len(recent_metrics) * 100) if recent_metrics else 100,
            'average_response_time': self.get_average_response_time(),
            'peak_memory_usage': max([m['memory_usage'] for m in recent_metrics]) if recent_metrics else 0,
            'peak_cpu_usage': max([m['cpu_usage'] for m in recent_metrics]) if recent_metrics else 0,
            'error_count': len([m for m in recent_metrics if not m['success']]),
            'active_alerts': len([a for a in self.alerts 
                                if (datetime.now() - a['timestamp']).seconds < 3600])
        }

# Enhanced AI System Implementation
class SaemsTunesAISystem:
    def __init__(self, supabase_integration: AdvancedSupabaseIntegration,
                 security_system: AdvancedSecuritySystem, 
                 monitor: ComprehensiveMonitor,
                 model_name: str = Config.MODEL_NAME,
                 model_repo: str = Config.MODEL_REPO,
                 model_file: str = Config.MODEL_FILE,
                 max_response_length: int = Config.MAX_RESPONSE_LENGTH,
                 temperature: float = Config.TEMPERATURE,
                 top_p: float = Config.TOP_P,
                 context_window: int = Config.CONTEXT_WINDOW):
        
        self.supabase = supabase_integration
        self.security = security_system
        self.monitor = monitor
        self.model_name = model_name
        self.model_repo = model_repo
        self.model_file = model_file
        self.max_response_length = max_response_length
        self.temperature = temperature
        self.top_p = top_p
        self.context_window = context_window
        
        self.model_loaded = False
        self.llm = None
        self.conversation_contexts = {}
        
        # Music platform knowledge base
        self.music_knowledge_base = self._initialize_music_knowledge_base()
        self.platform_features = self._initialize_platform_features()
        
        self._load_model()
        
        logger.info("🧠 Saem's Tunes AI System initialized")

    def _initialize_music_knowledge_base(self) -> Dict[str, Any]:
        """Initialize comprehensive music platform knowledge base"""
        return {
            "platform_overview": {
                "name": "Saem's Tunes",
                "description": "A comprehensive music streaming and education platform",
                "founded": 2024,
                "mission": "To make music accessible to everyone through streaming and education"
            },
            "features": {
                "streaming": ["High-quality audio streaming", "Offline listening", "Cross-platform sync"],
                "social": ["Playlist sharing", "Artist following", "Collaborative playlists"],
                "education": ["Music courses", "Tutorials", "Live masterclasses"],
                "creation": ["Music upload for artists", "Podcast hosting", "Live streaming"]
            },
            "subscription_tiers": {
                "free": {
                    "price": "$0/month",
                    "features": ["Basic streaming", "Limited skips", "Standard quality"]
                },
                "premium": {
                    "price": "$9.99/month",
                    "features": ["HD streaming", "Unlimited skips", "Offline mode", "No ads"]
                },
                "family": {
                    "price": "$14.99/month",
                    "features": ["6 accounts", "All premium features", "Parental controls"]
                },
                "student": {
                    "price": "$4.99/month",
                    "features": ["All premium features", "Student verification required"]
                }
            },
            "supported_genres": Config.MUSIC_GENRES,
            "supported_formats": ["MP3", "FLAC", "WAV", "AAC", "OGG"],
            "device_support": ["Web", "iOS", "Android", "Desktop", "Smart speakers"]
        }

    def _initialize_platform_features(self) -> Dict[str, Any]:
        """Initialize detailed platform features"""
        return {
            "music_management": {
                "playlists": ["Create", "Edit", "Share", "Collaborate", "Import/Export"],
                "library": ["Favorites", "Recently played", "Downloads", "History"],
                "discovery": ["Recommendations", "New releases", "Charts", "Mood-based"]
            },
            "artist_tools": {
                "upload": ["Music upload", "Metadata management", "Release scheduling"],
                "analytics": ["Listener stats", "Revenue tracking", "Geographic data"],
                "promotion": ["Artist profile", "Social integration", "Fan engagement"]
            },
            "learning_resources": {
                "courses": ["Beginner to advanced", "Genre-specific", "Instrument training"],
                "tutorials": ["Video lessons", "Sheet music", "Practice exercises"],
                "community": ["Forums", "Live Q&A", "Peer feedback"]
            }
        }

    def _load_model(self):
        """Load the AI model with comprehensive error handling"""
        try:
            logger.info(f"🔄 Loading AI model: {self.model_name}")
            
            # Initialize the language model
            from llama_cpp import Llama
            
            self.llm = Llama(
                model_path=f"./models/{self.model_file}",
                n_ctx=self.context_window,
                n_batch=512,
                n_gpu_layers=-1,  # Use all GPU layers
                f16_kv=True,
                use_mlock=False,
                use_mmap=True,
                low_vram=False,
                rope_freq_base=10000,
                rope_freq_scale=1.0,
                verbose=False
            )
            
            # Test the model with a simple prompt
            test_response = self.llm(
                "Hello, are you working?",
                max_tokens=10,
                temperature=0.1,
                stop=["\n"],
                echo=False
            )
            
            self.model_loaded = True
            logger.info("✅ AI model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load AI model: {e}")
            self.model_loaded = False
            # Fallback to a simple rule-based system
            self._initialize_fallback_system()

    def _initialize_fallback_system(self):
        """Initialize fallback system when AI model fails to load"""
        logger.info("🔄 Initializing fallback rule-based system")
        self.fallback_responses = {
            "greeting": [
                "Hello! I'm Saem's Tunes AI assistant. How can I help you with music today?",
                "Hi there! Welcome to Saem's Tunes. What would you like to know about our platform?",
                "Greetings! I'm here to help you with all things music on Saem's Tunes."
            ],
            "help": [
                "I can help you with: creating playlists, finding music, learning about features, troubleshooting, and more!",
                "I'm here to assist with music streaming, courses, artist tools, and platform features. What do you need help with?",
                "As your music assistant, I can guide you through Saem's Tunes features, help discover music, and answer your questions."
            ],
            "features": [
                f"Saem's Tunes offers: {', '.join(Config.PLATFORM_FEATURES)}",
                f"Our platform includes music streaming, educational courses, artist tools, and social features like {', '.join(self.platform_features['music_management']['playlists'])}",
                f"You can enjoy {len(Config.MUSIC_GENRES)} music genres, create playlists, take courses, and connect with artists on our platform."
            ],
            "subscription": [
                f"We have {len(self.music_knowledge_base['subscription_tiers'])} subscription tiers: {', '.join(self.music_knowledge_base['subscription_tiers'].keys())}",
                "Our subscriptions range from free to premium plans. Would you like details about a specific tier?",
                "You can start with our free plan and upgrade anytime to access premium features like offline listening and HD audio."
            ],
            "default": [
                "I understand you're asking about Saem's Tunes. Could you provide more details so I can help you better?",
                "That's an interesting question about our music platform. Let me look into that for you.",
                "I'd be happy to help with that! Could you rephrase your question so I can provide the best assistance?"
            ]
        }

    def process_query(self, message: str, user_id: str, conversation_id: str = None) -> str:
        """Process user query with comprehensive context handling"""
        start_time = time.time()
        
        try:
            if not self.model_loaded:
                return self._fallback_response(message)
            
            # Get or create conversation context
            conversation = self._get_conversation_context(user_id, conversation_id)
            
            # Update conversation with new message
            conversation.messages.append({"role": "user", "content": message})
            conversation.context_window.append(f"User: {message}")
            
            # Maintain context window size
            if len(conversation.context_window) > 10:
                conversation.context_window = conversation.context_window[-10:]
            
            conversation.last_updated = datetime.now()
            
            # Generate context-aware prompt
            prompt = self._build_contextual_prompt(message, conversation)
            
            # Generate response using the AI model
            response = self._generate_ai_response(prompt, message)
            
            # Update conversation with AI response
            conversation.messages.append({"role": "assistant", "content": response})
            conversation.context_window.append(f"Assistant: {response}")
            
            # Save conversation to database
            self.supabase.save_conversation(conversation)
            
            processing_time = time.time() - start_time
            
            # Log inference metrics
            self.monitor.log_inference(
                user_id=user_id,
                input_text=message,
                response=response,
                processing_time=processing_time,
                tokens_used=len(response.split()),  # Approximate token count
                success=True
            )
            
            logger.info(f"AI response generated in {processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing query: {str(e)}"
            
            self.monitor.log_inference(
                user_id=user_id,
                input_text=message,
                response=error_msg,
                processing_time=processing_time,
                tokens_used=0,
                success=False
            )
            
            self.monitor.log_error(
                error_type="QueryProcessingError",
                error_message=error_msg,
                user_id=user_id,
                context={"message": message, "conversation_id": conversation_id}
            )
            
            logger.error(error_msg)
            return "I apologize, but I'm experiencing technical difficulties. Please try again in a moment."

    def _get_conversation_context(self, user_id: str, conversation_id: str = None) -> ConversationContext:
        """Get or create conversation context for user"""
        if conversation_id and conversation_id in self.conversation_contexts:
            return self.conversation_contexts[conversation_id]
        
        # Create new conversation
        new_conversation_id = conversation_id or f"conv_{uuid.uuid4().hex[:8]}"
        
        # Try to load user profile for personalized context
        user_profile = self.supabase.get_user_profile(user_id)
        
        conversation = ConversationContext(
            conversation_id=new_conversation_id,
            user_id=user_id,
            messages=[],
            context_window=[],
            user_preferences=user_profile.preferences if user_profile else {},
            conversation_topic="general",
            created_at=datetime.now(),
            last_updated=datetime.now(),
            state=ConversationState.ACTIVE
        )
        
        self.conversation_contexts[new_conversation_id] = conversation
        return conversation

    def _build_contextual_prompt(self, message: str, conversation: ConversationContext) -> str:
        """Build comprehensive contextual prompt for the AI"""
        
        # Base system prompt
        system_prompt = f"""You are Saem's Tunes AI Assistant, an expert on the Saem's Tunes music streaming and education platform.

Platform Knowledge:
- Name: {self.music_knowledge_base['platform_overview']['name']}
- Mission: {self.music_knowledge_base['platform_overview']['mission']}
- Features: {', '.join(Config.PLATFORM_FEATURES)}
- Genres: {', '.join(Config.MUSIC_GENRES)}
- Subscription Tiers: {', '.join(self.music_knowledge_base['subscription_tiers'].keys())}

User Context:
- User ID: {conversation.user_id}
- Conversation Topic: {conversation.conversation_topic}
- User Preferences: {json.dumps(conversation.user_preferences, indent=2)}

Your Role:
1. Provide accurate information about Saem's Tunes platform
2. Help users with music discovery, playlists, and features
3. Assist with technical issues and account questions
4. Guide users through music courses and learning resources
5. Be friendly, professional, and music-knowledgeable

Response Guidelines:
- Be specific about platform features
- Provide step-by-step guidance when needed
- Suggest relevant music or features based on user interests
- Keep responses under {self.max_response_length} characters
- Use markdown formatting for lists and important points

Current Conversation Context:
{chr(10).join(conversation.context_window[-5:])}

User's Current Message: "{message}"

Please provide a helpful, accurate response:"""

        return system_prompt

    def _generate_ai_response(self, prompt: str, user_message: str) -> str:
        """Generate AI response with proper formatting and safety checks"""
        try:
            response = self.llm(
                prompt,
                max_tokens=self.max_response_length,
                temperature=self.temperature,
                top_p=self.top_p,
                stop=["User:", "Assistant:", "\n\n"],
                echo=False,
                stream=False
            )
            
            generated_text = response['choices'][0]['text'].strip()
            
            # Clean and format the response
            cleaned_response = self._clean_ai_response(generated_text, user_message)
            
            # Safety check - ensure response is appropriate
            if self._is_response_safe(cleaned_response):
                return cleaned_response
            else:
                return "I apologize, but I cannot provide that information. Please ask about Saem's Tunes music platform features, music recommendations, or technical support."
                
        except Exception as e:
            logger.error(f"AI generation error: {e}")
            return self._fallback_response(user_message)

    def _clean_ai_response(self, response: str, user_message: str) -> str:
        """Clean and format AI response"""
        # Remove any duplicate greetings or signatures
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('Hello!', 'Hi!', 'Hey!', 'Assistant:', 'AI:')):
                cleaned_lines.append(line)
        
        cleaned_response = '\n'.join(cleaned_lines)
        
        # Ensure response isn't too short or repetitive
        if len(cleaned_response) < 10 or cleaned_response == user_message:
            return self._fallback_response(user_message)
        
        return cleaned_response

    def _is_response_safe(self, response: str) -> bool:
        """Check if AI response is safe and appropriate"""
        unsafe_patterns = [
            "I'm sorry, I cannot",
            "I cannot provide",
            "I'm not able to",
            "I'm an AI",
            "As an AI",
            "I don't have personal opinions",
            "I cannot answer that",
            "I cannot help with that"
        ]
        
        # Check for unsafe patterns
        for pattern in unsafe_patterns:
            if pattern.lower() in response.lower():
                return False
        
        # Check response length
        if len(response) < 5 or len(response) > self.max_response_length * 2:
            return False
            
        return True

    def _fallback_response(self, message: str) -> str:
        """Provide fallback response when AI is unavailable"""
        message_lower = message.lower()
        
        # Greeting detection
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return self.fallback_responses['greeting'][0]
        
        # Help request
        elif any(word in message_lower for word in ['help', 'what can you do', 'features']):
            return self.fallback_responses['help'][0]
        
        # Platform features
        elif any(word in message_lower for word in ['feature', 'what can', 'how to']):
            return self.fallback_responses['features'][0]
        
        # Subscription questions
        elif any(word in message_lower for word in ['price', 'subscription', 'plan', 'cost']):
            return self.fallback_responses['subscription'][0]
        
        # Default response
        else:
            return self.fallback_responses['default'][0]

    def is_healthy(self) -> bool:
        """Check if AI system is healthy and operational"""
        return self.model_loaded and self.llm is not None

    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics and health information"""
        return {
            "model_loaded": self.model_loaded,
            "model_name": self.model_name,
            "model_repo": self.model_repo,
            "context_window": self.context_window,
            "max_response_length": self.max_response_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "conversations_active": len(self.conversation_contexts),
            "knowledge_base_size": len(self.music_knowledge_base)
        }

    def search_music_recommendations(self, user_id: str, preferences: Dict[str, Any] = None) -> List[MusicContent]:
        """Get personalized music recommendations"""
        try:
            user_profile = self.supabase.get_user_profile(user_id)
            user_preferences = preferences or (user_profile.preferences if user_profile else {})
            
            # Extract favorite genres
            favorite_genres = user_preferences.get('favorite_genres', [])
            if not favorite_genres and user_profile:
                favorite_genres = user_profile.favorite_genres
            
            # Get recommendations from database
            recommendations = []
            for genre in favorite_genres[:3]:  # Top 3 genres
                genre_content = self.supabase.get_music_content(genre=genre, limit=5)
                recommendations.extend(genre_content)
            
            # Remove duplicates and sort by popularity
            unique_recommendations = {}
            for rec in recommendations:
                if rec.content_id not in unique_recommendations:
                    unique_recommendations[rec.content_id] = rec
            
            sorted_recommendations = sorted(
                unique_recommendations.values(),
                key=lambda x: x.popularity_score,
                reverse=True
            )
            
            return sorted_recommendations[:10]  # Return top 10
            
        except Exception as e:
            logger.error(f"Error getting music recommendations: {e}")
            return []

# Global system instances
supabase_integration = None
security_system = None
monitor = None
ai_system = None
systems_ready = False
initialization_complete = False
initialization_errors = []
initialization_start_time = None

def initialize_systems():
    """Initialize all system components with comprehensive error handling"""
    global supabase_integration, security_system, monitor, ai_system
    global systems_ready, initialization_complete, initialization_errors
    
    logger.info("🚀 Initializing Saem's Tunes AI System...")
    
    try:
        # Initialize Supabase integration
        supabase_integration = AdvancedSupabaseIntegration(
            Config.SUPABASE_URL, 
            Config.SUPABASE_ANON_KEY
        )
        
        if not supabase_integration.is_connected:
            raise Exception("Supabase connection failed")
        logger.info("✅ Supabase integration initialized")
        
        # Initialize security system
        security_system = AdvancedSecuritySystem()
        logger.info("✅ Security system initialized")
        
        # Initialize monitoring system
        monitor = ComprehensiveMonitor(prometheus_port=8001)
        logger.info("✅ Monitoring system initialized")
        
        # Initialize AI system
        ai_system = SaemsTunesAISystem(
            supabase_integration=supabase_integration,
            security_system=security_system,
            monitor=monitor,
            model_name=Config.MODEL_NAME,
            model_repo=Config.MODEL_REPO,
            model_file=Config.MODEL_FILE,
            max_response_length=Config.MAX_RESPONSE_LENGTH,
            temperature=Config.TEMPERATURE,
            top_p=Config.TOP_P,
            context_window=Config.CONTEXT_WINDOW
        )
        logger.info("✅ AI system initialized")
        
        # Verify system health
        if ai_system.is_healthy():
            systems_ready = True
            initialization_complete = True
            logger.info("🎉 All systems initialized successfully!")
            
            # Log system status
            system_status = get_system_status()
            logger.info(f"System Status: {system_status}")
        else:
            initialization_errors.append("AI system health check failed")
            initialization_complete = True
            logger.error("❌ AI system failed health check")
        
        return True
        
    except Exception as e:
        error_msg = f"System initialization failed: {str(e)}"
        logger.error(error_msg)
        initialization_errors.append(error_msg)
        initialization_complete = True
        return False

def initialize_systems_background():
    """Run system initialization in background thread"""
    global initialization_start_time
    initialization_start_time = time.time()
    
    thread = threading.Thread(target=initialize_systems)
    thread.daemon = True
    thread.start()

# Enhanced API Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    user_id: Optional[str] = Field("anonymous", description="User identifier")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    stream: Optional[bool] = Field(False, description="Enable streaming response")

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI response")
    processing_time: float = Field(..., description="Processing time in seconds")
    conversation_id: str = Field(..., description="Conversation identifier")
    timestamp: str = Field(..., description="Response timestamp")
    model_used: str = Field(..., description="Model identifier")

class HealthResponse(BaseModel):
    status: str = Field(..., description="System status")
    timestamp: str = Field(..., description="Check timestamp")
    systems: Dict[str, bool] = Field(..., description="System components status")
    resources: Dict[str, float] = Field(..., description="Resource usage")
    performance: Dict[str, Any] = Field(..., description="Performance metrics")

class MusicRecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    genres: Optional[List[str]] = Field(None, description="Preferred genres")
    limit: Optional[int] = Field(10, ge=1, le=50, description="Number of recommendations")

class MusicRecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]] = Field(..., description="Music recommendations")
    user_id: str = Field(..., description="User identifier")
    generated_at: str = Field(..., description="Generation timestamp")

# Security dependency for API
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate API token and get current user"""
    if security_system and security_system.validate_api_key(credentials.credentials):
        return credentials.credentials
    raise HTTPException(status_code=401, detail="Invalid API key")

# Create FastAPI app
fastapi_app = FastAPI(
    title="Saem's Tunes AI API",
    description="Advanced AI Assistant for Saem's Tunes Music Platform",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Root endpoint for health checks
@fastapi_app.get("/", response_class=HTMLResponse)
def root():
    """Root endpoint with system information"""
    status = get_system_status()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Saem's Tunes AI API</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .status {{ padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .healthy {{ background: #d4edda; color: #155724; }}
            .initializing {{ background: #fff3cd; color: #856404; }}
            .error {{ background: #f8d7da; color: #721c24; }}
        </style>
    </head>
    <body>
        <h1>🎵 Saem's Tunes AI API</h1>
        <p>Advanced AI Assistant for Music Streaming and Education Platform</p>
        
        <div class="status {status['status']}">
            <h3>System Status: {status['status'].upper()}</h3>
            <p>Timestamp: {status['timestamp']}</p>
        </div>
        
        <h3>Available Endpoints:</h3>
        <ul>
            <li><a href="/api/chat">POST /api/chat</a> - Chat with AI Assistant</li>
            <li><a href="/api/health">GET /api/health</a> - System Health Check</li>
            <li><a href="/api/models">GET /api/models</a> - Model Information</li>
            <li><a href="/api/stats">GET /api/stats</a> - System Statistics</li>
            <li><a href="/api/recommendations">POST /api/recommendations</a> - Music Recommendations</li>
            <li><a href="/api/docs">API Documentation</a> - Interactive API Docs</li>
        </ul>
        
        <p><strong>Version:</strong> 2.0.0 | <strong>Environment:</strong> Hugging Face Spaces</p>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@fastapi_app.get("/api/health", response_model=HealthResponse)
def api_health():
    """Comprehensive health check endpoint"""
    try:
        status_data = get_system_status()
        return status_data
    except Exception as e:
        logger.error(f"Health endpoint error: {e}")
        return JSONResponse(
            content={"status": "error", "error": str(e)},
            status_code=500
        )

@fastapi_app.get("/api/models")
def api_models():
    """Get model information and capabilities"""
    models_info = {
        "available_models": ["TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"],
        "current_model": Config.MODEL_NAME,
        "model_repo": Config.MODEL_REPO,
        "model_file": Config.MODEL_FILE,
        "quantization": "Q4_K_M",
        "context_length": Config.CONTEXT_WINDOW,
        "parameters": "1.1B",
        "max_response_length": Config.MAX_RESPONSE_LENGTH,
        "temperature": Config.TEMPERATURE,
        "top_p": Config.TOP_P,
        "capabilities": [
            "music_recommendations",
            "platform_guidance", 
            "technical_support",
            "music_education",
            "conversational_ai"
        ]
    }
    
    if ai_system and systems_ready:
        try:
            model_stats = ai_system.get_model_stats()
            models_info.update(model_stats)
        except Exception as e:
            logger.warning(f"Could not get model stats: {e}")
    
    return models_info

@fastapi_app.get("/api/stats")
def api_stats():
    """Get comprehensive system statistics"""
    if not monitor or not systems_ready:
        return {
            "status": "initializing" if not systems_ready else "degraded",
            "systems_ready": systems_ready,
            "timestamp": datetime.now().isoformat(),
            "initialization_errors": initialization_errors,
            "initialization_duration": time.time() - initialization_start_time if initialization_start_time else 0
        }
    
    stats_data = {
        "status": "healthy",
        "total_requests": len(monitor.inference_metrics),
        "average_response_time": monitor.get_average_response_time(),
        "error_rate": monitor.get_error_rate(),
        "uptime": monitor.get_uptime(),
        "system_health": get_system_status(),
        "performance_report": monitor.generate_performance_report(),
        "timestamp": datetime.now().isoformat()
    }
    return stats_data

@fastapi_app.post("/api/chat", response_model=ChatResponse)
def api_chat(request: ChatRequest, user: str = Depends(get_current_user)):
    """Chat endpoint with comprehensive processing"""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if not systems_ready:
            raise HTTPException(
                status_code=503, 
                detail="Systems are still initializing. Please try again in a moment."
            )
        
        # Security check
        security_result = security_system.check_request(request.message, request.user_id, "api")
        if security_result["is_suspicious"]:
            raise HTTPException(
                status_code=429, 
                detail=f"Request blocked for security reasons: {', '.join(security_result['reasons'])}"
            )
        
        # Process the query
        start_time = time.time()
        response = ai_system.process_query(request.message, request.user_id, request.conversation_id)
        processing_time = time.time() - start_time
        
        return ChatResponse(
            response=response,
            processing_time=processing_time,
            conversation_id=request.conversation_id or f"conv_{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            model_used=Config.MODEL_NAME
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@fastapi_app.post("/api/recommendations", response_model=MusicRecommendationResponse)
def api_recommendations(request: MusicRecommendationRequest, user: str = Depends(get_current_user)):
    """Get personalized music recommendations"""
    try:
        if not systems_ready or not ai_system:
            raise HTTPException(status_code=503, detail="AI system not ready")
        
        preferences = {}
        if request.genres:
            preferences['favorite_genres'] = request.genres
        
        recommendations = ai_system.search_music_recommendations(request.user_id, preferences)
        
        # Convert to serializable format
        serializable_recommendations = []
        for rec in recommendations[:request.limit]:
            serializable_recommendations.append({
                "content_id": rec.content_id,
                "title": rec.title,
                "artist": rec.artist,
                "genre": rec.genre,
                "content_type": rec.content_type.value,
                "duration": rec.duration,
                "release_date": rec.release_date.isoformat(),
                "popularity_score": rec.popularity_score,
                "metadata": rec.metadata
            })
        
        return MusicRecommendationResponse(
            recommendations=serializable_recommendations,
            user_id=request.user_id,
            generated_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        raise HTTPException(status_code=500, detail="Error generating recommendations")

# Enhanced system status function
def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status with detailed metrics"""
    if not initialization_complete:
        return {
            "status": "initializing", 
            "details": "Systems are starting up...",
            "timestamp": datetime.now().isoformat(),
            "initialization_started": initialization_start_time is not None,
            "duration_seconds": time.time() - initialization_start_time if initialization_start_time else 0,
            "estimated_time_remaining": "30-60 seconds for model download"
        }
    
    if not systems_ready:
        return {
            "status": "degraded",
            "details": "Systems initialized but not fully ready",
            "errors": initialization_errors,
            "timestamp": datetime.now().isoformat(),
            "systems_ready": systems_ready
        }
    
    try:
        system_metrics = monitor.get_system_metrics() if monitor else {}
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "systems": {
                "supabase": supabase_integration.is_connected if supabase_integration else False,
                "security": bool(security_system),
                "monitoring": bool(monitor),
                "ai_system": ai_system.is_healthy() if ai_system else False,
                "model_loaded": ai_system.model_loaded if ai_system else False,
                "api_ready": True
            },
            "resources": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                "disk_percent": psutil.disk_usage('/').percent,
                "disk_used_gb": round(psutil.disk_usage('/').used / (1024**3), 2)
            },
            "performance": {
                "total_requests": len(monitor.inference_metrics) if monitor else 0,
                "avg_response_time": monitor.get_average_response_time() if monitor else 0,
                "error_rate": monitor.get_error_rate() if monitor else 0,
                "uptime": monitor.get_uptime() if monitor else "0s",
                "active_conversations": len(ai_system.conversation_contexts) if ai_system else 0
            },
            "model_info": ai_system.get_model_stats() if ai_system else {}
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Enhanced chat interface function
def chat_interface(message: str, history: List[List[str]], request: gr.Request) -> str:
    """Gradio chat interface with enhanced functionality"""
    try:
        if not message.strip():
            return "🎵 Hello! I'm Saem's Tunes AI Assistant. How can I help you with music today?"
        
        if not systems_ready:
            return "🔄 Systems are still initializing. Please wait a moment and try again..."
        
        # Get client information
        client_host = getattr(request, "client", None)
        if client_host:
            user_ip = client_host.host
        else:
            user_ip = "unknown"
        user_id = f"gradio_user_{user_ip}"
        
        # Security check
        security_result = security_system.check_request(message, user_id, user_ip)
        if security_result["is_suspicious"]:
            logger.warning(f"Suspicious request blocked from {user_ip}: {message}")
            return "🚫 Your request has been blocked for security reasons. Please try a different question about Saem's Tunes."
        
        # Update user activity
        if supabase_integration:
            supabase_integration.update_user_activity(user_id)
        
        # Process the query
        start_time = time.time()
        response = ai_system.process_query(message, user_id)
        processing_time = time.time() - start_time
        
        formatted_response = f"{response}\n\n_Generated in {processing_time:.1f}s_"
        
        logger.info(f"Chat processed: {message[:50]}... -> {processing_time:.2f}s")
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Chat interface error: {e}")
        return "😔 I apologize, but I'm experiencing technical difficulties. Please try again later."

# Enhanced Gradio interface creation
def create_gradio_interface():
    """Create comprehensive Gradio interface with enhanced features"""
    
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1000px;
        margin: 0 auto;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    .main-container {
        background: white;
        border-radius: 15px;
        margin: 20px;
        padding: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
        background-size: 20px 20px;
        animation: float 20s infinite linear;
    }
    @keyframes float {
        0% { transform: translate(0, 0) rotate(0deg); }
        100% { transform: translate(-20px, -20px) rotate(360deg); }
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .status-healthy { 
        background-color: #4CAF50;
        box-shadow: 0 0 10px #4CAF50;
    }
    .status-warning { 
        background-color: #FF9800;
        box-shadow: 0 0 10px #FF9800;
    }
    .status-error { 
        background-color: #F44336;
        box-shadow: 0 0 10px #F44336;
    }
    .quick-actions {
        display: flex;
        gap: 12px;
        margin: 20px 0;
        flex-wrap: wrap;
        justify-content: center;
    }
    .quick-action-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 20px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .quick-action-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    .chat-container {
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        background: #fafafa;
    }
    .footer {
        text-align: center;
        color: #666;
        margin-top: 25px;
        padding-top: 20px;
        border-top: 1px solid #eee;
        font-size: 0.9em;
    }
    .feature-highlight {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
    }
    .gradio-button {
        transition: all 0.3s ease !important;
    }
    .gradio-button:hover {
        transform: scale(1.05);
    }
    """
    
    with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue",
            neutral_hue="slate"
        ),
        title="Saem's Tunes AI Assistant",
        css=custom_css
    ) as demo:
        
        with gr.Column(elem_classes="main-container"):
            gr.Markdown("""
            <div class="header">
                <h1 style="margin: 0; font-size: 2.5em; font-weight: 700;">🎵 Saem's Tunes AI Assistant</h1>
                <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.95;">
                    Powered by TinyLlama 1.1B • Your Intelligent Music Companion
                </p>
                <div style="margin-top: 15px; display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
                    <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 15px;">🎶 Music Streaming</span>
                    <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 15px;">📚 Music Education</span>
                    <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 15px;">👨‍🎤 Artist Tools</span>
                    <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 15px;">🔍 Smart Recommendations</span>
                </div>
            </div>
            """)
            
            # System Status Section
            with gr.Row():
                with gr.Column(scale=3):
                    status_display = gr.HTML(
                        value="<div class='status-indicator status-warning'></div><strong>Initializing AI Systems...</strong><br><small>Downloading model and setting up services</small>"
                    )
                with gr.Column(scale=1):
                    refresh_btn = gr.Button("🔄 Refresh Status", size="sm", variant="secondary")
            
            # Quick Actions Section
            gr.Markdown("### 💡 Quick Actions & Common Questions")
            
            quick_actions = [
                ("🎵 Create Playlist", "How do I create and share a playlist?"),
                ("⭐ Premium Features", "What are the benefits of premium subscription?"),
                ("👨‍🎤 Artist Upload", "How can I upload my music as an artist?"),
                ("📚 Music Courses", "Tell me about available music courses"),
                ("🔍 Recommendations", "How does the music recommendation system work?"),
                ("📱 Mobile App", "Is there a mobile app and how do I use it?"),
                ("🎧 Audio Quality", "What audio quality options are available?"),
                ("👥 Social Features", "Can I collaborate on playlists with friends?")
            ]
            
            quick_buttons = []
            with gr.Row():
                for i in range(0, len(quick_actions), 4):
                    with gr.Row():
                        for j in range(4):
                            if i + j < len(quick_actions):
                                icon_text, question = quick_actions[i + j]
                                btn = gr.Button(icon_text, size="sm", elem_classes="quick-action-btn")
                                quick_buttons.append((btn, question))
            
            # Feature Highlights
            gr.Markdown("""
            <div class="feature-highlight">
                <h4 style="margin: 0 0 10px 0;">🎯 What I Can Help You With:</h4>
                <ul style="margin: 0; columns: 2;">
                    <li>Music recommendations based on your taste</li>
                    <li>Playlist creation and management</li>
                    <li>Artist profile and upload guidance</li>
                    <li>Music course information and learning paths</li>
                    <li>Technical support and troubleshooting</li>
                    <li>Subscription and billing questions</li>
                    <li>Mobile app features and usage</li>
                    <li>Social features and collaboration</li>
                </ul>
            </div>
            """)
            
            # Main Chat Interface
            gr.Markdown("### 💬 Chat with Your Music Assistant")
            
            with gr.Column(elem_classes="chat-container"):
                chatbot = gr.Chatbot(
                    label="Saem's Tunes AI Conversation",
                    height=500,
                    placeholder="Ask me anything about music, features, or help with Saem's Tunes...",
                    show_label=False,
                    type="messages",
                    avatar_images=(
                        "https://i.imgur.com/7k12EPD.png",  # User avatar
                        "https://i.imgur.com/7k12EPD.png"   # Bot avatar (same for now)
                    )
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your question here... (Press Enter to send, Shift+Enter for new line)",
                        show_label=False,
                        scale=4,
                        container=False,
                        lines=2,
                        max_lines=5,
                        autofocus=True
                    )
                    submit_btn = gr.Button("Send 🚀", variant="primary", scale=1, size="lg")
            
            # Examples and Tools Section
            gr.Markdown("### 🎯 Example Questions")
            
            gr.Examples(
                examples=[
                    "How do I create a collaborative playlist with friends?",
                    "What's the difference between free and premium subscriptions?",
                    "How can I upload my original music to the platform?",
                    "What music courses are available for beginners?",
                    "How does the recommendation algorithm work?",
                    "Can I download music for offline listening?",
                    "What genres of music are available on Saem's Tunes?",
                    "How do I follow my favorite artists?",
                    "Is there a family plan available?",
                    "How do I reset my password or recover my account?"
                ],
                inputs=msg,
                label="Try these questions to get started:"
            )
            
            # Chat Management Tools
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Conversation", size="sm", variant="secondary")
                export_btn = gr.Button("💾 Export Chat History", size="sm", variant="secondary")
                suggestion_btn = gr.Button("💡 Get Music Suggestions", size="sm", variant="secondary")
            
            # Footer
            gr.Markdown("""
            <div class="footer">
                <p>
                    <strong>Powered by TinyLlama 1.1B Chat Model</strong> • 
                    <a href="https://www.saemstunes.com" target="_blank" style="color: #667eea;">Visit Saem's Tunes Platform</a> •
                    <a href="/api/docs" target="_blank" style="color: #667eea;">API Documentation</a>
                </p>
                <p style="font-size: 0.85em; opacity: 0.7;">
                    Model: Q4_K_M Quantization • Context: 2K Tokens • Response Time: ~2-5s • 
                    Supports: Music Streaming, Education, Artist Tools, Recommendations
                </p>
            </div>
            """)
        
        # Enhanced interaction functions
        def update_status():
            """Update system status display with detailed information"""
            status = get_system_status()
            status_text = status.get("status", "unknown")
            status_class = f"status-{status_text}" if status_text in ["healthy", "warning", "error"] else "status-warning"
            
            if status_text == "healthy":
                systems = status.get("systems", {})
                resources = status.get("resources", {})
                performance = status.get("performance", {})
                
                html = f"""
                <div class='status-indicator {status_class}'></div>
                <strong>System Status: Operational 🟢</strong><br>
                <small>
                    AI Model: {'✅ Loaded' if systems.get('model_loaded') else '❌ Loading'} | 
                    Database: {'✅ Connected' if systems.get('supabase') else '❌ Offline'} | 
                    CPU: {resources.get('cpu_percent', 0):.1f}% | 
                    Memory: {resources.get('memory_percent', 0):.1f}% |
                    Requests: {performance.get('total_requests', 0)} |
                    Avg Time: {performance.get('avg_response_time', 0):.1f}s
                </small>
                """
            elif status_text == "initializing":
                duration = status.get('duration_seconds', 0)
                html = f"""
                <div class='status-indicator {status_class}'></div>
                <strong>System Status: Initializing 🟡</strong><br>
                <small>Started {duration:.0f}s ago • Downloading AI model and setting up services...</small>
                """
            else:
                details = status.get('details', 'System experiencing issues')
                html = f"<div class='status-indicator {status_class}'></div><strong>System Status: Attention Needed 🔴</strong><br><small>{details}</small>"
            
            return html
        
        def user_message(user_message, chat_history):
            """Handle user message submission"""
            return "", chat_history + [{"role": "user", "content": user_message}]
        
        def bot_response(chat_history):
            """Generate bot response for the conversation"""
            if not chat_history:
                return chat_history
            
            user_message = chat_history[-1]["content"]
            bot_message = chat_interface(user_message, chat_history, gr.Request())
            
            return chat_history + [{"role": "assistant", "content": bot_message}]
        
        def clear_chat():
            """Clear the chat history"""
            return []
        
        def export_chat(chat_history):
            """Export chat history as text"""
            if not chat_history:
                return "No conversation to export"
            
            export_text = f"Saem's Tunes AI Conversation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            export_text += "=" * 50 + "\n\n"
            
            for msg in chat_history:
                role = "You" if msg["role"] == "user" else "AI Assistant"
                export_text += f"{role}: {msg['content']}\n\n"
            
            export_text += f"\nExported from Saem's Tunes AI Assistant\n"
            export_text += f"Platform: https://www.saemstunes.com\n"
            export_text += f"Model: {Config.MODEL_NAME}"
            
            return export_text
        
        def get_music_suggestions(chat_history):
            """Get personalized music suggestions"""
            if not systems_ready:
                return chat_history + [{"role": "assistant", "content": "Systems are still initializing. Please try again in a moment."}]
            
            try:
                # Generate a sample user ID for demo
                user_id = "demo_user_gradio"
                suggestions = ai_system.search_music_recommendations(user_id)
                
                if suggestions:
                    response = "🎵 **Based on popular trends, you might enjoy:**\n\n"
                    for i, song in enumerate(suggestions[:5], 1):
                        response += f"{i}. **{song.title}** by {song.artist} ({song.genre})\n"
                    
                    response += f"\n_Found {len(suggestions)} recommendations in our library_"
                else:
                    response = "🎵 **Music Suggestions:**\n\nExplore these popular genres on Saem's Tunes:\n\n"
                    for genre in Config.MUSIC_GENRES[:8]:
                        response += f"• {genre.title()}\n"
                    
                    response += "\nTry searching for specific artists or songs you enjoy!"
                
                return chat_history + [{"role": "assistant", "content": response}]
                
            except Exception as e:
                logger.error(f"Music suggestions error: {e}")
                return chat_history + [{"role": "assistant", "content": "I couldn't fetch music suggestions right now. Please try asking about specific genres or artists!"}]
        
        # Event handlers
        refresh_btn.click(update_status, outputs=status_display)
        
        msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, chatbot, chatbot
        )
        
        submit_btn.click(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, chatbot, chatbot
        )
        
        clear_btn.click(clear_chat, outputs=chatbot)
        export_btn.click(export_chat, chatbot, msg)
        suggestion_btn.click(get_music_suggestions, chatbot, chatbot)
        
        # Quick action buttons
        for btn, question_text in quick_buttons:
            btn.click(
                fn=lambda q=question_text: q,
                outputs=msg
            ).then(
                user_message, [msg, chatbot], [msg, chatbot]
            ).then(
                bot_response, chatbot, chatbot
            )
        
        # Initialize status on load
        demo.load(update_status, outputs=status_display)
    
    return demo

# Create and mount Gradio app to FastAPI
demo = create_gradio_interface()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

# Initialize systems in background
initialize_systems_background()

# Main execution
if __name__ == "__main__":
    logger.info("🎵 Starting Saem's Tunes AI System on Hugging Face Spaces...")
    
    import uvicorn
    
    # Start the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=Config.PORT,
        log_level=Config.LOG_LEVEL.lower(),
        access_log=True,
        timeout_keep_alive=60
    )
