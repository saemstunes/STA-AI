import os
import time
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import json
import requests
import hashlib

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None
    print("Warning: llama-cpp-python not available. AI functionality will be limited.")

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None
    print("Warning: huggingface_hub not available. Model download will not work.")

from .supabase_integration import AdvancedSupabaseIntegration
from .security_system import AdvancedSecuritySystem
from .monitoring_system import ComprehensiveMonitor

class SaemsTunesAISystem:
    """
    Main AI system for Saem's Tunes music education and streaming platform.
    Handles user queries with context from the Supabase database.
    
    Saem's Tunes is a comprehensive music ecosystem featuring:
    - High-quality music streaming with advanced audio processing
    - Structured music education with courses, lessons, and learning paths
    - Social features for musicians and music lovers
    - Creator tools for artists to upload and promote their music
    - Premium subscription with enhanced features
    - Mobile and desktop applications
    - Community-driven content and collaborations
    """

    def __init__(
        self,
        supabase_integration: AdvancedSupabaseIntegration,
        security_system: AdvancedSecuritySystem,
        monitor: ComprehensiveMonitor,
        model_name: str = "TinyLlama-1.1B-Chat",
        model_repo: str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        model_file: str = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        max_response_length: int = 150,
        temperature: float = 0.6,
        top_p: float = 0.85,
        context_window: int = 2048
    ):
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
        
        self.model = None
        self.model_loaded = False
        self.model_path = None
        self.model_hash = None
        
        self.conversation_history = {}
        self.response_cache = {}
        
        self.setup_logging()
        self.load_model()

    def setup_logging(self):
        """Setup comprehensive logging for the AI system"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def load_model(self):
        """Load the optimized AI model with enhanced error handling and performance tuning"""
        try:
            self.logger.info(f"🔄 Loading {self.model_name} model with optimized configuration...")
            
            model_dir = "./models"
            os.makedirs(model_dir, exist_ok=True)
            
            local_path = os.path.join(model_dir, self.model_file)
            
            if os.path.exists(local_path):
                self.model_path = local_path
                self.logger.info(f"✅ Found local model: {local_path}")
                
                with open(local_path, 'rb') as f:
                    file_hash = hashlib.md5()
                    while chunk := f.read(8192):
                        file_hash.update(chunk)
                    self.model_hash = file_hash.hexdigest()
                    
            else:
                if hf_hub_download is None:
                    self.logger.error("❌ huggingface_hub not available for model download")
                    return
                
                self.logger.info(f"📥 Downloading optimized model from {self.model_repo}")
                self.model_path = hf_hub_download(
                    repo_id=self.model_repo,
                    filename=self.model_file,
                    cache_dir=model_dir,
                    local_dir_use_symlinks=False
                )
                self.logger.info(f"✅ Model downloaded and optimized: {self.model_path}")
                
                with open(self.model_path, 'rb') as f:
                    file_hash = hashlib.md5()
                    while chunk := f.read(8192):
                        file_hash.update(chunk)
                    self.model_hash = file_hash.hexdigest()
            
            if Llama is None:
                self.logger.error("❌ llama-cpp-python not available for model loading")
                return
            
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.context_window,
                n_threads=min(2, os.cpu_count() or 1),
                n_batch=256,
                n_gpu_layers=0,
                verbose=False,
                use_mlock=False,
                use_mmap=True,
                low_vram=True
            )
            
            test_response = self.model.create_completion(
                "Test response for Saem's Tunes AI system", 
                max_tokens=10,
                temperature=0.1,
                stop=["<|end|>", "</s>"]
            )
            
            if test_response and 'choices' in test_response and len(test_response['choices']) > 0:
                self.model_loaded = True
                self.logger.info("✅ Optimized model loaded and tested successfully!")
                self.logger.info(f"📊 Model info: {self.model_path} (Hash: {self.model_hash})")
                self.logger.info(f"⚡ Performance settings: 2 threads, 256 batch, CPU-only, low VRAM")
            else:
                self.logger.error("❌ Model test failed")
                self.model_loaded = False
            
        except Exception as e:
            self.logger.error(f"❌ Error loading optimized model: {e}")
            self.model_loaded = False

    def process_query(
        self, 
        query: str, 
        user_id: str, 
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Process user query and generate optimized response with context from Saem's Tunes platform.
        
        Args:
            query: User's question about music streaming, education, or platform features
            user_id: Unique user identifier for personalization
            conversation_id: Optional conversation ID for maintaining context
            
        Returns:
            AI-generated response tailored to Saem's Tunes ecosystem
        """
        if not self.model_loaded:
            self.logger.warning("Optimized model not loaded, returning fallback response")
            return self.get_fallback_response(query)
        
        cache_key = f"{user_id}:{hash(query)}"
        if cache_key in self.response_cache:
            cached_response, timestamp = self.response_cache[cache_key]
            if time.time() - timestamp < 300:
                self.logger.info("Returning cached response for performance")
                return cached_response
        
        try:
            start_time = time.time()
            
            context = self.supabase.get_music_context(query, user_id)
            
            prompt = self.build_optimized_prompt(query, context, user_id, conversation_id)
            
            response = self.model.create_completion(
                prompt,
                max_tokens=self.max_response_length,
                temperature=self.temperature,
                top_p=self.top_p,
                stop=["<|end|>", "</s>", "###", "Human:", "Assistant:", "<|endoftext|>"],
                echo=False,
                stream=False
            )
            
            processing_time = time.time() - start_time
            
            response_text = response['choices'][0]['text'].strip()
            
            response_text = self.clean_response(response_text)
            
            self.record_metrics(
                query=query,
                response=response_text,
                processing_time=processing_time,
                user_id=user_id,
                conversation_id=conversation_id,
                context_used=context,
                success=True
            )
            
            self.response_cache[cache_key] = (response_text, time.time())
            
            if conversation_id:
                self.update_conversation_history(conversation_id, query, response_text)
            
            self.logger.info(f"✅ Query processed in {processing_time:.2f}s: {query[:50]}...")
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"❌ Error processing query: {e}")
            
            self.record_metrics(
                query=query,
                response="",
                processing_time=0,
                user_id=user_id,
                conversation_id=conversation_id,
                error_message=str(e),
                success=False
            )
            
            return self.get_error_response(e)

    def build_optimized_prompt(
        self, 
        query: str, 
        context: Dict[str, Any],
        user_id: str,
        conversation_id: Optional[str] = None
    ) -> str:
        """Build optimized prompt for faster responses with Saem's Tunes context"""
        
        conversation_context = ""
        if conversation_id and conversation_id in self.conversation_history:
            recent_messages = self.conversation_history[conversation_id][-2:]
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n"
        
        # Enhanced system prompt with comprehensive Saem's Tunes context
        system_prompt = f"""<|system|>
You are Saem's Tunes AI assistant - the intelligent helper for a comprehensive music education and streaming platform.
SAEM'S TUNES PLATFORM OVERVIEW:
🎵 **Music Streaming**: High-quality audio streaming with advanced processing
📚 **Education**: Structured courses, interactive lessons, learning paths
👥 **Community**: Social features, collaborations, user profiles
🎨 **Creator Tools**: Music upload, analytics, promotion tools
💎 **Premium**: Enhanced features, offline listening, exclusive content
📱 **Mobile App**: Full-featured mobile experience
PLATFORM STATISTICS:
- Total Tracks: {context.get('stats', {}).get('track_count', 0)}
- Total Artists: {context.get('stats', {}).get('artist_count', 0)}
- Total Users: {context.get('stats', {}).get('user_count', 0)}
- Total Courses: {context.get('stats', {}).get('course_count', 0)}
- Active Playlists: {context.get('stats', {}).get('playlist_count', 0)}
CURRENT CONTEXT:
{context.get('summary', 'General platform information')}
POPULAR CONTENT:
{self.format_optimized_content(context)}
USER CONTEXT:
{self.format_user_context(context.get('user_context', {}))}
CONVERSATION HISTORY:
{conversation_context if conversation_context else 'No recent conversation history'}
RESPONSE GUIDELINES:
1. Be passionate about music education and streaming
2. Provide specific, actionable platform guidance
3. Keep responses concise (1-2 sentences maximum)
4. Focus on Saem's Tunes features and capabilities
5. Be encouraging and supportive of musical growth
6. Suggest specific platform sections or features
7. Personalize based on user context when available
8. Maintain professional, helpful tone always
PLATFORM FEATURES TO REFERENCE:
- Music streaming with high-quality audio
- Educational courses and learning paths  
- Playlist creation and sharing
- Artist tools and music upload
- Community features and collaborations
- Premium subscription benefits
- Mobile app functionality
- Music recommendations
- Learning progress tracking
ANSWER THE USER'S QUESTION BASED ON SAEM'S TUNES CONTEXT:<|end|>
"""
        
        user_prompt = f"<|user|>\n{query}<|end|>\n<|assistant|>\n"
        
        return system_prompt + user_prompt

    def format_optimized_content(self, context: Dict[str, Any]) -> str:
        """Format optimized content summary for faster processing"""
        content_lines = []
        
        if context.get('tracks'):
            content_lines.append("Popular Tracks:")
            for track in context['tracks'][:2]:
                title = track.get('title', 'Unknown Track')
                artist = track.get('artist', 'Unknown Artist')
                content_lines.append(f"- {title} by {artist}")
        
        if context.get('artists'):
            content_lines.append("Popular Artists:")
            for artist in context['artists'][:2]:
                name = artist.get('name', 'Unknown Artist')
                content_lines.append(f"- {name}")
        
        if context.get('courses'):
            content_lines.append("Recent Courses:")
            for course in context['courses'][:2]:
                title = course.get('title', 'Unknown Course')
                instructor = course.get('instructor', 'Unknown Instructor')
                content_lines.append(f"- {title} by {instructor}")
        
        return "\n".join(content_lines) if content_lines else "Popular content loading..."

    def format_user_context(self, user_context: Dict[str, Any]) -> str:
        """Format optimized user context"""
        if not user_context:
            return "New user exploring platform"
        
        user_lines = []
        
        if user_context.get('is_premium'):
            user_lines.append("• Premium subscriber")
        
        if user_context.get('favorite_genres'):
            genres = user_context['favorite_genres'][:2]
            user_lines.append(f"• Likes {', '.join(genres)}")
        
        if user_context.get('recent_activity'):
            activity = user_context['recent_activity'][:1]
            user_lines.append(f"• Recently: {activity[0]}")
        
        return "\n".join(user_lines) if user_lines else "Active platform user"

    def clean_response(self, response: str) -> str:
        """Clean and optimize the AI response for Saem's Tunes platform"""
        if not response:
            return "I'd love to help you explore Saem's Tunes! Our platform offers amazing music streaming and education features."
        
        response = response.strip()
        
        stop_phrases = [
            "<|end|>", "</s>", "###", "Human:", "Assistant:", 
            "<|endoftext|>", "<|assistant|>", "<|user|>"
        ]
        
        for phrase in stop_phrases:
            if phrase in response:
                response = response.split(phrase)[0].strip()
        
        sentences = response.split('. ')
        if len(sentences) > 2:
            response = '. '.join(sentences[:2]) + '.'
        
        if not response.endswith(('.', '!', '?')):
            response += '.'
        
        response = response.replace('**', '').replace('__', '').replace('*', '')
        
        if len(response) > self.max_response_length:
            response = response[:self.max_response_length].rsplit(' ', 1)[0] + '...'
        
        return response

    def update_conversation_history(self, conversation_id: str, query: str, response: str):
        """Update optimized conversation history"""
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
        
        self.conversation_history[conversation_id].extend([
            {"role": "user", "content": query, "timestamp": datetime.now()},
            {"role": "assistant", "content": response, "timestamp": datetime.now()}
        ])
        
        if len(self.conversation_history[conversation_id]) > 6:
            self.conversation_history[conversation_id] = self.conversation_history[conversation_id][-6:]

    def record_metrics(
        self,
        query: str,
        response: str,
        processing_time: float,
        user_id: str,
        conversation_id: Optional[str] = None,
        context_used: Optional[Dict] = None,
        error_message: Optional[str] = None,
        success: bool = True
    ):
        """Record comprehensive metrics for Saem's Tunes AI performance"""
        metrics = {
            'model_name': self.model_name,
            'processing_time_ms': processing_time * 1000,
            'input_tokens': len(query.split()),
            'output_tokens': len(response.split()) if response else 0,
            'total_tokens': len(query.split()) + (len(response.split()) if response else 0),
            'success': success,
            'user_id': user_id,
            'conversation_id': conversation_id,
            'timestamp': datetime.now(),
            'query_length': len(query),
            'response_length': len(response) if response else 0,
            'model_hash': self.model_hash,
            'platform': "saems_tunes"
        }
        
        if error_message:
            metrics['error_message'] = error_message
        
        if context_used:
            metrics['context_used'] = {
                'has_tracks': bool(context_used.get('tracks')),
                'has_artists': bool(context_used.get('artists')),
                'has_courses': bool(context_used.get('courses')),
                'has_user_context': bool(context_used.get('user_context')),
                'context_summary': context_used.get('summary', '')[:100]
            }
        
        self.monitor.record_inference(metrics)

    def get_fallback_response(self, query: str) -> str:
        """Get optimized fallback responses for Saem's Tunes platform"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['playlist', 'create', 'make']):
            return "Create playlists in your Library. Premium users can make collaborative playlists with friends."
        
        elif any(term in query_lower for term in ['course', 'learn', 'education', 'lesson']):
            return "Browse music courses in our Education section. We offer lessons for all skill levels with progress tracking."
        
        elif any(term in query_lower for term in ['upload', 'artist', 'creator']):
            return "Artists can upload music through Creator Studio. You'll need verified account and track files ready."
        
        elif any(term in query_lower for term in ['premium', 'subscribe', 'payment']):
            return "Premium offers ad-free listening, offline downloads, and exclusive content. Cancel anytime."
        
        elif any(term in query_lower for term in ['problem', 'issue', 'help', 'support']):
            return "Visit our Help Center for troubleshooting guides. Our support team is always ready to assist you."
        
        elif any(term in query_lower for term in ['stream', 'listen', 'music']):
            return "Stream millions of tracks in high quality. Discover new music through personalized recommendations."
        
        elif any(term in query_lower for term in ['social', 'friend', 'follow']):
            return "Connect with friends and artists. Share playlists and collaborate on music projects together."
        
        elif any(term in query_lower for term in ['mobile', 'app', 'download']):
            return "Get our mobile app for music on the go. Available on iOS and Android with full feature set."
        
        else:
            return "Welcome to Saem's Tunes! I help with music streaming, education, and platform features. What would you like to know?"

    def get_error_response(self, error: Exception) -> str:
        """Get user-friendly error responses"""
        error_str = str(error).lower()
        
        if "memory" in error_str or "gpu" in error_str:
            return "System is optimizing resources. Please try again in a moment."
        elif "timeout" in error_str or "slow" in error_str:
            return "Processing your request. Please try with a more specific question."
        else:
            return "Temporarily unavailable. Our team is working to restore full functionality. Please try again soon."

    def is_healthy(self) -> bool:
        """Check if optimized AI system is healthy"""
        return self.model_loaded and self.model is not None and self.supabase.is_connected()

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            "model_loaded": self.model_loaded,
            "model_name": self.model_name,
            "model_repo": self.model_repo,
            "model_file": self.model_file,
            "model_path": self.model_path,
            "model_hash": self.model_hash,
            "max_response_length": self.max_response_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "context_window": self.context_window,
            "supabase_connected": self.supabase.is_connected(),
            "conversations_active": len(self.conversation_history),
            "cache_size": len(self.response_cache),
            "optimized_performance": True,
            "cpu_threads": min(2, os.cpu_count() or 1),
            "low_vram_mode": True
        }

    def clear_cache(self, user_id: Optional[str] = None):
        """Clear response cache with optimization"""
        if user_id:
            keys_to_remove = [k for k in self.response_cache.keys() if k.startswith(f"{user_id}:")]
            for key in keys_to_remove:
                del self.response_cache[key]
            self.logger.info(f"Cleared cache for user {user_id}")
        else:
            self.response_cache.clear()
            self.logger.info("Cleared all response cache")

    def get_model_stats(self) -> Dict[str, Any]:
        """Get optimized model statistics"""
        if not self.model_loaded:
            return {"error": "Optimized model not loaded"}
        
        model_size = 0
        if self.model_path and os.path.exists(self.model_path):
            model_size = round(os.path.getsize(self.model_path) / (1024**3), 2)
        
        cache_hit_rate = 0
        total_requests = len(self.response_cache) + len(self.conversation_history)
        if total_requests > 0:
            cache_hit_rate = len(self.response_cache) / total_requests
        
        return {
            "model_name": self.model_name,
            "context_size": self.context_window,
            "parameters": "1.1B",
            "quantization": "Q4_K_M",
            "model_size_gb": model_size,
            "cache_hit_rate": round(cache_hit_rate, 2),
            "performance_optimized": True,
            "response_speed": "fast",
            "memory_usage": "low"
        }

    def switch_model(
        self,
        model_name: str,
        model_repo: str,
        model_file: str,
        max_response_length: int = 150,
        temperature: float = 0.6,
        top_p: float = 0.85,
        context_window: int = 2048
    ) -> bool:
        """Dynamically switch between different optimized models"""
        try:
            self.logger.info(f"🔄 Switching to model: {model_name}")
            
            self.model_name = model_name
            self.model_repo = model_repo
            self.model_file = model_file
            self.max_response_length = max_response_length
            self.temperature = temperature
            self.top_p = top_p
            self.context_window = context_window
            
            if self.model:
                del self.model
                self.model = None
            
            self.model_loaded = False
            self.load_model()
            
            if self.model_loaded:
                self.logger.info(f"✅ Successfully switched to {model_name}")
                return True
            else:
                self.logger.error(f"❌ Failed to switch to {model_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error switching models: {e}")
            return False

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available optimized models for Saem's Tunes"""
        return [
            {
                "name": "TinyLlama-1.1B-Chat",
                "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                "size_gb": 0.7,
                "speed": "fastest",
                "use_case": "General queries, fast responses"
            },
            {
                "name": "Phi-2",
                "repo": "TheBloke/phi-2-GGUF", 
                "file": "phi-2.Q4_K_M.gguf",
                "size_gb": 1.6,
                "speed": "balanced",
                "use_case": "Complex reasoning, education focus"
            },
            {
                "name": "Qwen-1.8B-Chat",
                "repo": "TheBloke/Qwen1.5-1.8B-Chat-GGUF",
                "file": "qwen1.5-1.8b-chat-q4_k_m.gguf",
                "size_gb": 1.1,
                "speed": "fast",
                "use_case": "Conversational, user interactions"
            }
        ]

    def optimize_performance(self, level: str = "balanced") -> Dict[str, Any]:
        """Apply performance optimization profiles"""
        optimizations = {
            "maximum_speed": {
                "max_response_length": 100,
                "temperature": 0.5,
                "n_threads": 1,
                "n_batch": 128
            },
            "balanced": {
                "max_response_length": 150,
                "temperature": 0.6,
                "n_threads": 2,
                "n_batch": 256
            },
            "quality": {
                "max_response_length": 200,
                "temperature": 0.7,
                "n_threads": 4,
                "n_batch": 512
            }
        }
        
        if level not in optimizations:
            level = "balanced"
        
        config = optimizations[level]
        self.max_response_length = config["max_response_length"]
        self.temperature = config["temperature"]
        
        if self.model_loaded and self.model:
            self.model.n_threads = config["n_threads"]
            self.model.n_batch = config["n_batch"]
        
        self.logger.info(f"🎯 Applied {level} performance optimization")
        
        return {
            "optimization_level": level,
            "config_applied": config,
            "current_performance": "enhanced"
        }

    def get_conversation_analytics(self, conversation_id: str) -> Dict[str, Any]:
        """Get analytics for specific conversation"""
        if conversation_id not in self.conversation_history:
            return {"error": "Conversation not found"}
        
        messages = self.conversation_history[conversation_id]
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
        
        return {
            "conversation_id": conversation_id,
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "first_message_time": messages[0]["timestamp"] if messages else None,
            "last_message_time": messages[-1]["timestamp"] if messages else None,
            "average_response_length": sum(len(msg["content"]) for msg in assistant_messages) / len(assistant_messages) if assistant_messages else 0,
            "common_topics": self.analyze_conversation_topics(messages)
        }

    def analyze_conversation_topics(self, messages: List[Dict]) -> List[str]:
        """Analyze conversation topics for insights"""
        topics = []
        content = " ".join([msg["content"] for msg in messages])
        
        topic_keywords = {
            "streaming": ["stream", "listen", "play", "music", "song", "track"],
            "education": ["learn", "course", "lesson", "education", "study", "practice"],
            "technical": ["problem", "issue", "error", "bug", "help", "support"],
            "account": ["premium", "subscribe", "payment", "account", "profile"],
            "social": ["friend", "follow", "share", "collaborate", "community"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content.lower() for keyword in keywords):
                topics.append(topic)
        
        return topics if topics else ["general_inquiry"]

    def backup_conversation_history(self, file_path: str) -> bool:
        """Backup conversation history to file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.conversation_history, f, indent=2, default=str)
            self.logger.info(f"✅ Conversation history backed up to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to backup conversation history: {e}")
            return False

    def restore_conversation_history(self, file_path: str) -> bool:
        """Restore conversation history from file"""
        try:
            with open(file_path, 'r') as f:
                self.conversation_history = json.load(f)
            self.logger.info(f"✅ Conversation history restored from {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to restore conversation history: {e}")
            return False

    def emergency_shutdown(self):
        """Emergency shutdown procedure for AI system"""
        self.logger.warning("🚨 Initiating emergency shutdown of AI system")
        
        try:
            if self.model:
                del self.model
                self.model = None
            
            self.model_loaded = False
            self.response_cache.clear()
            
            self.logger.info("✅ AI system emergency shutdown completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during emergency shutdown: {e}")

    def emergency_restart(self):
        """Emergency restart procedure for AI system"""
        self.logger.warning("🔄 Initiating emergency restart of AI system")
        
        self.emergency_shutdown()
        time.sleep(2)
        self.load_model()
        
        if self.model_loaded:
            self.logger.info("✅ AI system emergency restart completed successfully")
        else:
            self.logger.error("❌ AI system emergency restart failed")

# Additional utility functions for the AI system

def create_model_selector(
    supabase_integration: AdvancedSupabaseIntegration,
    security_system: AdvancedSecuritySystem,
    monitor: ComprehensiveMonitor,
    model_preference: str = "balanced"
) -> SaemsTunesAISystem:
    """Factory function to create AI system with preferred model configuration"""
    
    model_configs = {
        "fastest": {
            "model_name": "TinyLlama-1.1B-Chat",
            "model_repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "model_file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "max_response_length": 100,
            "temperature": 0.5,
            "context_window": 2048
        },
        "balanced": {
            "model_name": "TinyLlama-1.1B-Chat", 
            "model_repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "model_file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "max_response_length": 150,
            "temperature": 0.6,
            "context_window": 2048
        },
        "quality": {
            "model_name": "Phi-2",
            "model_repo": "TheBloke/phi-2-GGUF",
            "model_file": "phi-2.Q4_K_M.gguf", 
            "max_response_length": 200,
            "temperature": 0.7,
            "context_window": 2048
        },
        "conversational": {
            "model_name": "Qwen-1.8B-Chat",
            "model_repo": "TheBloke/Qwen1.5-1.8B-Chat-GGUF",
            "model_file": "qwen1.5-1.8b-chat-q4_k_m.gguf",
            "max_response_length": 250,
            "temperature": 0.7,
            "context_window": 4096
        }
    }
    
    config = model_configs.get(model_preference, model_configs["balanced"])
    
    ai_system = SaemsTunesAISystem(
        supabase_integration=supabase_integration,
        security_system=security_system,
        monitor=monitor,
        **config
    )
    
    return ai_system

def validate_ai_system_readiness(ai_system: SaemsTunesAISystem) -> Dict[str, Any]:
    """Comprehensive validation of AI system readiness for Saem's Tunes"""
    
    checks = {
        "model_loaded": ai_system.model_loaded,
        "supabase_connected": ai_system.supabase.is_connected(),
        "security_active": ai_system.security.is_active(),
        "monitoring_ready": ai_system.monitor.is_ready(),
        "model_file_exists": os.path.exists(ai_system.model_path) if ai_system.model_path else False,
        "sufficient_memory": check_system_memory(),
        "cache_clean": len(ai_system.response_cache) < 1000
    }
    
    all_passed = all(checks.values())
    
    return {
        "ready": all_passed,
        "checks": checks,
        "timestamp": datetime.now(),
        "system_info": ai_system.get_system_info(),
        "recommendations": generate_recommendations(checks)
    }

def check_system_memory() -> bool:
    """Check if system has sufficient memory for AI operations"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return memory.available > (512 * 1024 * 1024)  # 512MB minimum
    except ImportError:
        return True  # Assume sufficient if psutil not available

def generate_recommendations(checks: Dict[str, bool]) -> List[str]:
    """Generate recommendations based on system check results"""
    recommendations = []
    
    if not checks["model_loaded"]:
        recommendations.append("Restart AI system to reload model")
    
    if not checks["supabase_connected"]:
        recommendations.append("Check Supabase connection configuration")
    
    if not checks["sufficient_memory"]:
        recommendations.append("Increase system memory or optimize model")
    
    if not checks["cache_clean"]:
        recommendations.append("Clear response cache to free memory")
    
    return recommendations if recommendations else ["System optimized and ready"]

# Export the main class and utility functions
__all__ = [
    'SaemsTunesAISystem',
    'create_model_selector', 
    'validate_ai_system_readiness',
    'check_system_memory',
    'generate_recommendations'
]
