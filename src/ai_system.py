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
    """
    
    def __init__(
        self, 
        supabase_integration: AdvancedSupabaseIntegration,
        security_system: AdvancedSecuritySystem,
        monitor: ComprehensiveMonitor,
        model_name: str = "microsoft/Phi-3.5-mini-instruct",
        model_repo: str = "bartowski/Phi-3.5-mini-instruct-GGUF",
        model_file: str = "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        max_response_length: int = 300,
        temperature: float = 0.7,
        top_p: float = 0.9,
        context_window: int = 4096
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
        """Setup logging for the AI system"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    def load_model(self):
        """Load the Phi-3.5-mini-instruct model"""
        try:
            self.logger.info(f"🔄 Loading {self.model_name} model...")
            
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
                
                self.logger.info(f"📥 Downloading model from {self.model_repo}")
                self.model_path = hf_hub_download(
                    repo_id=self.model_repo,
                    filename=self.model_file,
                    cache_dir=model_dir,
                    local_dir_use_symlinks=False
                )
                self.logger.info(f"✅ Model downloaded: {self.model_path}")
                
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
                n_threads=min(4, os.cpu_count() or 1),
                n_batch=512,
                verbose=False,
                use_mlock=False,
                use_mmap=True,
                low_vram=False
            )
            
            test_response = self.model.create_completion(
                "Test", 
                max_tokens=10,
                temperature=0.1,
                stop=["<|end|>", "</s>"]
            )
            
            if test_response and 'choices' in test_response and len(test_response['choices']) > 0:
                self.model_loaded = True
                self.logger.info("✅ Model loaded and tested successfully!")
                self.logger.info(f"📊 Model info: {self.model_path} (Hash: {self.model_hash})")
            else:
                self.logger.error("❌ Model test failed")
                self.model_loaded = False
            
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
            self.model_loaded = False
    
    def process_query(
        self, 
        query: str, 
        user_id: str, 
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Process user query and generate response with context from Supabase.
        
        Args:
            query: User's question
            user_id: Unique user identifier
            conversation_id: Optional conversation ID for context
            
        Returns:
            AI-generated response
        """
        if not self.model_loaded:
            self.logger.warning("Model not loaded, returning fallback response")
            return self.get_fallback_response(query)
        
        cache_key = f"{user_id}:{hash(query)}"
        if cache_key in self.response_cache:
            cached_response, timestamp = self.response_cache[cache_key]
            if time.time() - timestamp < 300:
                self.logger.info("Returning cached response")
                return cached_response
        
        try:
            start_time = time.time()
            
            context = self.supabase.get_music_context(query, user_id)
            
            prompt = self.build_enhanced_prompt(query, context, user_id, conversation_id)
            
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
    
    def build_enhanced_prompt(
        self, 
        query: str, 
        context: Dict[str, Any],
        user_id: str,
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Build comprehensive prompt with context from Saem's Tunes platform.
        """
        conversation_context = ""
        if conversation_id and conversation_id in self.conversation_history:
            recent_messages = self.conversation_history[conversation_id][-3:]
            for msg in recent_messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n"
        
        system_prompt = f"""<|system|>
You are the AI assistant for Saem's Tunes, a comprehensive music education and streaming platform.
PLATFORM OVERVIEW:
🎵 **Music Streaming**: {context.get('stats', {}).get('track_count', 0)}+ tracks, {context.get('stats', {}).get('artist_count', 0)}+ artists
📚 **Education**: Courses, lessons, quizzes, and learning paths
👥 **Community**: User profiles, favorites, social features
🎨 **Creator Tools**: Music upload, artist analytics, promotion tools
💎 **Premium**: Subscription-based premium features
PLATFORM STATISTICS:
- Total Tracks: {context.get('stats', {}).get('track_count', 0)}
- Total Artists: {context.get('stats', {}).get('artist_count', 0)} 
- Total Users: {context.get('stats', {}).get('user_count', 0)}
- Total Courses: {context.get('stats', {}).get('course_count', 0)}
- Active Playlists: {context.get('stats', {}).get('playlist_count', 0)}
CURRENT CONTEXT:
{context.get('summary', 'General platform information')}
POPULAR CONTENT:
{self.format_popular_content(context)}
USER CONTEXT:
{self.format_user_context(context.get('user_context', {}))}
CONVERSATION HISTORY:
{conversation_context if conversation_context else 'No recent conversation history'}
RESPONSE GUIDELINES:
1. Be passionate about music and education
2. Provide specific, actionable information about Saem's Tunes
3. Reference platform features when relevant
4. Keep responses concise (under {self.max_response_length} words)
5. Be encouraging and supportive
6. If unsure, guide users to relevant platform sections
7. Personalize responses when user context is available
8. Always maintain a professional, helpful tone
9. Focus on music education, streaming, and platform features
10. Avoid discussing unrelated topics
PLATFORM FEATURES TO MENTION:
- Music streaming and discovery
- Educational courses and learning paths
- Playlist creation and sharing
- Artist tools and music upload
- Community features and social interaction
- Premium subscription benefits
- Mobile app availability
- Music recommendations
- Learning progress tracking
ANSWER THE USER'S QUESTION BASED ON THE ABOVE CONTEXT:<|end|>
"""
        
        user_prompt = f"<|user|>\n{query}<|end|>\n<|assistant|>\n"
        
        return system_prompt + user_prompt
    
    def format_popular_content(self, context: Dict[str, Any]) -> str:
        """Format popular content for the prompt"""
        content_lines = []
        
        if context.get('tracks'):
            content_lines.append("🎵 Popular Tracks:")
            for track in context['tracks'][:3]:
                title = track.get('title', 'Unknown Track')
                artist = track.get('artist', 'Unknown Artist')
                genre = track.get('genre', 'Various')
                plays = track.get('plays', 0)
                content_lines.append(f"  - {title} by {artist} ({genre}) - {plays} plays")
        
        if context.get('artists'):
            content_lines.append("👨‍🎤 Popular Artists:")
            for artist in context['artists'][:3]:
                name = artist.get('name', 'Unknown Artist')
                genre = artist.get('genre', 'Various')
                followers = artist.get('followers', 0)
                verified = "✓" if artist.get('verified') else ""
                content_lines.append(f"  - {name} {verified} ({genre}) - {followers} followers")
        
        if context.get('courses'):
            content_lines.append("📚 Recent Courses:")
            for course in context['courses'][:2]:
                title = course.get('title', 'Unknown Course')
                instructor = course.get('instructor', 'Unknown Instructor')
                level = course.get('level', 'All Levels')
                students = course.get('students', 0)
                content_lines.append(f"  - {title} by {instructor} ({level}) - {students} students")
        
        return "\n".join(content_lines) if content_lines else "No specific content data available"
    
    def format_user_context(self, user_context: Dict[str, Any]) -> str:
        """Format user context for the prompt"""
        if not user_context:
            return "No specific user context available"
        
        user_lines = []
        
        if user_context.get('is_premium'):
            user_lines.append("• User has premium subscription")
        
        if user_context.get('favorite_genres'):
            genres = user_context['favorite_genres'][:3]
            user_lines.append(f"• Favorite genres: {', '.join(genres)}")
        
        if user_context.get('recent_activity'):
            activity = user_context['recent_activity'][:2]
            user_lines.append(f"• Recent activity: {', '.join(activity)}")
        
        if user_context.get('learning_progress'):
            progress = user_context['learning_progress']
            user_lines.append(f"• Learning progress: {progress.get('completed_lessons', 0)} lessons completed")
        
        return "\n".join(user_lines) if user_lines else "Basic user account"
    
    def clean_response(self, response: str) -> str:
        """Clean and format the AI response"""
        if not response:
            return "I apologize, but I couldn't generate a response. Please try again."
        
        response = response.strip()
        
        if response.startswith("I'm sorry") or response.startswith("I apologize"):
            if len(response) < 20:
                response = "I'd be happy to help you with that! Our platform offers comprehensive music education and streaming features."
        
        stop_phrases = [
            "<|end|>", "</s>", "###", "Human:", "Assistant:", 
            "<|endoftext|>", "<|assistant|>", "<|user|>"
        ]
        
        for phrase in stop_phrases:
            if phrase in response:
                response = response.split(phrase)[0].strip()
        
        sentences = response.split('. ')
        if len(sentences) > 1:
            response = '. '.join(sentences[:-1]) + '.' if not sentences[-1].endswith('.') else '. '.join(sentences)
        
        if not response.endswith(('.', '!', '?')):
            response += '.'
        
        response = response.replace('**', '').replace('__', '').replace('*', '')
        
        if len(response) > self.max_response_length:
            response = response[:self.max_response_length].rsplit(' ', 1)[0] + '...'
        
        return response
    
    def update_conversation_history(self, conversation_id: str, query: str, response: str):
        """Update conversation history for context"""
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
        
        self.conversation_history[conversation_id].extend([
            {"role": "user", "content": query, "timestamp": datetime.now()},
            {"role": "assistant", "content": response, "timestamp": datetime.now()}
        ])
        
        if len(self.conversation_history[conversation_id]) > 10:
            self.conversation_history[conversation_id] = self.conversation_history[conversation_id][-10:]
    
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
        """Record metrics for monitoring and analytics"""
        metrics = {
            'model_name': 'phi3.5-mini-Q4_K_M',
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
            'model_hash': self.model_hash
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
        """Get fallback response when model is unavailable"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['playlist', 'create', 'make']):
            return "You can create playlists by navigating to your Library and selecting 'Create New Playlist'. You can add tracks, customize the order, and share with friends. Premium users can create collaborative playlists."
        
        elif any(term in query_lower for term in ['course', 'learn', 'education', 'lesson']):
            return "We offer comprehensive music education courses for all skill levels. Visit the Education section to browse courses in music theory, instrument mastery, production techniques, and more. Each course includes video lessons, exercises, and progress tracking."
        
        elif any(term in query_lower for term in ['upload', 'artist', 'creator']):
            return "Artists can upload their music through the Creator Studio after account verification. You'll need to provide track files, metadata, and cover art. Once uploaded, your music will be available across our platform with analytics and revenue sharing."
        
        elif any(term in query_lower for term in ['premium', 'subscribe', 'payment']):
            return "Our premium subscription offers ad-free listening, offline downloads, high-quality audio, exclusive content, and advanced features. You can subscribe monthly or annually with cancel-anytime flexibility."
        
        elif any(term in query_lower for term in ['problem', 'issue', 'help', 'support']):
            return "I'd be happy to help troubleshoot any issues. Please describe the problem you're experiencing, or visit our Help Center for detailed guides and contact information for our support team."
        
        else:
            return "I'd love to help you with Saem's Tunes! Our platform combines music streaming with comprehensive education features. You can discover new music, learn instruments, connect with artists, and develop your musical skills—all in one place. What specific aspect would you like to know more about?"
    
    def get_error_response(self, error: Exception) -> str:
        """Get user-friendly error response"""
        error_str = str(error).lower()
        
        if "memory" in error_str or "gpu" in error_str:
            return "I'm experiencing high resource usage right now. Please try a simpler query or wait a moment before trying again."
        elif "timeout" in error_str or "slow" in error_str:
            return "The response is taking longer than expected. Please try again with a more specific question about Saem's Tunes features."
        else:
            return "I apologize, but I'm having technical difficulties right now. Please try again in a few moments, or contact support if the issue persists. Our team is constantly working to improve the AI assistant."
    
    def is_healthy(self) -> bool:
        """Check if AI system is healthy and ready"""
        return self.model_loaded and self.model is not None and self.supabase.is_connected()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for monitoring"""
        return {
            "model_loaded": self.model_loaded,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "model_hash": self.model_hash,
            "max_response_length": self.max_response_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "context_window": self.context_window,
            "supabase_connected": self.supabase.is_connected(),
            "conversations_active": len(self.conversation_history),
            "cache_size": len(self.response_cache)
        }
    
    def clear_cache(self, user_id: Optional[str] = None):
        """Clear response cache"""
        if user_id:
            keys_to_remove = [k for k in self.response_cache.keys() if k.startswith(f"{user_id}:")]
            for key in keys_to_remove:
                del self.response_cache[key]
        else:
            self.response_cache.clear()
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        if not self.model_loaded:
            return {"error": "Model not loaded"}
        
        return {
            "context_size": self.context_window,
            "parameters": "3.8B",
            "quantization": "Q4_K_M",
            "model_size_gb": round(os.path.getsize(self.model_path) / (1024**3), 2) if self.model_path else 0,
            "cache_hit_rate": len(self.response_cache) / (len(self.response_cache) + len(self.conversation_history)) if self.conversation_history else 0
        }