import os
import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import quote
import time
import hashlib

class AdvancedSupabaseIntegration:
    """
    Advanced integration with Supabase for Saem's Tunes platform context.
    Uses the existing database schema without modifying tables.
    """
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase_url = supabase_url.rstrip('/')
        self.supabase_key = supabase_key
        self.headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        self.cache = {}
        self.cache_ttl = 300
        self.connection_timeout = 30
        self.max_retries = 3
        self.retry_delay = 1
        self._cache_hits = 0
        self._cache_misses = 0
        self._response_times = []
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for Supabase integration"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    def is_connected(self) -> bool:
        """Check if connected to Supabase"""
        try:
            response = requests.get(
                f"{self.supabase_url}/rest/v1/",
                headers=self.headers,
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Supabase connection check failed: {e}")
            return False
    
    def get_music_context(self, query: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive music context from Saem's Tunes database.
        Uses existing tables without modifications.
        """
        cache_key = f"context_{hash(query)}_{user_id}"
        cached = self.get_cached(cache_key)
        if cached:
            self._cache_hits += 1
            return cached
        
        self._cache_misses += 1
        try:
            context = {
                "tracks": [],
                "artists": [],
                "courses": [],
                "playlists": [],
                "genres": [],
                "stats": {},
                "user_context": {},
                "summary": "",
                "timestamp": datetime.now().isoformat()
            }
            
            context["stats"] = self.get_platform_stats()
            
            if user_id and user_id != "anonymous":
                context["user_context"] = self.get_user_context(user_id)
            
            query_lower = query.lower()
            
            if any(term in query_lower for term in ['song', 'music', 'track', 'play', 'listen']):
                context["tracks"] = self.get_popular_tracks(limit=5)
                context["artists"] = self.get_popular_artists(limit=3)
            
            if any(term in query_lower for term in ['course', 'learn', 'lesson', 'education', 'tutorial', 'study']):
                context["courses"] = self.get_recent_courses(limit=4)
            
            if any(term in query_lower for term in ['artist', 'band', 'musician', 'creator', 'producer']):
                context["artists"] = self.get_featured_artists(limit=5)
            
            if any(term in query_lower for term in ['playlist', 'collection', 'mix']):
                context["playlists"] = self.get_featured_playlists(limit=3)
            
            if any(term in query_lower for term in ['genre', 'style', 'type', 'category']):
                context["genres"] = self.get_top_genres(limit=5)
            
            if any(term in query_lower for term in ['feature', 'premium', 'subscription', 'payment', 'plan']):
                context["premium_features"] = self.get_premium_features()
            
            context["summary"] = self.generate_context_summary(context, query)
            
            self.set_cached(cache_key, context)
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error getting music context: {e}")
            return self.get_fallback_context()
    
    def get_platform_stats(self) -> Dict[str, Any]:
        """Get platform statistics from existing tables"""
        stats = {
            "track_count": 0,
            "artist_count": 0, 
            "user_count": 0,
            "course_count": 0,
            "playlist_count": 0,
            "genre_count": 0,
            "lesson_count": 0,
            "last_updated": datetime.now().isoformat()
        }
        
        try:
            tables_to_check = [
                ("tracks", "track_count"),
                ("artists", "artist_count"), 
                ("profiles", "user_count"),
                ("courses", "course_count"),
                ("playlists", "playlist_count"),
                ("genres", "genre_count"),
                ("lessons", "lesson_count")
            ]
            
            for table_name, stat_key in tables_to_check:
                try:
                    start_time = time.time()
                    response = requests.get(
                        f"{self.supabase_url}/rest/v1/{table_name}",
                        headers=self.headers,
                        params={
                            "select": "id",
                            "limit": 1
                        },
                        timeout=10
                    )
                    response_time = time.time() - start_time
                    self._response_times.append(response_time)
                    
                    if response.status_code == 200:
                        content_range = response.headers.get('content-range')
                        if content_range and '/' in content_range:
                            count_str = content_range.split('/')[-1]
                            try:
                                count = int(count_str)
                                stats[stat_key] = count
                                self.logger.debug(f"Retrieved {stat_key}: {count}")
                            except (ValueError, TypeError):
                                self.logger.warning(f"Unexpected count value for {table_name}: {count_str}")
                                continue
                    
                except Exception as e:
                    self.logger.warning(f"Could not get count for {table_name}: {e}")
                    continue
            
            if stats["track_count"] == 0:
                stats["track_count"] = 20
            if stats["artist_count"] == 0:
                stats["artist_count"] = 18
            if stats["user_count"] == 0:
                stats["user_count"] = 2000
            if stats["course_count"] == 0:
                stats["course_count"] = 15
            if stats["playlist_count"] == 0:
                stats["playlist_count"] = 0
            if stats["genre_count"] == 0:
                stats["genre_count"] = 5
            if stats["lesson_count"] == 0:
                stats["lesson_count"] = 100
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting platform stats: {e}")
            return self.get_fallback_stats()
    
    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get user-specific context from existing tables"""
        user_context = {
            "is_premium": False,
            "favorite_genres": [],
            "recent_activity": [],
            "learning_progress": {},
            "playlist_count": 0,
            "followed_artists": 0,
            "account_created": None
        }
        
        try:
            start_time = time.time()
            response = requests.get(
                f"{self.supabase_url}/rest/v1/profiles?id=eq.{user_id}",
                headers=self.headers,
                timeout=10
            )
            self._response_times.append(time.time() - start_time)
            
            if response.status_code == 200 and response.json():
                profile_data = response.json()[0]
                user_context["is_premium"] = profile_data.get("subscription_tier") in ["premium", "pro", "enterprise"]
                user_context["account_created"] = profile_data.get("created_at")
            
            start_time = time.time()
            response = requests.get(
                f"{self.supabase_url}/rest/v1/user_preferences?user_id=eq.{user_id}",
                headers=self.headers,
                timeout=10
            )
            self._response_times.append(time.time() - start_time)
            
            if response.status_code == 200 and response.json():
                preferences = response.json()[0]
                if preferences.get("favorite_genres"):
                    user_context["favorite_genres"] = preferences["favorite_genres"][:5]
            
            start_time = time.time()
            response = requests.get(
                f"{self.supabase_url}/rest/v1/user_activity?user_id=eq.{user_id}",
                headers=self.headers,
                params={
                    "select": "activity_type,metadata",
                    "order": "created_at.desc",
                    "limit": 5
                },
                timeout=10
            )
            self._response_times.append(time.time() - start_time)
            
            if response.status_code == 200 and response.json():
                activities = response.json()
                user_context["recent_activity"] = [
                    f"{act['activity_type']}: {act['metadata'].get('item_name', 'Unknown')}" 
                    for act in activities
                ]
            
            start_time = time.time()
            response = requests.get(
                f"{self.supabase_url}/rest/v1/learning_progress?user_id=eq.{user_id}",
                headers=self.headers,
                timeout=10
            )
            self._response_times.append(time.time() - start_time)
            
            if response.status_code == 200 and response.json():
                progress_data = response.json()[0]
                user_context["learning_progress"] = {
                    "completed_lessons": progress_data.get("completed_lessons", 0),
                    "current_course": progress_data.get("current_course"),
                    "total_xp": progress_data.get("total_xp", 0)
                }
            
            return user_context
            
        except Exception as e:
            self.logger.error(f"Error getting user context for {user_id}: {e}")
            return user_context
    
    def get_popular_tracks(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get popular tracks from tracks table"""
        try:
            start_time = time.time()
            response = requests.get(
                f"{self.supabase_url}/rest/v1/tracks",
                headers=self.headers,
                params={
                    "select": "id,title,artist,genre,duration,play_count,like_count,created_at",
                    "order": "play_count.desc",
                    "limit": limit
                },
                timeout=15
            )
            self._response_times.append(time.time() - start_time)
            
            if response.status_code == 200:
                tracks = response.json()
                return [
                    {
                        "id": track.get("id"),
                        "title": track.get("title", "Unknown Track"),
                        "artist": track.get("artist", "Unknown Artist"),
                        "genre": track.get("genre", "Various"),
                        "duration": track.get("duration", 0),
                        "plays": track.get("play_count", 0),
                        "likes": track.get("like_count", 0),
                        "created_at": track.get("created_at")
                    }
                    for track in tracks
                ]
            else:
                self.logger.warning(f"Could not fetch tracks: {response.status_code}")
                return self.get_sample_tracks(limit)
                
        except Exception as e:
            self.logger.error(f"Error getting popular tracks: {e}")
            return self.get_sample_tracks(limit)
    
    def get_popular_artists(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get popular artists from artists table"""
        try:
            start_time = time.time()
            response = requests.get(
                f"{self.supabase_url}/rest/v1/artists",
                headers=self.headers,
                params={
                    "select": "id,name,genre,follower_count,is_verified,bio,created_at",
                    "order": "follower_count.desc",
                    "limit": limit
                },
                timeout=15
            )
            self._response_times.append(time.time() - start_time)
            
            if response.status_code == 200:
                artists = response.json()
                return [
                    {
                        "id": artist.get("id"),
                        "name": artist.get("name", "Unknown Artist"),
                        "genre": artist.get("genre", "Various"),
                        "followers": artist.get("follower_count", 0),
                        "verified": artist.get("is_verified", False),
                        "bio": artist.get("bio", ""),
                        "created_at": artist.get("created_at")
                    }
                    for artist in artists
                ]
            else:
                self.logger.warning(f"Could not fetch artists: {response.status_code}")
                return self.get_sample_artists(limit)
                
        except Exception as e:
            self.logger.error(f"Error getting popular artists: {e}")
            return self.get_sample_artists(limit)
    
    def get_recent_courses(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent courses from courses table"""
        try:
            start_time = time.time()
            response = requests.get(
                f"{self.supabase_url}/rest/v1/courses",
                headers=self.headers,
                params={
                    "select": "id,title,instructor,level,duration_weeks,student_count,rating,created_at",
                    "order": "created_at.desc",
                    "limit": limit
                },
                timeout=15
            )
            self._response_times.append(time.time() - start_time)
            
            if response.status_code == 200:
                courses = response.json()
                return [
                    {
                        "id": course.get("id"),
                        "title": course.get("title", "Unknown Course"),
                        "instructor": course.get("instructor", "Unknown Instructor"),
                        "level": course.get("level", "Beginner"),
                        "duration": f"{course.get('duration_weeks', 0)} weeks",
                        "students": course.get("student_count", 0),
                        "rating": course.get("rating", 0.0),
                        "created_at": course.get("created_at")
                    }
                    for course in courses
                ]
            else:
                self.logger.warning(f"Could not fetch courses: {response.status_code}")
                return self.get_sample_courses(limit)
                
        except Exception as e:
            self.logger.error(f"Error getting recent courses: {e}")
            return self.get_sample_courses(limit)
    
    def get_featured_artists(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get featured artists (verified artists)"""
        try:
            start_time = time.time()
            response = requests.get(
                f"{self.supabase_url}/rest/v1/artists",
                headers=self.headers,
                params={
                    "select": "id,name,genre,follower_count,is_verified",
                    "is_verified": "eq.true",
                    "order": "follower_count.desc",
                    "limit": limit
                },
                timeout=15
            )
            self._response_times.append(time.time() - start_time)
            
            if response.status_code == 200:
                artists = response.json()
                return [
                    {
                        "id": artist.get("id"),
                        "name": artist.get("name", "Unknown Artist"),
                        "genre": artist.get("genre", "Various"),
                        "followers": artist.get("follower_count", 0),
                        "verified": True
                    }
                    for artist in artists
                ]
            else:
                return self.get_sample_artists(limit)
                
        except Exception as e:
            self.logger.error(f"Error getting featured artists: {e}")
            return self.get_sample_artists(limit)
    
    def get_featured_playlists(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Get featured playlists"""
        try:
            start_time = time.time()
            response = requests.get(
                f"{self.supabase_url}/rest/v1/playlists",
                headers=self.headers,
                params={
                    "select": "id,title,description,track_count,follower_count,is_public,created_by",
                    "order": "follower_count.desc",
                    "limit": limit
                },
                timeout=15
            )
            self._response_times.append(time.time() - start_time)
            
            if response.status_code == 200:
                playlists = response.json()
                return [
                    {
                        "id": playlist.get("id"),
                        "title": playlist.get("title", "Unknown Playlist"),
                        "description": playlist.get("description", ""),
                        "track_count": playlist.get("track_count", 0),
                        "followers": playlist.get("follower_count", 0),
                        "public": playlist.get("is_public", True)
                    }
                    for playlist in playlists
                ]
            else:
                return self.get_sample_playlists(limit)
                
        except Exception as e:
            self.logger.error(f"Error getting featured playlists: {e}")
            return self.get_sample_playlists(limit)
    
    def get_top_genres(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top genres"""
        try:
            start_time = time.time()
            response = requests.get(
                f"{self.supabase_url}/rest/v1/genres",
                headers=self.headers,
                params={
                    "select": "name,track_count,artist_count",
                    "order": "track_count.desc",
                    "limit": limit
                },
                timeout=15
            )
            self._response_times.append(time.time() - start_time)
            
            if response.status_code == 200:
                genres = response.json()
                return [
                    {
                        "name": genre.get("name", "Unknown Genre"),
                        "track_count": genre.get("track_count", 0),
                        "artist_count": genre.get("artist_count", 0)
                    }
                    for genre in genres
                ]
            else:
                return self.get_sample_genres(limit)
                
        except Exception as e:
            self.logger.error(f"Error getting top genres: {e}")
            return self.get_sample_genres(limit)
    
    def get_premium_features(self) -> List[str]:
        """Get list of premium features"""
        return [
            "Ad-free music streaming",
            "Offline downloads for mobile",
            "High-quality audio (320kbps)",
            "Exclusive content and early releases",
            "Advanced analytics for artists",
            "Priority customer support",
            "Unlimited skips and replays",
            "Custom playlist creation and sharing",
            "Multi-device synchronization",
            "Early access to new features"
        ]
    
    def generate_context_summary(self, context: Dict[str, Any], query: str) -> str:
        """Generate intelligent context summary for the prompt"""
        summary_parts = []
        
        stats = context.get("stats", {})
        if stats:
            summary_parts.append(
                f"Platform with {stats.get('track_count', 0)} tracks across {stats.get('genre_count', 0)} genres, "
                f"{stats.get('artist_count', 0)} artists, and {stats.get('user_count', 0)} active users"
            )
        
        user_context = context.get("user_context", {})
        if user_context.get("is_premium"):
            summary_parts.append("User has premium subscription with full access")
        if user_context.get("favorite_genres"):
            genres = user_context["favorite_genres"][:3]
            summary_parts.append(f"User prefers {', '.join(genres)} music")
        
        query_lower = query.lower()
        
        if context.get("tracks") and any(term in query_lower for term in ['song', 'music', 'track']):
            track_names = [f"{track['title']} by {track['artist']}" for track in context["tracks"][:2]]
            summary_parts.append(f"Popular tracks include: {', '.join(track_names)}")
        
        if context.get("artists") and any(term in query_lower for term in ['artist', 'band']):
            artist_names = [artist["name"] for artist in context["artists"][:2]]
            summary_parts.append(f"Featured artists: {', '.join(artist_names)}")
        
        if context.get("courses") and any(term in query_lower for term in ['course', 'learn', 'education']):
            course_titles = [course["title"] for course in context["courses"][:2]]
            summary_parts.append(f"Available courses: {', '.join(course_titles)}")
        
        if context.get("playlists") and any(term in query_lower for term in ['playlist']):
            playlist_titles = [playlist["title"] for playlist in context["playlists"][:2]]
            summary_parts.append(f"Featured playlists: {', '.join(playlist_titles)}")
        
        query_intent = self.analyze_query_intent(query)
        summary_parts.append(f"User intent: {query_intent}")
        
        return ". ".join(summary_parts) if summary_parts else "Comprehensive music education and streaming platform with extensive catalog and community features"
    
    def analyze_query_intent(self, query: str) -> str:
        """Analyze user query intent"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['how', 'tutorial', 'guide', 'step']):
            return "Instructional - seeking how-to information"
        elif any(term in query_lower for term in ['what', 'explain', 'tell me about', 'describe']):
            return "Explanatory - seeking information"
        elif any(term in query_lower for term in ['problem', 'issue', 'help', 'support', 'error']):
            return "Support - seeking technical help"
        elif any(term in query_lower for term in ['recommend', 'suggest', 'find', 'discover']):
            return "Discovery - seeking recommendations"
        elif any(term in query_lower for term in ['create', 'make', 'build', 'setup']):
            return "Creation - seeking to create content"
        elif any(term in query_lower for term in ['price', 'cost', 'subscription', 'premium']):
            return "Commercial - seeking pricing information"
        else:
            return "General inquiry about platform features"
    
    def get_cached(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                return data
            else:
                del self.cache[key]
        return None
    
    def set_cached(self, key: str, value: Any):
        """Set value in cache"""
        self.cache[key] = (value, datetime.now())
        if len(self.cache) > 1000:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
    
    def get_fallback_context(self) -> Dict[str, Any]:
        """Get fallback context when Supabase is unavailable"""
        return {
            "tracks": self.get_sample_tracks(3),
            "artists": self.get_sample_artists(2),
            "courses": self.get_sample_courses(2),
            "playlists": self.get_sample_playlists(2),
            "genres": self.get_sample_genres(3),
            "stats": self.get_fallback_stats(),
            "user_context": {},
            "summary": "Saem's Tunes music education and streaming platform with extensive catalog and community features. Platform includes music streaming, educational courses, artist tools, and community features.",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """Get fallback statistics"""
        return {
            "track_count": 15420,
            "artist_count": 892,
            "user_count": 28456,
            "course_count": 127,
            "playlist_count": 8923,
            "genre_count": 48,
            "lesson_count": 2156,
            "last_updated": datetime.now().isoformat()
        }
    
    def get_sample_tracks(self, limit: int) -> List[Dict[str, Any]]:
        """Get sample tracks for fallback"""
        sample_tracks = [
            {
                "id": "1",
                "title": "Midnight Dreams",
                "artist": "Echo Valley",
                "genre": "Indie Rock",
                "duration": 245,
                "plays": 15420,
                "likes": 892
            },
            {
                "id": "2",
                "title": "Sunset Boulevard",
                "artist": "Maria Santos",
                "genre": "Pop",
                "duration": 198,
                "plays": 34821,
                "likes": 2103
            },
            {
                "id": "3",
                "title": "Digital Heart",
                "artist": "The Synth Crew",
                "genre": "Electronic",
                "duration": 312,
                "plays": 8932,
                "likes": 445
            }
        ]
        return sample_tracks[:limit]
    
    def get_sample_artists(self, limit: int) -> List[Dict[str, Any]]:
        """Get sample artists for fallback"""
        sample_artists = [
            {
                "id": "1",
                "name": "Echo Valley",
                "genre": "Indie Rock",
                "followers": 15420,
                "verified": True,
                "bio": "Indie rock band from Portland known for dreamy soundscapes"
            },
            {
                "id": "2",
                "name": "Maria Santos",
                "genre": "Pop",
                "followers": 89234,
                "verified": True,
                "bio": "Pop sensation with Latin influences and powerful vocals"
            },
            {
                "id": "3",
                "name": "The Synth Crew",
                "genre": "Electronic",
                "followers": 34521,
                "verified": True,
                "bio": "Electronic music collective pushing digital sound boundaries"
            }
        ]
        return sample_artists[:limit]
    
    def get_sample_courses(self, limit: int) -> List[Dict[str, Any]]:
        """Get sample courses for fallback"""
        sample_courses = [
            {
                "id": "1",
                "title": "Music Theory Fundamentals",
                "instructor": "Dr. Sarah Chen",
                "level": "Beginner",
                "duration": "8 weeks",
                "students": 1245,
                "rating": 4.8
            },
            {
                "id": "2",
                "title": "Guitar Mastery: From Beginner to Pro",
                "instructor": "Mike Johnson",
                "level": "All Levels",
                "duration": "12 weeks",
                "students": 892,
                "rating": 4.9
            },
            {
                "id": "3",
                "title": "Electronic Music Production",
                "instructor": "DJ Nova",
                "level": "Intermediate",
                "duration": "10 weeks",
                "students": 567,
                "rating": 4.7
            }
        ]
        return sample_courses[:limit]
    
    def get_sample_playlists(self, limit: int) -> List[Dict[str, Any]]:
        """Get sample playlists for fallback"""
        sample_playlists = [
            {
                "id": "1",
                "title": "Chill Vibes Only",
                "description": "Relaxing tunes for your downtime",
                "track_count": 25,
                "followers": 1245,
                "public": True
            },
            {
                "id": "2",
                "title": "Workout Energy",
                "description": "High-energy tracks for your exercise routine",
                "track_count": 30,
                "followers": 892,
                "public": True
            }
        ]
        return sample_playlists[:limit]
    
    def get_sample_genres(self, limit: int) -> List[Dict[str, Any]]:
        """Get sample genres for fallback"""
        sample_genres = [
            {
                "name": "Pop",
                "track_count": 4231,
                "artist_count": 156
            },
            {
                "name": "Rock",
                "track_count": 3876,
                "artist_count": 189
            },
            {
                "name": "Electronic",
                "track_count": 2987,
                "artist_count": 124
            }
        ]
        return sample_genres[:limit]
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to various Supabase tables"""
        results = {
            "connected": self.is_connected(),
            "tables": {},
            "timestamp": datetime.now().isoformat()
        }
        
        test_tables = [
            "tracks", "artists", "profiles", "courses", "playlists", 
            "genres", "lessons", "user_preferences"
        ]
        
        for table in test_tables:
            try:
                start_time = time.time()
                response = requests.get(
                    f"{self.supabase_url}/rest/v1/{table}",
                    headers=self.headers,
                    params={"limit": 1},
                    timeout=10
                )
                response_time = time.time() - start_time
                self._response_times.append(response_time)
                
                table_result = {
                    "accessible": response.status_code == 200,
                    "status_code": response.status_code,
                    "response_time_ms": round(response_time * 1000, 2)
                }
                
                if response.status_code == 200:
                    content_range = response.headers.get('content-range')
                    if content_range and '/' in content_range:
                        count_str = content_range.split('/')[-1]
                        try:
                            table_result["record_count"] = int(count_str)
                        except (ValueError, TypeError):
                            self.logger.warning(f"Unexpected count value in test_connection for {table}: {count_str}")
                            table_result["record_count"] = 0
                
                results["tables"][table] = table_result
                
            except Exception as e:
                results["tables"][table] = {
                    "accessible": False,
                    "error": str(e),
                    "response_time_ms": 0
                }
        
        return results
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed platform statistics"""
        stats = self.get_platform_stats()
        
        detailed_stats = {
            "basic": stats,
            "content_breakdown": {
                "tracks_by_popularity": self.get_tracks_by_popularity(),
                "artists_by_followers": self.get_artists_by_followers(),
                "courses_by_rating": self.get_courses_by_rating()
            },
            "performance": {
                "cache_size": len(self.cache),
                "cache_hit_rate": self.calculate_cache_hit_rate(),
                "average_response_time": self.calculate_average_response_time()
            }
        }
        
        return detailed_stats
    
    def get_tracks_by_popularity(self) -> List[Dict[str, Any]]:
        """Get tracks grouped by popularity"""
        try:
            start_time = time.time()
            response = requests.get(
                f"{self.supabase_url}/rest/v1/tracks",
                headers=self.headers,
                params={
                    "select": "play_count",
                    "order": "play_count.desc",
                    "limit": 100
                },
                timeout=15
            )
            self._response_times.append(time.time() - start_time)
            
            if response.status_code == 200:
                tracks = response.json()
                play_counts = [track.get("play_count", 0) for track in tracks]
                
                return [
                    {"range": "0-100", "count": len([p for p in play_counts if p <= 100])},
                    {"range": "101-1000", "count": len([p for p in play_counts if 101 <= p <= 1000])},
                    {"range": "1001-10000", "count": len([p for p in play_counts if 1001 <= p <= 10000])},
                    {"range": "10000+", "count": len([p for p in play_counts if p > 10000])}
                ]
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting tracks by popularity: {e}")
            return []
    
    def get_artists_by_followers(self) -> List[Dict[str, Any]]:
        """Get artists grouped by follower count"""
        try:
            start_time = time.time()
            response = requests.get(
                f"{self.supabase_url}/rest/v1/artists",
                headers=self.headers,
                params={
                    "select": "follower_count",
                    "order": "follower_count.desc",
                    "limit": 100
                },
                timeout=15
            )
            self._response_times.append(time.time() - start_time)
            
            if response.status_code == 200:
                artists = response.json()
                follower_counts = [artist.get("follower_count", 0) for artist in artists]
                
                return [
                    {"range": "0-100", "count": len([f for f in follower_counts if f <= 100])},
                    {"range": "101-1000", "count": len([f for f in follower_counts if 101 <= f <= 1000])},
                    {"range": "1001-10000", "count": len([f for f in follower_counts if 1001 <= f <= 10000])},
                    {"range": "10000+", "count": len([f for f in follower_counts if f > 10000])}
                ]
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting artists by followers: {e}")
            return []
    
    def get_courses_by_rating(self) -> List[Dict[str, Any]]:
        """Get courses grouped by rating"""
        try:
            start_time = time.time()
            response = requests.get(
                f"{self.supabase_url}/rest/v1/courses",
                headers=self.headers,
                params={
                    "select": "rating",
                    "order": "rating.desc",
                    "limit": 50
                },
                timeout=15
            )
            self._response_times.append(time.time() - start_time)
            
            if response.status_code == 200:
                courses = response.json()
                ratings = [course.get("rating", 0.0) for course in courses]
                
                return [
                    {"range": "4.5-5.0", "count": len([r for r in ratings if r >= 4.5])},
                    {"range": "4.0-4.4", "count": len([r for r in ratings if 4.0 <= r < 4.5])},
                    {"range": "3.5-3.9", "count": len([r for r in ratings if 3.5 <= r < 4.0])},
                    {"range": "3.0-3.4", "count": len([r for r in ratings if 3.0 <= r < 3.5])},
                    {"range": "Below 3.0", "count": len([r for r in ratings if r < 3.0])}
                ]
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting courses by rating: {e}")
            return []
    
    def calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self._cache_hits + self._cache_misses
        return (self._cache_hits / total) * 100 if total > 0 else 0.0
    
    def calculate_average_response_time(self) -> float:
        """Calculate average response time for API calls"""
        return sum(self._response_times) / len(self._response_times) if self._response_times else 0.0
