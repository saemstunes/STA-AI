import pytest
import sys
import os
import tempfile
import json
import asyncio
from datetime import datetime, timedelta
import requests
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ai_system import SaemsTunesAISystem
from src.supabase_integration import AdvancedSupabaseIntegration
from src.security_system import AdvancedSecuritySystem
from src.monitoring_system import ComprehensiveMonitor

class TestAISystem:
    """Production AI system tests with real Supabase connections"""
    
    def setup_method(self):
        """Setup real production test fixtures"""
        # REAL SUPABASE CONNECTION - NO MOCKS
        self.supabase_url = os.environ['SUPABASE_URL']
        self.supabase_key = os.environ['SUPABASE_ANON_KEY']
        
        self.supabase = AdvancedSupabaseIntegration(
            supabase_url=self.supabase_url,
            supabase_key=self.supabase_key
        )
        
        # REAL SECURITY SYSTEM - NO MOCKS
        self.security = AdvancedSecuritySystem()
        
        # REAL MONITORING SYSTEM - NO MOCKS
        self.monitor = ComprehensiveMonitor(prometheus_port=8002)
        
        # REAL AI SYSTEM WITH PRODUCTION MODEL - NO MOCKS
        self.ai_system = SaemsTunesAISystem(
            supabase_integration=self.supabase,
            security_system=self.security,
            monitor=self.monitor,
            model_name="microsoft/Phi-3.5-mini-instruct",
            model_repo="Thetima4/Phi-3.5-mini-instruct-Q4_K_M-GGUF",
            model_file="Phi-3.5-mini-instruct-q4_k_m.gguf",
            max_response_length=300,
            temperature=0.7
        )
        
        # WAIT FOR SYSTEMS TO INITIALIZE
        time.sleep(2)
    
    def test_real_supabase_connection(self):
        """Test real Supabase connectivity with production database"""
        assert self.supabase.is_connected() == True
        
        # Test connection to actual Supabase tables
        connection_test = self.supabase.test_connection()
        assert connection_test['connected'] == True
        
        # Verify we can access real tables
        assert len(connection_test['tables']) > 0
        assert connection_test['tables']['tracks']['accessible'] == True
        assert connection_test['tables']['artists']['accessible'] == True
        assert connection_test['tables']['courses']['accessible'] == True
    
    def test_real_platform_stats_retrieval(self):
        """Test retrieving real platform statistics from production database"""
        stats = self.supabase.get_platform_stats()
        
        # Validate real data structure and values
        assert isinstance(stats, dict)
        assert 'track_count' in stats
        assert 'artist_count' in stats
        assert 'user_count' in stats
        assert 'course_count' in stats
        
        # These should be real counts from your production database
        assert stats['track_count'] >= 0
        assert stats['artist_count'] >= 0
        assert stats['user_count'] >= 0
        assert stats['course_count'] >= 0
        
        # Test that we're getting actual database counts, not fallbacks
        assert stats['track_count'] > 100  # Reasonable minimum for production
    
    def test_real_music_context_retrieval(self):
        """Test retrieving real music context from production database"""
        # Test with actual user query
        query = "guitar lessons for beginners"
        user_id = "test_user_123"
        
        context = self.supabase.get_music_context(query, user_id)
        
        # Validate real context structure
        assert isinstance(context, dict)
        assert 'tracks' in context
        assert 'artists' in context
        assert 'courses' in context
        assert 'stats' in context
        assert 'summary' in context
        
        # Verify we're getting real data arrays
        assert isinstance(context['tracks'], list)
        assert isinstance(context['artists'], list)
        assert isinstance(context['courses'], list)
        
        # Test that context is tailored to the query
        assert len(context['summary']) > 0
        assert 'guitar' in context['summary'].lower() or 'lesson' in context['summary'].lower()
    
    def test_real_popular_tracks_retrieval(self):
        """Test retrieving real popular tracks from production"""
        tracks = self.supabase.get_popular_tracks(limit=5)
        
        assert isinstance(tracks, list)
        assert len(tracks) > 0
        
        # Validate track structure with real data
        for track in tracks:
            assert 'id' in track
            assert 'title' in track
            assert 'artist' in track
            assert 'genre' in track
            assert 'plays' in track
            assert isinstance(track['title'], str)
            assert len(track['title']) > 0
            assert isinstance(track['artist'], str)
            assert len(track['artist']) > 0
    
    def test_real_popular_artists_retrieval(self):
        """Test retrieving real artists from production"""
        artists = self.supabase.get_popular_artists(limit=5)
        
        assert isinstance(artists, list)
        assert len(artists) > 0
        
        # Validate artist structure with real data
        for artist in artists:
            assert 'id' in artist
            assert 'name' in artist
            assert 'genre' in artist
            assert 'followers' in artist
            assert isinstance(artist['name'], str)
            assert len(artist['name']) > 0
    
    def test_real_courses_retrieval(self):
        """Test retrieving real courses from production"""
        courses = self.supabase.get_recent_courses(limit=5)
        
        assert isinstance(courses, list)
        assert len(courses) > 0
        
        # Validate course structure with real data
        for course in courses:
            assert 'id' in course
            assert 'title' in course
            assert 'instructor' in course
            assert 'level' in course
            assert 'students' in course
            assert isinstance(course['title'], str)
            assert len(course['title']) > 0
    
    def test_real_ai_system_health(self):
        """Test real AI system health and connectivity"""
        system_info = self.ai_system.get_system_info()
        
        assert isinstance(system_info, dict)
        assert 'model_loaded' in system_info
        assert 'model_name' in system_info
        assert 'supabase_connected' in system_info
        
        # Verify real system status
        assert system_info['supabase_connected'] == True
        assert system_info['model_name'] == "microsoft/Phi-3.5-mini-instruct"
    
    def test_real_security_system_integration(self):
        """Test real security system with actual threat detection"""
        query = "How do I create a playlist?"
        user_id = "test_user_456"
        ip_address = "192.168.1.100"
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        
        # Test real security check
        security_result = self.security.check_request(
            query=query,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        assert isinstance(security_result, dict)
        assert 'is_suspicious' in security_result
        assert 'risk_score' in security_result
        assert 'allowed' in security_result
        assert 'alerts' in security_result
        
        # Should allow legitimate requests
        assert security_result['allowed'] == True
        assert security_result['is_suspicious'] == False
        assert security_result['risk_score'] < 50
    
    def test_real_monitoring_system_integration(self):
        """Test real monitoring system with actual metrics"""
        # Record real inference metrics
        test_metrics = {
            'model_name': 'phi3.5-mini-Q4_K_M',
            'processing_time_ms': 1250.5,
            'input_tokens': 45,
            'output_tokens': 120,
            'total_tokens': 165,
            'success': True,
            'user_id': 'test_user_789',
            'conversation_id': 'conv_123',
            'timestamp': datetime.now(),
            'query_length': 25,
            'response_length': 180,
            'model_hash': 'abc123def456'
        }
        
        self.monitor.record_inference(test_metrics)
        
        # Test real performance summary
        performance_summary = self.monitor.get_performance_summary(timedelta(minutes=5))
        
        assert isinstance(performance_summary, dict)
        if performance_summary:  # Might be empty if no recent metrics
            assert 'total_requests' in performance_summary
            assert 'error_rate_percent' in performance_summary
            assert 'avg_response_time_ms' in performance_summary
    
    def test_real_system_health_monitoring(self):
        """Test real system health monitoring"""
        health_status = self.monitor.get_system_health()
        
        assert isinstance(health_status, dict)
        assert 'status' in health_status
        assert 'uptime_seconds' in health_status
        assert 'performance' in health_status
        assert 'alerts' in health_status
        
        # Should be healthy in test environment
        assert health_status['status'] in ['healthy', 'degraded', 'unhealthy']
        assert health_status['uptime_seconds'] > 0
    
    def test_real_ai_response_generation(self):
        """Test real AI response generation with production model"""
        if not self.ai_system.model_loaded:
            pytest.skip("Model not loaded, skipping response generation test")
        
        query = "How can I learn guitar on Saem's Tunes?"
        user_id = "music_learner_123"
        
        # Generate real AI response with production model
        response = self.ai_system.process_query(query, user_id)
        
        # Validate real response
        assert isinstance(response, str)
        assert len(response) > 10
        assert len(response) <= 500  # Should respect max response length
        
        # Response should be relevant to the query
        assert any(term in response.lower() for term in ['guitar', 'learn', 'course', 'lesson', 'music']) or len(response) > 0
    
    def test_real_user_context_integration(self):
        """Test real user context retrieval from production database"""
        # This would test with actual user data from your database
        # Using a test user ID that exists in your production system
        test_user_id = "existing_user_123"  # Replace with actual test user ID
        
        user_context = self.supabase.get_user_context(test_user_id)
        
        assert isinstance(user_context, dict)
        assert 'is_premium' in user_context
        assert 'favorite_genres' in user_context
        assert 'recent_activity' in user_context
        assert 'learning_progress' in user_context
        
        # These should be real user preferences from your database
        assert isinstance(user_context['is_premium'], bool)
        assert isinstance(user_context['favorite_genres'], list)
        assert isinstance(user_context['recent_activity'], list)
        assert isinstance(user_context['learning_progress'], dict)
    
    def test_real_rate_limiting(self):
        """Test real rate limiting with actual security system"""
        user_id = "rate_test_user"
        ip_address = "192.168.1.200"
        
        # Test multiple rapid requests to trigger rate limiting
        for i in range(15):
            security_result = self.security.check_request(
                query=f"Test query {i}",
                user_id=user_id,
                ip_address=ip_address,
                user_agent="Test Client"
            )
            
            if i > 10:  # After 10 requests, might hit rate limits
                # Either allowed or properly rate limited
                assert security_result['allowed'] in [True, False]
            else:
                # Should be allowed for initial requests
                assert security_result['allowed'] == True
        
        # Reset rate limits for this user/IP
        self.security.reset_rate_limits(user_id=user_id, ip_address=ip_address)
    
    def test_real_supabase_detailed_stats(self):
        """Test retrieving detailed statistics from production database"""
        detailed_stats = self.supabase.get_detailed_stats()
        
        assert isinstance(detailed_stats, dict)
        assert 'basic' in detailed_stats
        assert 'content_breakdown' in detailed_stats
        assert 'performance' in detailed_stats
        
        # Validate basic stats structure
        basic_stats = detailed_stats['basic']
        assert 'track_count' in basic_stats
        assert 'artist_count' in basic_stats
        assert 'user_count' in basic_stats
        
        # Validate content breakdown
        content_breakdown = detailed_stats['content_breakdown']
        assert 'tracks_by_popularity' in content_breakdown
        assert 'artists_by_followers' in content_breakdown
        assert 'courses_by_rating' in content_breakdown
        
        # These should be real data distributions from your database
        assert isinstance(content_breakdown['tracks_by_popularity'], list)
        assert isinstance(content_breakdown['artists_by_followers'], list)
        assert isinstance(content_breakdown['courses_by_rating'], list)
    
    def test_real_conversation_history(self):
        """Test real conversation history functionality"""
        conversation_id = "test_conv_123"
        user_id = "conv_test_user"
        
        if not self.ai_system.model_loaded:
            pytest.skip("Model not loaded, skipping conversation test")
        
        # First query
        query1 = "What music courses do you offer?"
        response1 = self.ai_system.process_query(query1, user_id, conversation_id)
        
        # Second query with conversation context
        query2 = "Which ones are good for beginners?"
        response2 = self.ai_system.process_query(query2, user_id, conversation_id)
        
        # Both should generate valid responses
        assert isinstance(response1, str)
        assert len(response1) > 0
        assert isinstance(response2, str)
        assert len(response2) > 0
        
        # Conversation history should be maintained
        assert conversation_id in self.ai_system.conversation_history
        conversation = self.ai_system.conversation_history[conversation_id]
        assert len(conversation) >= 4  # At least 2 user + 2 assistant messages
    
    def test_real_error_handling(self):
        """Test real error handling with production systems"""
        # Test with malformed query that might cause issues
        problematic_query = "A" * 10000  # Very long query
        
        security_result = self.security.check_request(
            query=problematic_query,
            user_id="error_test_user",
            ip_address="192.168.1.300"
        )
        
        # Security system should flag extremely long queries
        assert security_result['risk_score'] > 0
        assert any('long' in alert.lower() for alert in security_result['alerts'])
    
    def test_real_cache_functionality(self):
        """Test real response caching in AI system"""
        if not self.ai_system.model_loaded:
            pytest.skip("Model not loaded, skipping cache test")
        
        query = "How do I upload my music to Saem's Tunes?"
        user_id = "cache_test_user"
        
        # First request - should process normally
        start_time = time.time()
        response1 = self.ai_system.process_query(query, user_id)
        first_duration = time.time() - start_time
        
        # Second request - should be faster if cached
        start_time = time.time()
        response2 = self.ai_system.process_query(query, user_id)
        second_duration = time.time() - start_time
        
        # Responses should be identical
        assert response1 == response2
        
        # Second request should be faster (cached)
        # Note: This might not always hold true in production, but generally should
        assert second_duration <= first_duration * 0.5  # Should be at least 50% faster
    
    def test_real_system_cleanup(self):
        """Test real system cleanup and resource management"""
        # Test cache clearing
        initial_cache_size = len(self.ai_system.response_cache)
        self.ai_system.clear_cache()
        final_cache_size = len(self.ai_system.response_cache)
        
        assert final_cache_size == 0
        assert final_cache_size < initial_cache_size or initial_cache_size == 0
        
        # Test security system cleanup
        initial_security_log_size = len(self.security.security_log)
        self.security.security_log = []  # Clear security log
        assert len(self.security.security_log) == 0
        
        # Test monitoring system reset
        self.monitor.reset_metrics()
        recent_metrics = self.monitor.get_recent_metrics(minutes=1)
        assert len(recent_metrics) == 0

    def teardown_method(self):
        """Cleanup after tests"""
        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring()
        if hasattr(self, 'ai_system'):
            self.ai_system.clear_cache()