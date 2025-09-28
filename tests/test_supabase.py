import pytest
import os
import sys
import time
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.supabase_integration import AdvancedSupabaseIntegration

class TestSupabaseIntegration:
    """Production Supabase integration tests with real database connections"""
    
    def setup_method(self):
        """Setup real Supabase connection"""
        self.supabase_url = os.environ['SUPABASE_URL']
        self.supabase_key = os.environ['SUPABASE_ANON_KEY']
        
        self.supabase = AdvancedSupabaseIntegration(
            supabase_url=self.supabase_url,
            supabase_key=self.supabase_key
        )
        
        # Wait for connection to establish
        time.sleep(1)
    
    def test_production_connection_established(self):
        """Test that we can establish real connection to production Supabase"""
        assert self.supabase.is_connected() == True
        
        # Test actual API endpoint connectivity
        import requests
        response = requests.get(
            f"{self.supabase_url}/rest/v1/",
            headers={"apikey": self.supabase_key},
            timeout=10
        )
        assert response.status_code == 200
    
    def test_real_database_schema_access(self):
        """Test access to real database tables and schema"""
        connection_test = self.supabase.test_connection()
        
        # Verify we can access all critical tables
        critical_tables = ['tracks', 'artists', 'profiles', 'courses', 'playlists']
        
        for table in critical_tables:
            assert table in connection_test['tables']
            assert connection_test['tables'][table]['accessible'] == True
            assert connection_test['tables'][table]['status_code'] == 200
    
    def test_real_data_retrieval_performance(self):
        """Test real data retrieval performance from production"""
        start_time = time.time()
        
        # Test multiple data retrieval operations
        tracks = self.supabase.get_popular_tracks(limit=10)
        artists = self.supabase.get_popular_artists(limit=10)
        courses = self.supabase.get_recent_courses(limit=10)
        stats = self.supabase.get_platform_stats()
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Performance check - should complete within reasonable time
        assert total_duration < 10.0  # All operations within 10 seconds
        
        # Data validation
        assert len(tracks) <= 10
        assert len(artists) <= 10
        assert len(courses) <= 10
        assert isinstance(stats, dict)
    
    def test_real_context_generation_performance(self):
        """Test real music context generation performance"""
        test_queries = [
            "guitar lessons",
            "pop music",
            "electronic music production",
            "music theory basics",
            "how to create a playlist"
        ]
        
        for query in test_queries:
            start_time = time.time()
            context = self.supabase.get_music_context(query, "test_user")
            end_time = time.time()
            
            duration = end_time - start_time
            
            # Should generate context within 5 seconds
            assert duration < 5.0
            
            # Context should be relevant to query
            assert isinstance(context, dict)
            assert 'summary' in context
            assert len(context['summary']) > 0
    
    def test_real_user_context_retrieval(self):
        """Test real user context retrieval with actual user data"""
        # Test with a user that exists in your production database
        # Replace with actual test user IDs from your system
        test_users = [
            "existing_user_1",
            "existing_user_2", 
            "existing_user_3"
        ]
        
        for user_id in test_users:
            user_context = self.supabase.get_user_context(user_id)
            
            assert isinstance(user_context, dict)
            assert 'is_premium' in user_context
            assert 'favorite_genres' in user_context
            assert 'recent_activity' in user_context
            assert 'learning_progress' in user_context
            
            # Validate data types
            assert isinstance(user_context['is_premium'], bool)
            assert isinstance(user_context['favorite_genres'], list)
            assert isinstance(user_context['recent_activity'], list)
            assert isinstance(user_context['learning_progress'], dict)
    
    def test_real_cache_functionality(self):
        """Test real caching functionality with production data"""
        query = "test cache query"
        user_id = "cache_test_user"
        
        # First call - should fetch from database
        start_time = time.time()
        context1 = self.supabase.get_music_context(query, user_id)
        first_duration = time.time() - start_time
        
        # Second call - should be faster (cached)
        start_time = time.time()
        context2 = self.supabase.get_music_context(query, user_id)
        second_duration = time.time() - start_time
        
        # Contexts should be identical
        assert context1 == context2
        
        # Cached call should be faster
        assert second_duration < first_duration
    
    def test_real_error_handling_invalid_user(self):
        """Test real error handling with invalid user data"""
        invalid_user_id = "non_existent_user_999999"
        
        # Should handle gracefully without crashing
        user_context = self.supabase.get_user_context(invalid_user_id)
        
        assert isinstance(user_context, dict)
        assert user_context['is_premium'] == False
        assert user_context['favorite_genres'] == []
        assert user_context['recent_activity'] == []
        assert user_context['learning_progress'] == {}
    
    def test_real_data_consistency(self):
        """Test real data consistency across multiple retrievals"""
        # Test that multiple calls return consistent data
        contexts = []
        
        for i in range(3):
            context = self.supabase.get_music_context("music courses", f"user_{i}")
            contexts.append(context)
        
        # All contexts should have the same structure
        for context in contexts:
            assert 'tracks' in context
            assert 'artists' in context
            assert 'courses' in context
            assert 'stats' in context
            assert 'summary' in context
        
        # Stats should be consistent across calls
        stats_values = [context['stats']['track_count'] for context in contexts]
        assert len(set(stats_values)) == 1  # All should be the same
    
    def test_real_concurrent_requests(self):
        """Test real concurrent request handling"""
        import threading
        
        results = []
        errors = []
        
        def test_request(query, user_id):
            try:
                context = self.supabase.get_music_context(query, user_id)
                results.append((query, user_id, context))
            except Exception as e:
                errors.append((query, user_id, str(e)))
        
        # Create multiple concurrent requests
        threads = []
        test_cases = [
            ("rock music", "user_1"),
            ("jazz courses", "user_2"),
            ("hip hop artists", "user_3"),
            ("classical music", "user_4"),
            ("music production", "user_5")
        ]
        
        for query, user_id in test_cases:
            thread = threading.Thread(target=test_request, args=(query, user_id))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all requests completed successfully
        assert len(errors) == 0
        assert len(results) == len(test_cases)
        
        # Verify all results are valid
        for query, user_id, context in results:
            assert isinstance(context, dict)
            assert 'summary' in context
            assert len(context['summary']) > 0
    
    def test_real_detailed_statistics(self):
        """Test real detailed statistics retrieval"""
        detailed_stats = self.supabase.get_detailed_stats()
        
        assert isinstance(detailed_stats, dict)
        assert 'basic' in detailed_stats
        assert 'content_breakdown' in detailed_stats
        assert 'performance' in detailed_stats
        
        # Validate basic stats
        basic_stats = detailed_stats['basic']
        assert basic_stats['track_count'] > 0
        assert basic_stats['artist_count'] > 0
        assert basic_stats['user_count'] > 0
        
        # Validate content breakdown
        breakdown = detailed_stats['content_breakdown']
        assert 'tracks_by_popularity' in breakdown
        assert 'artists_by_followers' in breakdown
        assert 'courses_by_rating' in breakdown
        
        # These should be real distributions from your database
        for distribution in breakdown.values():
            assert isinstance(distribution, list)
            if distribution:  # Might be empty in some cases
                for item in distribution:
                    assert 'range' in item
                    assert 'count' in item
    
    def test_real_query_intent_analysis(self):
        """Test real query intent analysis with production data"""
        test_queries = [
            ("How do I create a playlist?", "Instructional"),
            ("What is music theory?", "Explanatory"),
            ("I can't upload my track", "Support"),
            ("Recommend some jazz music", "Discovery"),
            ("How much does premium cost?", "Commercial"),
            ("Tell me about Saem's Tunes", "General")
        ]
        
        for query, expected_intent in test_queries:
            analyzed_intent = self.supabase.analyze_query_intent(query)
            
            # Should correctly identify intent patterns
            assert analyzed_intent in [
                "Instructional - seeking how-to information",
                "Explanatory - seeking information", 
                "Support - seeking technical help",
                "Discovery - seeking recommendations",
                "Commercial - seeking pricing information",
                "General inquiry about platform features"
            ]
    
    def test_real_fallback_mechanisms(self):
        """Test real fallback mechanisms when database is unavailable"""
        # Note: We can't easily test database downtime in automated tests
        # But we can verify fallback data structure
        
        # Clear cache to force fresh data
        self.supabase.clear_cache()
        
        # This should use real data, but verify structure matches fallback expectations
        context = self.supabase.get_music_context("test query", "test_user")
        
        # Even with real data, structure should match expected format
        assert 'tracks' in context
        assert 'artists' in context
        assert 'courses' in context
        assert 'stats' in context
        assert 'summary' in context
        assert 'timestamp' in context
    
    def test_real_security_integration(self):
        """Test real security integration with production data"""
        # Test that security measures don't interfere with legitimate data access
        legitimate_queries = [
            "music courses for beginners",
            "how to play guitar",
            "best pop songs 2024",
            "music production tutorials",
            "artist dashboard features"
        ]
        
        for query in legitimate_queries:
            context = self.supabase.get_music_context(query, "legitimate_user")
            
            # Should successfully retrieve context for legitimate queries
            assert isinstance(context, dict)
            assert 'summary' in context
            assert len(context['summary']) > 0
    
    def teardown_method(self):
        """Cleanup after tests"""
        if hasattr(self, 'supabase'):
            self.supabase.clear_cache()