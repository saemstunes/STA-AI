import pytest
import os
import sys
import asyncio
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.supabase_integration import AdvancedSupabaseIntegration
from src.security_system import AdvancedSecuritySystem
from src.monitoring_system import ComprehensiveMonitor
from src.ai_system import SaemsTunesAISystem

def pytest_configure(config):
    """Configure pytest with production settings"""
    # Set production testing flags
    os.environ['PYTEST_CURRENT_TEST'] = 'production'
    
    # Verify required environment variables are set
    required_env_vars = ['SUPABASE_URL', 'SUPABASE_ANON_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        pytest.exit(f"Missing required environment variables: {', '.join(missing_vars)}")

@pytest.fixture(scope="session")
def real_supabase_connection():
    """Real Supabase connection fixture for production testing"""
    supabase_url = os.environ['SUPABASE_URL']
    supabase_key = os.environ['SUPABASE_ANON_KEY']
    
    # Create real Supabase integration
    supabase = AdvancedSupabaseIntegration(
        supabase_url=supabase_url,
        supabase_key=supabase_key
    )
    
    # Verify connection
    if not supabase.is_connected():
        pytest.fail("Cannot establish connection to production Supabase database")
    
    yield supabase
    
    # Cleanup
    supabase.clear_cache()

@pytest.fixture(scope="session")
def real_security_system():
    """Real security system fixture for production testing"""
    security = AdvancedSecuritySystem()
    yield security
    
    # Cleanup security logs
    security.security_log.clear()

@pytest.fixture(scope="session") 
def real_monitoring_system():
    """Real monitoring system fixture for production testing"""
    monitor = ComprehensiveMonitor(prometheus_port=8003)
    yield monitor
    
    # Stop monitoring
    monitor.stop_monitoring()

@pytest.fixture(scope="session")
def real_ai_system(real_supabase_connection, real_security_system, real_monitoring_system):
    """Real AI system fixture with production dependencies"""
    ai_system = SaemsTunesAISystem(
        supabase_integration=real_supabase_connection,
        security_system=real_security_system,
        monitor=real_monitoring_system,
        model_name="microsoft/Phi-3.5-mini-instruct",
        model_repo="Thetima4/Phi-3.5-mini-instruct-Q4_K_M-GGUF", 
        model_file="Phi-3.5-mini-instruct-q4_k_m.gguf",
        max_response_length=400,
        temperature=0.7
    )
    
    yield ai_system
    
    # Cleanup
    ai_system.clear_cache()

@pytest.fixture(scope="function")
def production_test_user():
    """Fixture providing test user data for production tests"""
    return {
        "user_id": f"test_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "ip_address": "192.168.1.100",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

@pytest.fixture(scope="function")
def sample_queries():
    """Fixture providing sample queries for testing"""
    return [
        "How do I create a playlist?",
        "What music courses do you offer?",
        "How can I upload my music?",
        "Tell me about guitar lessons",
        "What's included in premium?",
        "How do I find new artists?",
        "Can I download music for offline?",
        "What music genres are available?",
        "How do music recommendations work?",
        "Can I collaborate with other artists?"
    ]

@pytest.fixture(scope="session", autouse=True)
def verify_production_environment():
    """Verify we're configured for production testing"""
    # Check for production environment indicators
    supabase_url = os.getenv('SUPABASE_URL', '')
    
    if 'supabase.co' not in supabase_url:
        pytest.fail("Not configured for production Supabase instance")
    
    # Verify we have proper credentials
    supabase_key = os.getenv('SUPABASE_ANON_KEY', '')
    if not supabase_key.startswith('eyJ'):
        pytest.fail("Invalid Supabase key format")
    
    print("\n" + "="*60)
    print("PRODUCTION TEST ENVIRONMENT VERIFIED")
    print(f"Supabase URL: {supabase_url}")
    print(f"Test Timestamp: {datetime.now().isoformat()}")
    print("="*60 + "\n")

@pytest.fixture(scope="function")
def performance_thresholds():
    """Fixture defining performance thresholds for production tests"""
    return {
        "supabase_connection_ms": 1000,
        "context_generation_ms": 3000,
        "ai_response_ms": 10000,
        "security_check_ms": 100,
        "cache_response_ms": 50
    }

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()