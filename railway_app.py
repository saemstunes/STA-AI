import os
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response 
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import time
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our systems
from src.ai_system import SaemsTunesAISystem
from src.supabase_integration import AdvancedSupabaseIntegration
from src.security_system import AdvancedSecuritySystem
from src.monitoring_system import ComprehensiveMonitor

# Configuration
class Config:
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    MODEL_REPO = os.getenv("MODEL_REPO", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    MODEL_FILE = os.getenv("MODEL_FILE", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    PORT = int(os.getenv("PORT", 8000))
    ENVIRONMENT = os.getenv("RAILWAY_ENVIRONMENT", "production")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MAX_RESPONSE_LENGTH = int(os.getenv("MAX_RESPONSE_LENGTH", "500"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_P = float(os.getenv("TOP_P", "0.9"))
    CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "2048"))
    PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "8001"))
    RAILWAY_HEALTHCHECK_TIMEOUT = int(os.getenv("RAILWAY_HEALTHCHECK_TIMEOUT", "30"))

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"
    conversation_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    processing_time: float
    conversation_id: str
    timestamp: str
    model_used: str
    cache_hit: bool = False
    tokens_used: int = 0

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    environment: str
    systems: Dict[str, Any]
    resources: Dict[str, float]
    performance: Dict[str, Any]
    initialization_progress: Dict[str, Any]

class FeedbackRequest(BaseModel):
    conversation_id: str
    helpful: bool
    comments: Optional[str] = None
    user_id: Optional[str] = None
    response_quality: Optional[int] = None

class SystemInfoResponse(BaseModel):
    ai_system: Dict[str, Any]
    supabase: Dict[str, Any]
    security: Dict[str, Any]
    monitoring: Dict[str, Any]
    environment: str

# Setup logging - Railway-friendly (only StreamHandler)
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Only StreamHandler for Railway
)
logger = logging.getLogger(__name__)

# Global systems and initialization state
supabase_integration = None
security_system = None
monitor = None
ai_system = None
initialization_complete = False
systems_ready = False  # Separate flag for Railway health checks
initialization_start_time = None
initialization_errors = []

# Thread pool for background initialization
_executor = ThreadPoolExecutor(max_workers=1)

def initialize_systems():
    """Initialize all systems synchronously - runs in background thread"""
    global supabase_integration, security_system, monitor, ai_system, initialization_complete, systems_ready, initialization_errors
    
    logger.info("🚀 Starting system initialization in background thread...")
    
    try:
        # Initialize Supabase integration
        logger.info("📡 Connecting to Supabase...")
        supabase_integration = AdvancedSupabaseIntegration(
            Config.SUPABASE_URL, 
            Config.SUPABASE_ANON_KEY
        )
        
        if not supabase_integration.is_connected():
            raise Exception("Failed to connect to Supabase")
        logger.info("✅ Supabase integration initialized and connected")
        
        # Initialize security system
        logger.info("🔒 Initializing security system...")
        security_system = AdvancedSecuritySystem()
        logger.info("✅ Security system initialized")
        
        # Initialize monitoring
        logger.info("📊 Initializing monitoring system...")
        monitor = ComprehensiveMonitor(prometheus_port=Config.PROMETHEUS_PORT)
        logger.info("✅ Monitoring system initialized")
        
        # Initialize AI system
        logger.info("🤖 Initializing AI system with TinyLlama...")
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
        
        # Verify all systems are healthy
        if not ai_system.is_healthy():
            raise Exception("AI system health check failed")
            
        initialization_complete = True
        systems_ready = True
        initialization_time = time.time() - initialization_start_time
        logger.info(f"🎉 All systems initialized successfully in {initialization_time:.2f} seconds!")
        logger.info(f"📍 Environment: {Config.ENVIRONMENT}")
        logger.info(f"🔗 Supabase: Connected")
        logger.info(f"🤖 Model: {Config.MODEL_NAME} ({'Loaded' if ai_system.model_loaded else 'Failed'})")
        logger.info(f"🌐 API ready: http://0.0.0.0:{Config.PORT}")
        logger.info(f"📊 Metrics: http://0.0.0.0:{Config.PROMETHEUS_PORT}/metrics")
        
        return True
        
    except Exception as e:
        error_msg = f"System initialization failed: {str(e)}"
        logger.error(error_msg)
        initialization_errors.append(error_msg)
        initialization_complete = True  # Mark as complete but failed
        systems_ready = False
        return False

async def _initialize_in_background():
    """
    Run the synchronous initialize_systems() in a separate thread
    so the event loop / server doesn't block.
    """
    global initialization_start_time
    initialization_start_time = time.time()
    
    loop = asyncio.get_event_loop()
    try:
        success = await loop.run_in_executor(_executor, initialize_systems)
        if success:
            logger.info("✅ Background initialization succeeded")
        else:
            logger.error("❌ Background initialization failed")
    except Exception as e:
        logger.exception("Exception during background initialization: %s", e)

# Create FastAPI application
app = FastAPI(
    title="Saem's Tunes AI API",
    description="Production AI API for Saem's Tunes music education and streaming platform",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.saemstunes.com",
        "https://saemstunes.com",
        "http://localhost:3000",
        "http://localhost:5173",
        "https://saemstunes.vercel.app",
        "https://saems-tunes-git-main-saems-projects.vercel.app",
        "https://saemstunes.lovable.app",
        "https://preview--saemstunes.lovable.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Dependency to check if systems are ready
async def get_ai_system():
    if not systems_ready:
        raise HTTPException(
            status_code=503, 
            detail="System initializing. Please try again in a few moments."
        )
    if not ai_system or not ai_system.is_healthy():
        raise HTTPException(
            status_code=503, 
            detail="AI system is not ready. Please check system status."
        )
    return ai_system

async def get_security_system():
    if not systems_ready or not security_system:
        raise HTTPException(
            status_code=503, 
            detail="Security system initializing."
        )
    return security_system

# Health check endpoint - ALWAYS returns 200 for Railway
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health information - ALWAYS returns 200 for Railway"""
    # Get system resources (with fallback if psutil not available)
    resources = {}
    try:
        import psutil
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        resources = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024 ** 3),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024 ** 3)
        }
    except ImportError:
        resources = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "memory_used_gb": 0.0,
            "disk_percent": 0.0,
            "disk_free_gb": 0.0
        }
    
    # Determine status - Railway needs "healthy" status to pass health checks
    if systems_ready:
        status = "healthy"
    elif initialization_complete:
        status = "degraded"
    else:
        status = "initializing"
    
    # Get initialization progress
    init_progress = {
        "complete": initialization_complete,
        "ready": systems_ready,
        "started": initialization_start_time is not None,
        "errors": initialization_errors,
        "duration_seconds": time.time() - initialization_start_time if initialization_start_time else 0,
        "timeout_seconds": Config.RAILWAY_HEALTHCHECK_TIMEOUT
    }
    
    if monitor and systems_ready:
        performance_data = {
            "total_requests": len(monitor.inference_metrics),
            "average_response_time": monitor.get_average_response_time(),
            "error_rate": monitor.get_error_rate(),
            "uptime_seconds": monitor.get_uptime(),
            "throughput_rpm": monitor.get_throughput(),
            "cache_hit_rate": monitor.get_cache_hit_rate()
        }
    else:
        performance_data = {}
    
    return HealthResponse(
        status=status,  # This is what Railway checks - must be "healthy" to pass
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        environment=Config.ENVIRONMENT,
        systems={
            "supabase": supabase_integration.is_connected() if supabase_integration else False,
            "security": bool(security_system),
            "monitoring": bool(monitor),
            "ai_system": ai_system.is_healthy() if ai_system else False,
            "model_loaded": ai_system.model_loaded if ai_system else False,
            "systems_ready": systems_ready
        },
        resources=resources,
        performance=performance_data,
        initialization_progress=init_progress
    )

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """API health check endpoint - ALWAYS returns 200 for Railway"""
    return await root()

@app.get("/health", response_model=HealthResponse)
async def legacy_health_check():
    """Legacy health endpoint for Railway compatibility - ALWAYS returns 200"""
    return await health_check()

# Main chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    ai_system: SaemsTunesAISystem = Depends(get_ai_system),
    security_system: AdvancedSecuritySystem = Depends(get_security_system)
):
    """Main chat endpoint for React frontend"""
    try:
        if not request.message or not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if len(request.message) > 5000:
            raise HTTPException(status_code=400, detail="Message too long. Maximum 5000 characters.")
        
        # Security check with real IP and user agent - wrapped for safety
        security_result = {"allowed": True, "risk_score": 0, "alerts": []}  # Default fallback
        try:
            security_result = security_system.check_request(
                query=request.message,
                user_id=request.user_id,
                ip_address=request.ip_address,
                user_agent=request.user_agent
            )
        except Exception as security_error:
            logger.warning(f"Security system error, allowing request: {security_error}")
            # Continue with default allowed result
        
        if not security_result["allowed"]:
            logger.warning(
                f"Security block: user={request.user_id}, ip={request.ip_address}, "
                f"risk_score={security_result['risk_score']}, alerts={security_result['alerts']}"
            )
            raise HTTPException(
                status_code=429, 
                detail=f"Request blocked for security reasons: {', '.join(security_result['alerts'])}"
            )
        
        # Process query with real AI system
        start_time = time.time()
        response = ai_system.process_query(
            query=request.message, 
            user_id=request.user_id, 
            conversation_id=request.conversation_id
        )
        processing_time = time.time() - start_time
        
        # Check if response was from cache
        cache_key = f"{request.user_id}:{hash(request.message)}"
        cache_hit = cache_key in ai_system.response_cache
        
        # Estimate token usage
        tokens_used = len(request.message.split()) + len(response.split())
        
        logger.info(
            f"Chat processed: user={request.user_id}, "
            f"time={processing_time:.2f}s, "
            f"cache_hit={cache_hit}, "
            f"tokens={tokens_used}"
        )
        
        return ChatResponse(
            response=response,
            processing_time=processing_time,
            conversation_id=request.conversation_id or f"conv_{int(time.time())}_{hash(request.user_id)}",
            timestamp=datetime.now().isoformat(),
            model_used=Config.MODEL_NAME,
            cache_hit=cache_hit,
            tokens_used=tokens_used
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error processing your request")

# Additional API endpoints
@app.get("/api/models")
async def get_models():
    """Get available models information"""
    model_info = {
        "available_models": ["TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"],
        "current_model": Config.MODEL_NAME,
        "model_repo": Config.MODEL_REPO,
        "model_file": Config.MODEL_FILE,
        "quantization": "Q4_K_M",
        "context_length": Config.CONTEXT_WINDOW,
        "parameters": "1.1B",
        "max_response_length": Config.MAX_RESPONSE_LENGTH,
        "temperature": Config.TEMPERATURE,
        "top_p": Config.TOP_P
    }
    
    if ai_system and systems_ready:
        try:
            model_stats = ai_system.get_model_stats()
            model_info.update(model_stats)
        except Exception as e:
            logger.warning(f"Could not get model stats: {e}")
    
    return model_info

@app.get("/api/stats")
async def get_stats():
    """Get comprehensive system statistics"""
    if not monitor or not systems_ready:
        return {
            "status": "initializing" if not systems_ready else "degraded",
            "initialization_complete": initialization_complete,
            "systems_ready": systems_ready,
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        # Get performance summary
        performance_summary = monitor.get_performance_summary(timedelta(hours=1))
        
        # Get security stats
        security_stats = security_system.get_security_stats() if security_system else {}
        
        # Get AI system info
        ai_info = ai_system.get_system_info() if ai_system else {}
        
        return {
            "status": "healthy",
            "performance": performance_summary,
            "security": security_stats,
            "ai_system": ai_info,
            "supabase_connected": supabase_integration.is_connected() if supabase_integration else False,
            "systems_ready": systems_ready,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for AI responses"""
    try:
        # Log feedback for analysis
        logger.info(
            f"Feedback received: conversation={request.conversation_id}, "
            f"user={request.user_id}, helpful={request.helpful}, "
            f"quality={request.response_quality}, comments={request.comments}"
        )
        
        # Store feedback in Supabase if available
        if supabase_integration and supabase_integration.is_connected() and systems_ready:
            try:
                # This would be implemented to store in your feedback table
                # For now, we'll log it for later implementation
                pass
            except Exception as e:
                logger.warning(f"Could not store feedback in database: {e}")
        
        return {
            "status": "success", 
            "message": "Feedback received and logged",
            "conversation_id": request.conversation_id
        }
        
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        raise HTTPException(status_code=500, detail="Error submitting feedback")

@app.get("/api/context")
async def get_context_sample():
    """Get sample context data for debugging"""
    if not supabase_integration or not systems_ready:
        return {
            "status": "initializing",
            "systems_ready": systems_ready,
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        context = supabase_integration.get_music_context(
            "sample query for context debugging", 
            "debug_user"
        )
        return {
            "status": "success",
            "context_sample": {
                "summary": context.get("summary", ""),
                "stats": context.get("stats", {}),
                "tracks_count": len(context.get("tracks", [])),
                "artists_count": len(context.get("artists", [])),
                "courses_count": len(context.get("courses", []))
            },
            "supabase_connected": supabase_integration.is_connected(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e), 
            "supabase_connected": supabase_integration.is_connected() if supabase_integration else False,
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/system/info")
async def get_system_info():
    """Get detailed system information"""
    if not systems_ready:
        return {
            "status": "initializing",
            "systems_ready": systems_ready,
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        ai_info = ai_system.get_system_info() if ai_system else {}
        model_stats = ai_system.get_model_stats() if ai_system else {}
        
        supabase_info = {
            "connected": supabase_integration.is_connected() if supabase_integration else False,
            "url": Config.SUPABASE_URL if supabase_integration else None,
            "cache_size": len(supabase_integration.cache) if supabase_integration else 0
        }
        
        security_info = {
            "active": bool(security_system),
            "blocked_ips": len(security_system.blocked_ips) if security_system else 0,
            "rate_limits_tracked": len(security_system.rate_limits) if security_system else 0
        }
        
        monitoring_info = {
            "active": bool(monitor),
            "inference_metrics": len(monitor.inference_metrics) if monitor else 0,
            "system_metrics": len(monitor.system_metrics) if monitor else 0,
            "alerts": len(monitor.alerts) if monitor else 0
        }
        
        return SystemInfoResponse(
            ai_system={**ai_info, **model_stats},
            supabase=supabase_info,
            security=security_info,
            monitoring=monitoring_info,
            environment=Config.ENVIRONMENT
        )
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving system information")

@app.post("/api/security/reset-rate-limit")
async def reset_rate_limit(user_id: Optional[str] = None, ip_address: Optional[str] = None):
    """Reset rate limits for a user or IP (admin endpoint)"""
    if not security_system or not systems_ready:
        raise HTTPException(status_code=503, detail="Security system not available")
    
    try:
        security_system.reset_rate_limits(user_id=user_id, ip_address=ip_address)
        
        logger.info(f"Rate limits reset: user={user_id}, ip={ip_address}")
        
        return {
            "status": "success",
            "message": f"Rate limits reset for user={user_id}, ip={ip_address}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error resetting rate limits: {e}")
        raise HTTPException(status_code=500, detail="Error resetting rate limits")

@app.get("/api/monitoring/metrics")
async def get_prometheus_metrics():
    """Get Prometheus metrics"""
    if not monitor or not systems_ready:
        return Response(
            content="# Metrics not available - system initializing\n",
            media_type="text/plain; version=0.0.4"
        )
    
    try:
        metrics_data = monitor.get_prometheus_metrics()
        return Response(
            content=metrics_data,
            media_type="text/plain; version=0.0.4"
        )
    except Exception as e:
        logger.error(f"Error getting Prometheus metrics: {e}")
        return Response(
            content=f"# Error getting metrics: {e}\n",
            media_type="text/plain; version=0.0.4",
            status_code=500
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception in {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """
    Schedule the heavy initialization in background and return immediately.
    This allows the server to start and respond to health checks while
    initialization continues.
    """
    # Schedule background initialization (non-blocking)
    asyncio.create_task(_initialize_in_background())

    # Lightweight startup logging
    logger.info("🚀 Saem's Tunes AI API starting (initialization running in background)")
    logger.info(f"📍 Environment: {Config.ENVIRONMENT}")
    logger.info(f"🔗 Supabase URL: {Config.SUPABASE_URL}")
    logger.info(f"🤖 Model: {Config.MODEL_NAME}")
    logger.info(f"🌐 API docs: http://0.0.0.0:{Config.PORT}/docs")
    logger.info(f"📊 Metrics: http://0.0.0.0:{Config.PROMETHEUS_PORT}/metrics")
    logger.info(f"⚡ Max response length: {Config.MAX_RESPONSE_LENGTH}")
    logger.info(f"🎛️ Temperature: {Config.TEMPERATURE}, Top P: {Config.TOP_P}")
    logger.info(f"⏱️ Railway healthcheck timeout: {Config.RAILWAY_HEALTHCHECK_TIMEOUT}s")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Saem's Tunes AI API...")
    
    # Stop monitoring
    if monitor:
        monitor.stop_monitoring()
        logger.info("✅ Monitoring system stopped")
    
    # Shutdown thread pool
    _executor.shutdown(wait=False)
    logger.info("✅ Thread pool shutdown")
    
    logger.info("🎯 Shutdown completed successfully")

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint (legacy route)"""
    return await get_prometheus_metrics()

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=Config.PORT,
        log_level=Config.LOG_LEVEL.lower(),
        access_log=True,
        reload=Config.ENVIRONMENT == "development"
    )
