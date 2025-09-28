import os
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/Phi-3.5-mini-instruct")
    PORT = int(os.getenv("PORT", 8000))
    ENVIRONMENT = os.getenv("RAILWAY_ENVIRONMENT", "production")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MAX_RESPONSE_LENGTH = int(os.getenv("MAX_RESPONSE_LENGTH", "500"))

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    processing_time: float
    conversation_id: str
    timestamp: str
    model_used: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    environment: str
    systems: Dict[str, bool]
    resources: Dict[str, float]
    performance: Dict[str, Any]

class FeedbackRequest(BaseModel):
    conversation_id: str
    helpful: bool
    comments: Optional[str] = None

# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('railway_ai.log')
    ]
)
logger = logging.getLogger(__name__)

# Global systems
supabase_integration = None
security_system = None
monitor = None
ai_system = None

def initialize_systems():
    """Initialize all systems"""
    global supabase_integration, security_system, monitor, ai_system
    
    logger.info("🚀 Initializing Saem's Tunes AI System for Railway...")
    
    try:
        # Initialize Supabase integration
        supabase_integration = AdvancedSupabaseIntegration(
            Config.SUPABASE_URL, 
            Config.SUPABASE_ANON_KEY
        )
        logger.info("✅ Supabase integration initialized")
        
        # Initialize security system
        security_system = AdvancedSecuritySystem()
        logger.info("✅ Security system initialized")
        
        # Initialize monitoring
        monitor = ComprehensiveMonitor()
        logger.info("✅ Monitoring system initialized")
        
        # Initialize AI system
        ai_system = SaemsTunesAISystem(
            supabase_integration, 
            security_system, 
            monitor,
            model_name=Config.MODEL_NAME,
            max_response_length=Config.MAX_RESPONSE_LENGTH
        )
        logger.info("✅ AI system initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize systems: {e}")
        return False

# Create FastAPI application
app = FastAPI(
    title="Saem's Tunes AI API",
    description="Backup AI API for Saem's Tunes music education and streaming platform",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Dependency to check if systems are ready
async def get_ai_system():
    if not ai_system or not ai_system.is_healthy():
        raise HTTPException(status_code=503, detail="AI system is not ready")
    return ai_system

# Health check endpoint
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health information"""
    import psutil
    
    systems_ready = all([supabase_integration, security_system, monitor, ai_system])
    
    if not systems_ready:
        return HealthResponse(
            status="initializing",
            timestamp=datetime.now().isoformat(),
            version="2.0.0",
            environment=Config.ENVIRONMENT,
            systems={
                "supabase": False,
                "security": False,
                "monitoring": False,
                "ai_system": False
            },
            resources={
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "disk_percent": 0.0
            },
            performance={}
        )
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        environment=Config.ENVIRONMENT,
        systems={
            "supabase": supabase_integration.is_connected(),
            "security": True,
            "monitoring": True,
            "ai_system": ai_system.is_healthy()
        },
        resources={
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        },
        performance={
            "total_requests": len(monitor.inference_metrics),
            "average_response_time": monitor.get_average_response_time(),
            "error_rate": monitor.get_error_rate(),
            "uptime_seconds": monitor.get_uptime()
        }
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return await root()

# Main chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    ai_system: SaemsTunesAISystem = Depends(get_ai_system)
):
    """Main chat endpoint for React frontend"""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Security check
        security_result = security_system.check_request(request.message, request.user_id)
        if security_result["is_suspicious"]:
            logger.warning(f"Suspicious request blocked from {request.user_id}: {request.message}")
            raise HTTPException(status_code=429, detail="Request blocked for security reasons")
        
        # Process query
        start_time = time.time()
        response = ai_system.process_query(request.message, request.user_id, request.conversation_id)
        processing_time = time.time() - start_time
        
        logger.info(f"API chat processed for {request.user_id}: {processing_time:.2f}s")
        
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
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Additional API endpoints
@app.get("/api/models")
async def get_models():
    """Get available models information"""
    return {
        "available_models": ["microsoft/Phi-3.5-mini-instruct"],
        "current_model": Config.MODEL_NAME,
        "quantization": "Q4_K_M",
        "context_length": 4096,
        "parameters": "3.8B",
        "max_response_length": Config.MAX_RESPONSE_LENGTH
    }

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    if not monitor:
        return {"error": "Monitoring system not available"}
    
    return {
        "total_requests": len(monitor.inference_metrics),
        "average_response_time": monitor.get_average_response_time(),
        "error_rate": monitor.get_error_rate(),
        "uptime_seconds": monitor.get_uptime(),
        "system_health": await root()
    }

@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for AI responses"""
    try:
        # Log feedback for analysis
        logger.info(f"Feedback received for {request.conversation_id}: helpful={request.helpful}, comments={request.comments}")
        
        # In a real implementation, you would store this in your database
        # For now, we'll just log it
        
        return {"status": "success", "message": "Feedback received"}
        
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        raise HTTPException(status_code=500, detail="Error submitting feedback")

@app.get("/api/context")
async def get_context_sample():
    """Get sample context data for debugging"""
    if not supabase_integration:
        return {"error": "Supabase integration not available"}
    
    try:
        context = supabase_integration.get_music_context("sample query for context")
        return {
            "context_sample": context,
            "supabase_connected": supabase_integration.is_connected()
        }
    except Exception as e:
        return {"error": str(e), "supabase_connected": False}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize systems on startup"""
    success = initialize_systems()
    if success:
        logger.info("✅ Saem's Tunes AI API is ready!")
        logger.info(f"📍 Environment: {Config.ENVIRONMENT}")
        logger.info(f"🔗 Supabase: {'Connected' if supabase_integration.is_connected() else 'Disconnected'}")
        logger.info(f"🤖 Model: {Config.MODEL_NAME}")
        logger.info(f"🌐 API docs: http://localhost:{Config.PORT}/docs")
    else:
        logger.error("❌ Failed to initialize systems on startup")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Saem's Tunes AI API...")
    if monitor:
        monitor.stop_monitoring()

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    if not monitor:
        return {"error": "Monitoring not available"}
    
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    except ImportError:
        return {"error": "Prometheus client not installed"}

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