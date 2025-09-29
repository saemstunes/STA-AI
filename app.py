import os
import gradio as gr
import json
import time
import logging
import psutil
import asyncio
import threading
from datetime import datetime
from typing import List, Dict, Optional, Any
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

# Import systems - but don't initialize them until needed
try:
    from src.ai_system import SaemsTunesAISystem
    from src.supabase_integration import AdvancedSupabaseIntegration
    from src.security_system import AdvancedSecuritySystem
    from src.monitoring_system import ComprehensiveMonitor
    SYSTEMS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import systems: {e}")
    SYSTEMS_AVAILABLE = False

class Config:
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    MODEL_REPO = os.getenv("MODEL_REPO", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    MODEL_FILE = os.getenv("MODEL_FILE", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    HF_SPACE = os.getenv("HF_SPACE", "saemstunes/STA-AI")
    PORT = int(os.getenv("PORT", 7860))  # Hugging Face Spaces uses 7860
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MAX_RESPONSE_LENGTH = int(os.getenv("MAX_RESPONSE_LENGTH", "500"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_P = float(os.getenv("TOP_P", "0.9"))
    CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "2048"))
    ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "true").lower() == "true"

# Setup minimal logging first
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Global systems - initialize as None
supabase_integration = None
security_system = None
monitor = None
ai_system = None
systems_ready = False
initialization_complete = False
initialization_errors = []
initialization_start_time = None
initialization_thread = None

def initialize_systems():
    """Initialize all systems - runs in background thread"""
    global supabase_integration, security_system, monitor, ai_system, systems_ready, initialization_complete, initialization_errors
    
    if not SYSTEMS_AVAILABLE:
        initialization_errors.append("System dependencies not available")
        initialization_complete = True
        return False
    
    logger.info("🚀 Initializing Saem's Tunes AI System...")
    
    try:
        # Initialize Supabase integration
        logger.info("📡 Connecting to Supabase...")
        supabase_integration = AdvancedSupabaseIntegration(
            Config.SUPABASE_URL, 
            Config.SUPABASE_ANON_KEY
        )
        
        if not supabase_integration.is_connected():
            logger.warning("⚠️ Supabase connection failed, continuing with fallback data")
        else:
            logger.info("✅ Supabase integration initialized")
        
        # Initialize security system
        logger.info("🔒 Initializing security system...")
        security_system = AdvancedSecuritySystem()
        logger.info("✅ Security system initialized")
        
        # Initialize monitoring
        logger.info("📊 Initializing monitoring system...")
        monitor = ComprehensiveMonitor(prometheus_port=8001)
        logger.info("✅ Monitoring system initialized")
        
        # Initialize AI system - this is the heavy part
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
        
        # Check if AI system is healthy
        if ai_system and ai_system.is_healthy():
            systems_ready = True
            initialization_complete = True
            logger.info("🎉 All systems initialized successfully!")
        else:
            initialization_errors.append("AI system health check failed")
            initialization_complete = True
            logger.warning("⚠️ AI system not fully healthy, but initialization complete")
        
        return True
        
    except Exception as e:
        error_msg = f"System initialization failed: {str(e)}"
        logger.error(error_msg)
        initialization_errors.append(error_msg)
        initialization_complete = True
        return False

def start_initialization():
    """Start system initialization in background"""
    global initialization_thread, initialization_start_time
    initialization_start_time = time.time()
    
    initialization_thread = threading.Thread(target=initialize_systems, daemon=True)
    initialization_thread.start()
    logger.info("🔄 Started system initialization in background thread")

def get_system_status() -> Dict[str, Any]:
    """Get current system status - lightweight and safe"""
    try:
        # Get resource usage safely
        resources = {}
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            resources = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024 ** 3),
                "disk_percent": disk.percent
            }
        except Exception as e:
            resources = {"error": f"Resource monitoring failed: {e}"}
        
        if not initialization_complete:
            return {
                "status": "initializing", 
                "details": "Systems are starting up...",
                "timestamp": datetime.now().isoformat(),
                "initialization_started": initialization_start_time is not None,
                "duration_seconds": time.time() - initialization_start_time if initialization_start_time else 0,
                "resources": resources
            }
        
        if not systems_ready:
            return {
                "status": "degraded",
                "details": "Systems initialized but not fully ready",
                "errors": initialization_errors,
                "timestamp": datetime.now().isoformat(),
                "resources": resources
            }
        
        # Systems are ready - get detailed status
        systems_status = {
            "supabase": supabase_integration.is_connected() if supabase_integration else False,
            "security": bool(security_system),
            "monitoring": bool(monitor),
            "ai_system": ai_system.is_healthy() if ai_system else False,
            "model_loaded": ai_system.model_loaded if ai_system else False
        }
        
        performance = {}
        if monitor:
            try:
                performance = {
                    "total_requests": len(monitor.inference_metrics),
                    "avg_response_time": monitor.get_average_response_time(),
                    "error_rate": monitor.get_error_rate()
                }
            except Exception as e:
                performance = {"error": f"Performance monitoring failed: {e}"}
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "systems": systems_status,
            "resources": resources,
            "performance": performance
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def chat_interface(message: str, history: List[List[str]], request: gr.Request) -> str:
    """Gradio chat interface - with proper error handling"""
    try:
        if not message.strip():
            return "Please ask me anything about Saem's Tunes!"
        
        if not systems_ready or not ai_system:
            return "🔄 Systems are still initializing. Please wait a moment and try again..."
        
        # Get client info safely
        user_ip = "unknown"
        try:
            client_host = getattr(request, "client", None)
            if client_host:
                user_ip = client_host.host
        except:
            pass
        
        user_id = f"gradio_user_{user_ip}"
        
        # Security check with fallback
        security_check_passed = True
        try:
            if security_system:
                security_result = security_system.check_request(message, user_id)
                if security_result["is_suspicious"]:
                    logger.warning(f"Suspicious request blocked from {user_ip}: {message}")
                    return "Your request has been blocked for security reasons. Please try a different question."
        except Exception as e:
            logger.warning(f"Security check failed, allowing request: {e}")
        
        # Process query
        start_time = time.time()
        try:
            response = ai_system.process_query(message, user_id)
            processing_time = time.time() - start_time
            
            formatted_response = f"{response}\n\n_Generated in {processing_time:.1f}s_"
            
            logger.info(f"Chat processed: {message[:50]}... -> {processing_time:.2f}s")
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"AI processing error: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again later."
        
    except Exception as e:
        logger.error(f"Chat interface error: {e}")
        return "I apologize, but I'm experiencing technical difficulties. Please try again later."

# Create FastAPI app at module level - CRITICAL FOR HUGGING FACE
fastapi_app = FastAPI(
    title="Saem's Tunes AI API",
    description="AI Assistant for Saem's Tunes Music Platform",
    version="2.0.0"
)

# Root endpoint - ALWAYS returns 200 for Hugging Face health checks
@fastapi_app.get("/")
def root():
    """Root endpoint - MUST return 200 immediately for Hugging Face"""
    return {
        "status": "healthy" if systems_ready else "initializing",
        "message": "Saem's Tunes AI API is running",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "environment": "huggingface-spaces"
    }

# Health endpoint - ALWAYS returns 200
@fastapi_app.get("/api/health")
def api_health():
    """Health endpoint - ALWAYS returns 200 for Hugging Face"""
    try:
        status_data = get_system_status()
        return status_data
    except Exception as e:
        logger.error(f"Health endpoint error: {e}")
        return JSONResponse(
            content={
                "status": "error", 
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=200  # Always 200 for Hugging Face
        )

# Other API endpoints with proper error handling
@fastapi_app.get("/api/models")
def api_models():
    """Get model information"""
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
        "top_p": Config.TOP_P
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
    """Get system statistics"""
    if not systems_ready:
        return {
            "status": "initializing",
            "systems_ready": systems_ready,
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        stats_data = {
            "status": "healthy",
            "system_health": get_system_status(),
            "timestamp": datetime.now().isoformat()
        }
        
        if monitor:
            stats_data.update({
                "total_requests": len(monitor.inference_metrics),
                "average_response_time": monitor.get_average_response_time(),
                "error_rate": monitor.get_error_rate(),
                "uptime": monitor.get_uptime(),
            })
            
        return stats_data
        
    except Exception as e:
        logger.error(f"Stats endpoint error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def create_gradio_interface():
    """Create Gradio interface - lightweight and fast"""
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        max-width: 900px;
        margin: 0 auto;
    }
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-healthy { background-color: #4CAF50; }
    .status-warning { background-color: #FF9800; }
    .status-error { background-color: #F44336; }
    .quick-actions {
        display: flex;
        gap: 10px;
        margin: 15px 0;
        flex-wrap: wrap;
    }
    .quick-action-btn {
        background: #f0f0f0;
        border: 1px solid #ddd;
        border-radius: 20px;
        padding: 8px 16px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .quick-action-btn:hover {
        background: #e0e0e0;
        border-color: #667eea;
    }
    .footer {
        text-align: center;
        color: #666;
        margin-top: 20px;
        padding-top: 15px;
        border-top: 1px solid #eee;
    }
    """
    
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="purple"),
        title="Saem's Tunes AI Assistant",
        css=custom_css
    ) as demo:
        
        gr.Markdown("""
        <div class="header">
            <h1 style="margin: 0; font-size: 2.2em;">🎵 Saem's Tunes AI Assistant</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.1em; opacity: 0.9;">
                Powered by TinyLlama 1.1B • Built for music education and streaming
            </p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                status_display = gr.HTML(
                    value="<div class='status-indicator status-warning'></div>Initializing systems..."
                )
            with gr.Column(scale=1):
                refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
        
        gr.Markdown("### 💡 Quick Questions")
        
        quick_questions = [
            "How do I create a playlist?",
            "What are the premium features?",
            "How do I upload my music?",
            "Tell me about music courses",
            "How does the recommendation system work?"
        ]
        
        quick_buttons = []
        with gr.Row():
            for question in quick_questions:
                btn = gr.Button(question, size="sm", elem_classes="quick-action-btn")
                quick_buttons.append(btn)
        
        gr.Markdown("### 💬 Chat with Saem's Tunes AI")
        
        chatbot = gr.Chatbot(
            label="Saem's Tunes Chat",
            height=450,
            placeholder="Ask me anything about Saem's Tunes music platform...",
            show_label=False
        )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type your question here... (Press Enter to send)",
                show_label=False,
                scale=4,
                container=False,
                lines=2
            )
            submit_btn = gr.Button("Send 🚀", variant="primary", scale=1)
        
        gr.Examples(
            examples=[
                "How do I create a playlist?",
                "What are the premium features?",
                "How do I upload my music as an artist?",
                "Tell me about the music courses available",
                "How does the recommendation system work?"
            ],
            inputs=msg,
            label="💡 Example Questions"
        )
        
        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
        
        gr.Markdown("""
        <div class="footer">
            <p>
                <strong>Powered by TinyLlama 1.1B Chat</strong> • 
                <a href="https://www.saemstunes.com" target="_blank">Saem's Tunes Music Platform</a>
            </p>
            <p style="font-size: 0.9em; opacity: 0.7;">
                Model: Q4_K_M quantization • Context: 2K tokens
            </p>
        </div>
        """)
        
        def update_status():
            """Update status display - lightweight"""
            status = get_system_status()
            status_text = status.get("status", "unknown")
            
            if status_text == "healthy":
                html = """
                <div class='status-indicator status-healthy'></div>
                <strong>System Status: Healthy</strong><br>
                <small>AI Assistant is ready to help!</small>
                """
            elif status_text == "initializing":
                duration = status.get('duration_seconds', 0)
                html = f"""
                <div class='status-indicator status-warning'></div>
                <strong>System Status: Initializing</strong><br>
                <small>Loading AI model... ({duration:.0f}s)</small>
                """
            else:
                html = f"<div class='status-indicator status-error'></div>System Status: {status_text}"
            
            return html
        
        def user_message(user_message, chat_history):
            return "", chat_history + [[user_message, None]]
        
        def bot_response(chat_history):
            if not chat_history:
                return chat_history
            
            user_message = chat_history[-1][0]
            bot_message = chat_interface(user_message, chat_history, gr.Request())
            
            chat_history[-1][1] = bot_message
            return chat_history
        
        def clear_chat():
            return []
        
        # Connect components
        refresh_btn.click(update_status, outputs=status_display)
        
        msg.submit(
            user_message, [msg, chatbot], [msg, chatbot], queue=False
        ).then(
            bot_response, chatbot, chatbot
        )
        
        submit_btn.click(
            user_message, [msg, chatbot], [msg, chatbot], queue=False
        ).then(
            bot_response, chatbot, chatbot
        )
        
        clear_btn.click(clear_chat, outputs=chatbot)
        
        # Connect quick buttons
        for btn in quick_buttons:
            btn.click(
                lambda x=btn.value: x,
                outputs=msg
            ).then(
                user_message, [msg, chatbot], [msg, chatbot]
            ).then(
                bot_response, chatbot, chatbot
            )
        
        demo.load(update_status, outputs=status_display)
    
    return demo

# Create Gradio interface and mount to FastAPI - AT MODULE LEVEL
demo = create_gradio_interface()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

# Start initialization AFTER app is created
start_initialization()

logger.info(f"🚀 Saem's Tunes AI Assistant starting on port {Config.PORT}")
logger.info("📡 FastAPI and Gradio apps mounted successfully")
logger.info("🔄 System initialization started in background")

# Hugging Face Spaces entry point
if __name__ == "__main__":
    # This runs when developing locally
    demo.launch(
        server_name="0.0.0.0",
        server_port=Config.PORT,
        show_error=True,
        share=False
    )
