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

from src.ai_system import SaemsTunesAISystem
from src.supabase_integration import AdvancedSupabaseIntegration
from src.security_system import AdvancedSecuritySystem
from src.monitoring_system import ComprehensiveMonitor

class Config:
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    MODEL_REPO = os.getenv("MODEL_REPO", "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    MODEL_FILE = os.getenv("MODEL_FILE", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    HF_SPACE = os.getenv("HF_SPACE", "saemstunes/STA-AI")
    PORT = int(os.getenv("PORT", 8000))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MAX_RESPONSE_LENGTH = int(os.getenv("MAX_RESPONSE_LENGTH", "500"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_P = float(os.getenv("TOP_P", "0.9"))
    CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "2048"))
    ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "true").lower() == "true"

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Only StreamHandler for Hugging Face Spaces
)
logger = logging.getLogger(__name__)

supabase_integration = None
security_system = None
monitor = None
ai_system = None
systems_ready = False
initialization_complete = False
initialization_errors = []
initialization_start_time = None

def initialize_systems():
    global supabase_integration, security_system, monitor, ai_system, systems_ready, initialization_complete, initialization_errors
    
    logger.info("🚀 Initializing Saem's Tunes AI System...")
    
    try:
        supabase_integration = AdvancedSupabaseIntegration(
            Config.SUPABASE_URL, 
            Config.SUPABASE_ANON_KEY
        )
        logger.info("✅ Supabase integration initialized")
        
        security_system = AdvancedSecuritySystem()
        logger.info("✅ Security system initialized")
        
        monitor = ComprehensiveMonitor(prometheus_port=8001)
        logger.info("✅ Monitoring system initialized")
        
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
        
        if ai_system.is_healthy():
            systems_ready = True
            initialization_complete = True
            logger.info("🎉 All systems initialized successfully!")
        else:
            initialization_errors.append("AI system health check failed")
            initialization_complete = True
        
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

def chat_interface(message: str, history: List[List[str]], request: gr.Request) -> str:
    try:
        if not message.strip():
            return "Please ask me anything about Saem's Tunes!"
        
        if not systems_ready:
            return "🔄 Systems are still initializing. Please wait a moment and try again..."
        
        client_host = getattr(request, "client", None)
        if client_host:
            user_ip = client_host.host
        else:
            user_ip = "unknown"
        user_id = f"gradio_user_{user_ip}"
        
        security_result = security_system.check_request(message, user_id)
        if security_result["is_suspicious"]:
            logger.warning(f"Suspicious request blocked from {user_ip}: {message}")
            return "Your request has been blocked for security reasons. Please try a different question."
        
        start_time = time.time()
        response = ai_system.process_query(message, user_id)
        processing_time = time.time() - start_time
        
        formatted_response = f"{response}\n\n_Generated in {processing_time:.1f}s_"
        
        logger.info(f"Chat processed: {message[:50]}... -> {processing_time:.2f}s")
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return "I apologize, but I'm experiencing technical difficulties. Please try again later."

def get_system_status() -> Dict[str, Any]:
    if not initialization_complete:
        return {
            "status": "initializing", 
            "details": "Systems are starting up...",
            "timestamp": datetime.now().isoformat(),
            "initialization_started": initialization_start_time is not None,
            "duration_seconds": time.time() - initialization_start_time if initialization_start_time else 0
        }
    
    if not systems_ready:
        return {
            "status": "degraded",
            "details": "Systems initialized but not fully ready",
            "errors": initialization_errors,
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "systems": {
                "supabase": supabase_integration.is_connected() if supabase_integration else False,
                "security": bool(security_system),
                "monitoring": bool(monitor),
                "ai_system": ai_system.is_healthy() if ai_system else False,
                "model_loaded": ai_system.model_loaded if ai_system else False
            },
            "resources": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            },
            "performance": {
                "total_requests": len(monitor.inference_metrics) if monitor else 0,
                "avg_response_time": monitor.get_average_response_time() if monitor else 0,
                "error_rate": monitor.get_error_rate() if monitor else 0
            }
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

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

# Create FastAPI app at module level - REQUIRED FOR HUGGING FACE
fastapi_app = FastAPI(title="Saem's Tunes AI API", version="2.0.0")

# Add root route - REQUIRED FOR HUGGING FACE HEALTH CHECKS
@fastapi_app.get("/")
def root():
    """Root endpoint for Hugging Face health checks"""
    return {
        "status": "healthy" if systems_ready else "initializing",
        "message": "Saem's Tunes AI API is running",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "environment": "huggingface-spaces"
    }

@fastapi_app.get("/api/health")
def api_health():
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
    if not monitor or not systems_ready:
        return JSONResponse(
            content={
                "status": "initializing" if not systems_ready else "degraded",
                "systems_ready": systems_ready,
                "timestamp": datetime.now().isoformat()
            },
            status_code=200  # Always return 200 for Hugging Face
        )
    
    stats_data = {
        "status": "healthy",
        "total_requests": len(monitor.inference_metrics),
        "average_response_time": monitor.get_average_response_time(),
        "error_rate": monitor.get_error_rate(),
        "uptime": monitor.get_uptime(),
        "system_health": get_system_status(),
        "timestamp": datetime.now().isoformat()
    }
    return stats_data

@fastapi_app.post("/api/chat")
def api_chat(request: ChatRequest):
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if not systems_ready:
            raise HTTPException(
                status_code=503, 
                detail="Systems are still initializing. Please try again in a moment."
            )
        
        security_result = security_system.check_request(request.message, request.user_id)
        if security_result["is_suspicious"]:
            raise HTTPException(status_code=429, detail="Request blocked for security reasons")
        
        start_time = time.time()
        response = ai_system.process_query(request.message, request.user_id, request.conversation_id)
        processing_time = time.time() - start_time
        
        return {
            "response": response,
            "processing_time": processing_time,
            "conversation_id": request.conversation_id or f"conv_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "model_used": Config.MODEL_NAME
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

def create_gradio_interface():
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
                quick_buttons.append((btn, question))
        
        gr.Markdown("### 💬 Chat with Saem's Tunes AI")
        
        chatbot = gr.Chatbot(
            label="Saem's Tunes Chat",
            height=450,
            placeholder="Ask me anything about Saem's Tunes music platform...",
            show_label=False,
            type="messages"
        )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type your question here... (Press Enter to send, Shift+Enter for new line)",
                show_label=False,
                scale=4,
                container=False,
                lines=2,
                max_lines=4
            )
            submit_btn = gr.Button("Send 🚀", variant="primary", scale=1, size="lg")
        
        gr.Examples(
            examples=[
                "How do I create a playlist?",
                "What are the premium features?",
                "How do I upload my music as an artist?",
                "Tell me about the music courses available",
                "How does the recommendation system work?",
                "Can I share playlists with friends?",
                "What music genres are available?",
                "How do I follow artists?",
                "Is there a mobile app?",
                "How do I reset my password?"
            ],
            inputs=msg,
            label="💡 Example Questions"
        )
        
        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
            export_btn = gr.Button("💾 Export Chat", size="sm")
        
        gr.Markdown("""
        <div class="footer">
            <p>
                <strong>Powered by TinyLlama 1.1B Chat</strong> • 
                <a href="https://www.saemstunes.com" target="_blank">Saem's Tunes Music Platform</a>
            </p>
            <p style="font-size: 0.9em; opacity: 0.7;">
                Model: Q4_K_M quantization • Context: 2K tokens • Response time: ~2-5s
            </p>
        </div>
        """)
        
        def update_status():
            status = get_system_status()
            status_text = status.get("status", "unknown")
            status_class = f"status-{status_text}" if status_text in ["healthy", "warning", "error"] else "status-warning"
            
            if status_text == "healthy":
                systems = status.get("systems", {})
                resources = status.get("resources", {})
                html = f"""
                <div class='status-indicator {status_class}'></div>
                <strong>System Status: Healthy</strong><br>
                <small>
                    Supabase: {'✅' if systems.get('supabase') else '❌'} |
                    AI System: {'✅' if systems.get('ai_system') else '❌'} |
                    Model: {'✅' if systems.get('model_loaded') else '❌'} |
                    CPU: {resources.get('cpu_percent', 0):.1f}% |
                    Memory: {resources.get('memory_percent', 0):.1f}%
                </small>
                """
            elif status_text == "initializing":
                duration = status.get('duration_seconds', 0)
                html = f"""
                <div class='status-indicator {status_class}'></div>
                <strong>System Status: Initializing</strong><br>
                <small>Started {duration:.0f}s ago • Downloading AI model...</small>
                """
            else:
                html = f"<div class='status-indicator {status_class}'></div>{status.get('details', 'Unknown status')}"
            
            return html
        
        def user_message(user_message, chat_history):
            return "", chat_history + [{"role": "user", "content": user_message}]
        
        def bot_response(chat_history):
            if not chat_history:
                return chat_history
            
            user_message = chat_history[-1]["content"]
            bot_message = chat_interface(user_message, chat_history, gr.Request())
            
            return chat_history + [{"role": "assistant", "content": bot_message}]
        
        def clear_chat():
            return []
        
        def export_chat(chat_history):
            if not chat_history:
                return "No conversation to export"
            
            export_text = f"Saem's Tunes AI Conversation - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            for msg in chat_history:
                role = "You" if msg["role"] == "user" else "AI Assistant"
                export_text += f"{role}: {msg['content']}\n\n"
            
            return export_text
        
        refresh_btn.click(update_status, outputs=status_display)
        
        msg.submit(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, chatbot, chatbot
        )
        
        submit_btn.click(user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_response, chatbot, chatbot
        )
        
        clear_btn.click(clear_chat, outputs=chatbot)
        export_btn.click(export_chat, chatbot, msg)
        
        for btn, question_text in quick_buttons:
            btn.click(
                fn=lambda q=question_text: q,
                outputs=msg
            ).then(
                user_message, [msg, chatbot], [msg, chatbot]
            ).then(
                bot_response, chatbot, chatbot
            )
        
        demo.load(update_status, outputs=status_display)
    
    return demo

# Create Gradio interface and mount to FastAPI - AT MODULE LEVEL FOR HUGGING FACE
demo = create_gradio_interface()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

# Start background initialization
initialize_systems_background()

if __name__ == "__main__":
    logger.info("🎵 Starting Saem's Tunes AI on Hugging Face Spaces...")
    
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=Config.PORT,
        log_level=Config.LOG_LEVEL.lower()
    )
