import os
import gradio as gr
import json
import time
import logging
import psutil
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
import requests
from dotenv import load_dotenv

load_dotenv()

from src.ai_system import SaemsTunesAISystem
from src.supabase_integration import AdvancedSupabaseIntegration
from src.security_system import AdvancedSecuritySystem
from src.monitoring_system import ComprehensiveMonitor

class Config:
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/Phi-3.5-mini-instruct")
    HF_SPACE = os.getenv("HF_SPACE", "saemstunes/STA-AI")
    PORT = int(os.getenv("PORT", 7860))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MAX_RESPONSE_LENGTH = int(os.getenv("MAX_RESPONSE_LENGTH", "500"))
    ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "true").lower() == "true"

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('saems_ai.log')
    ]
)
logger = logging.getLogger(__name__)

supabase_integration = None
security_system = None
monitor = None
ai_system = None

def initialize_systems():
    global supabase_integration, security_system, monitor, ai_system
    
    logger.info("🚀 Initializing Saem's Tunes AI System...")
    
    try:
        supabase_integration = AdvancedSupabaseIntegration(
            Config.SUPABASE_URL, 
            Config.SUPABASE_ANON_KEY
        )
        logger.info("✅ Supabase integration initialized")
        
        security_system = AdvancedSecuritySystem()
        logger.info("✅ Security system initialized")
        
        monitor = ComprehensiveMonitor()
        logger.info("✅ Monitoring system initialized")
        
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

def chat_interface(message: str, history: List[List[str]], request: gr.Request) -> str:
    try:
        if not message.strip():
            return "Please ask me anything about Saem's Tunes!"
        
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
    if not all([supabase_integration, security_system, monitor, ai_system]):
        return {"status": "initializing", "details": "Systems are starting up..."}
    
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "systems": {
                "supabase": supabase_integration.is_connected(),
                "security": True,
                "monitoring": True,
                "ai_system": ai_system.is_healthy()
            },
            "resources": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            },
            "performance": {
                "total_requests": len(monitor.inference_metrics),
                "avg_response_time": monitor.get_average_response_time(),
                "error_rate": monitor.get_error_rate()
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

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
                Powered by Microsoft Phi-3.5-mini-instruct • Built for music education and streaming
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
                <strong>Powered by Microsoft Phi-3.5-mini-instruct</strong> • 
                <a href="https://www.saemstunes.com" target="_blank">Saem's Tunes Music Platform</a>
            </p>
            <p style="font-size: 0.9em; opacity: 0.7;">
                Model: Q4_K_M quantization • Context: 4K tokens • Response time: ~2-5s
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
                    CPU: {resources.get('cpu_percent', 0):.1f}% |
                    Memory: {resources.get('memory_percent', 0):.1f}%
                </small>
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

def setup_api_endpoints(demo):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import Optional
    
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
    
    @demo.app.post("/api/chat")
    async def api_chat(request: ChatRequest):
        try:
            if not request.message.strip():
                raise HTTPException(status_code=400, detail="Message cannot be empty")
            
            security_result = security_system.check_request(request.message, request.user_id)
            if security_result["is_suspicious"]:
                raise HTTPException(status_code=429, detail="Request blocked for security reasons")
            
            start_time = time.time()
            response = ai_system.process_query(request.message, request.user_id, request.conversation_id)
            processing_time = time.time() - start_time
            
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
            logger.error(f"API chat error: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    @demo.app.get("/api/health")
    async def api_health():
        return get_system_status()
    
    @demo.app.get("/api/models")
    async def api_models():
        return {
            "available_models": ["microsoft/Phi-3.5-mini-instruct"],
            "current_model": Config.MODEL_NAME,
            "quantization": "Q4_K_M",
            "context_length": 4096,
            "parameters": "3.8B"
        }
    
    @demo.app.get("/api/stats")
    async def api_stats():
        if not monitor:
            return {"error": "Monitoring system not available"}
        
        return {
            "total_requests": len(monitor.inference_metrics),
            "average_response_time": monitor.get_average_response_time(),
            "error_rate": monitor.get_error_rate(),
            "uptime": monitor.get_uptime(),
            "system_health": get_system_status()
        }

if __name__ == "__main__":
    logger.info("🎵 Starting Saem's Tunes AI on Hugging Face Spaces...")
    
    if not initialize_systems():
        logger.error("Failed to initialize systems. Exiting.")
        exit(1)
    
    demo = create_gradio_interface()
    
    setup_api_endpoints(demo)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=Config.PORT,
        share=False,
        show_error=True,
        debug=Config.LOG_LEVEL == "DEBUG",
        show_api=True,
        allowed_paths=["./models", "./config"]
    )