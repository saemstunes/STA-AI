#!/bin/bash
echo "🚀 DEPLOYING SAEM'S TUNES AI TO HUGGING FACE SPACES"

# Configuration
SPACE_NAME="saemstunes/STA-AI"
MODEL_REPO="Thetima4/Phi-3.5-mini-instruct-Q4_K_M-GGUF"
MODEL_FILE="Phi-3.5-mini-instruct-q4_k_m.gguf"

# Create necessary directories
mkdir -p models
mkdir -p logs
mkdir -p config

# Download the pre-quantized model
echo "📥 DOWNLOADING PRE-QUANTIZED MODEL"
wget -O models/phi3.5-mini.Q4_K_M.gguf \
    "https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"

# Create configuration files
cat > config/spaces_config.json << EOF
{
    "space_name": "STA-AI",
    "model_used": "Phi-3.5-mini-instruct-Q4_K_M",
    "deployment_date": "$(date -I)",
    "features": [
        "multi-model-support",
        "performance-monitoring",
        "supabase-integration",
        "advanced-prompting",
        "real-time-chat"
    ]
}
EOF

# Create README for the space
cat > README.md << 'EOF'
# 🎵 Saem's Tunes AI Assistant Pro

Advanced AI-powered assistant for Saem's Tunes music platform, built with Microsoft Phi-3.5-mini-instruct.

## Features

- **Multiple Quantization Support**: Q4_K_M, Q5_K_M, Q8_0 models
- **Real-time Performance Monitoring**: Track response times and system metrics
- **Supabase Integration**: Live music platform context
- **Advanced Prompt Engineering**: Context-aware responses
- **Production Ready**: Error handling, logging, monitoring

## Model Information

- **Base Model**: microsoft/Phi-3.5-mini-instruct
- **Quantization**: Q4_K_M (Optimal balance of quality/speed)
- **Context Window**: 4K tokens
- **Specialization**: Music platform assistance

## Usage

Ask about:
- Platform features and capabilities
- Artist and song information  
- Technical support
- Premium features
- Music discovery

## Technical Details

- Built with Gradio for web interface
- llama.cpp for efficient inference
- Comprehensive monitoring and logging
- Designed for Hugging Face Spaces deployment

---
*Powered by Microsoft Phi-3.5-mini-instruct | Built for [Saem's Tunes](https://www.saemstunes.com)*
EOF

echo "✅ HUGGING FACE SPACES DEPLOYMENT CONFIGURED"
echo "📁 Files prepared:"
echo "   - app.py (Main application)"
echo "   - requirements.txt (Dependencies)" 
echo "   - models/ (Quantized model files)"
echo "   - README.md (Documentation)"
echo "   - config/spaces_config.json (Configuration)"

# Instructions for manual deployment
echo ""
echo "📋 DEPLOYMENT INSTRUCTIONS:"
echo "1. Go to https://huggingface.co/spaces"
echo "2. Create new Space with SDK: Gradio"
echo "3. Upload all files from this directory"
echo "4. Set environment variables:"
echo "   - SUPABASE_URL=your_supabase_url"
echo "   - SUPABASE_ANON_KEY=your_anon_key"
echo "5. Deploy and test!"
