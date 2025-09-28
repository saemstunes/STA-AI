#!/bin/bash

set -e  # Exit on any error

echo "🚀 Deploying Saem's Tunes AI to Hugging Face Spaces"
echo "==================================================="

# Configuration
SPACE_NAME="saemstunes/STA-AI"
MODEL_REPO="Thetima4/Phi-3.5-mini-instruct-Q4_K_M-GGUF"
MODEL_FILE="Phi-3.5-mini-instruct-q4_k_m.gguf"
VERSION="2.0.0"

echo "📦 Preparing deployment for space: $SPACE_NAME"
echo "🔖 Version: $VERSION"
echo "🤖 Model: $MODEL_REPO"

# Check if required files exist
echo "🔍 Checking required files..."
required_files=("app.py" "requirements.txt" "config/spaces_config.json" "src/")
for file in "${required_files[@]}"; do
    if [ ! -e "$file" ]; then
        echo "❌ Missing required file/directory: $file"
        exit 1
    fi
done

echo "✅ All required files present"

# Create models directory if it doesn't exist
mkdir -p models

# Create comprehensive README for the space
echo "📝 Creating README.md..."
cat > README.md << 'EOF'
# 🎵 Saem's Tunes AI Assistant

Production-ready AI assistant for Saem's Tunes music education and streaming platform.

## 🚀 Features

- **Smart Music Assistance**: Help with platform features, music discovery, and educational content
- **Fast Responses**: Powered by Microsoft Phi-3.5-mini-instruct (Q4_K_M quantization)
- **Platform Integration**: Full integration with Saem's Tunes existing database
- **Secure**: Input sanitization, rate limiting, and suspicious activity detection
- **Real-time Monitoring**: Performance metrics and health checks
- **Dual Deployment**: Hugging Face Spaces (primary) + Railway (backup)

## 📚 API Usage

### Chat Endpoint
```python
import requests

response = requests.post(
    "https://saemstunes-sta-ai.hf.space/api/chat",
    json={
        "message": "How do I create a playlist?",
        "user_id": "user_123",
        "conversation_id": "conv_123"
    }
)
Health Check
python
import requests

response = requests.get("https://saemstunes-sta-ai.hf.space/api/health")
print(response.json())
🛠️ For Developers
Environment Variables
SUPABASE_URL: Your Supabase project URL

SUPABASE_ANON_KEY: Your Supabase anon key

MODEL_NAME: Phi-3.5-mini-instruct (default)

LOG_LEVEL: INFO (default)

MAX_RESPONSE_LENGTH: 500 (default)

Local Development
bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py

# Run tests
pytest tests/
🎯 Platform Features Supported
The AI assistant understands and can help with:

Music Streaming: Tracks, artists, playlists, recommendations

Education: Courses, lessons, quizzes, learning paths

Community: User profiles, favorites, follows, social features

Creator Tools: Music upload, artist profiles, analytics

E-commerce: Subscriptions, payments, premium content

Technical Support: Platform help, troubleshooting

🔧 Technical Details
Base Model: microsoft/Phi-3.5-mini-instruct

Quantization: Q4_K_M (optimal quality/speed balance)

Context Window: 4K tokens

Response Time: 2-5 seconds

Security: Rate limiting (60 requests/minute), input sanitization

Monitoring: Real-time metrics, health checks, error tracking

📊 Monitoring
Real-time metrics at /metrics

Health checks at /health

Performance statistics at /api/stats

🆘 Support
Documentation: This README and inline code documentation

Issues: GitHub issue tracker

Contact: development@saemstunes.com

Built with ❤️ for Saem's Tunes - Your complete music education and streaming platform

Version: 2.0.0
Last Updated: $(date +%Y-%m-%d)
EOF

echo "✅ README.md created"

Create .env.example for environment variables
echo "🔧 Creating environment template..."
cat > .env.example << 'EOF'

Saem's Tunes AI Configuration
Copy this file to .env and fill in your actual values
Supabase Configuration
SUPABASE_URL=your_supabase_project_url_here
SUPABASE_ANON_KEY=your_supabase_anon_key_here

Model Configuration
MODEL_NAME=microsoft/Phi-3.5-mini-instruct
HF_SPACE=saemstunes/STA-AI

Server Configuration
PORT=7860
LOG_LEVEL=INFO
MAX_RESPONSE_LENGTH=500

Monitoring
ENABLE_MONITORING=true
EOF

echo "✅ Environment template created"

Create deployment info file
echo "📋 Creating deployment info..."
cat > deployment-info.md << EOF

Deployment Information
Space Details
Name: $SPACE_NAME

Version: $VERSION

Model: $MODEL_REPO

Deployment Date: $(date)

Required Setup
Set the following environment variables in your Space settings:

SUPABASE_URL

SUPABASE_ANON_KEY

File Structure
```
$(find . -type f -name ".py" -o -name ".json" -o -name ".md" -o -name ".txt" -o -name "*.yml" | head -20 | sed 's|^./||' | sort)
```

API Endpoints
`/api/chat` - Main chat endpoint

`/api/health` - Health check

`/api/models` - Model information

`/api/stats` - System statistics

`/api/feedback` - User feedback

Monitoring
Health checks run every 30 seconds

Metrics available at `/metrics`

Logs are stored in `saems_ai.log`
EOF

echo "✅ Deployment info created"

Create verification script
echo "🔍 Creating verification script..."
cat > verify_deployment.py << 'EOF'
#!/usr/bin/env python3

import os
import sys

def check_file_exists(filepath):
"""Check if file exists"""
if os.path.exists(filepath):
print(f"✅ {filepath}")
return True
else:
print(f"❌ {filepath}")
return False

def main():
print("🔍 Verifying deployment setup...")

text
required_files = [
    "app.py",
    "requirements.txt",
    "README.md",
    "config/spaces_config.json",
    "src/__init__.py",
    "src/ai_system.py",
    "src/supabase_integration.py",
    "src/security_system.py",
    "src/monitoring_system.py",
    "src/utils.py"
]

all_exists = True
for filepath in required_files:
    if not check_file_exists(filepath):
        all_exists = False

if all_exists:
    print("\n🎉 All required files are present!")
    print("📦 Deployment package is ready for Hugging Face Spaces")
else:
    print("\n❌ Some required files are missing!")
    sys.exit(1)
if name == "main":
main()
EOF

python verify_deployment.py

echo ""
echo "📋 Deployment checklist:"
echo " ✅ app.py (main application)"
echo " ✅ requirements.txt (dependencies)"
echo " ✅ README.md (documentation)"
echo " ✅ config/ (configuration files)"
echo " ✅ src/ (source code)"
echo " ✅ scripts/ (utility scripts)"
echo " ✅ .env.example (environment template)"

echo ""
echo "🎯 Deployment Instructions:"
echo "1. Go to https://huggingface.co/spaces"
echo "2. Create new Space or update existing: $SPACE_NAME"
echo "3. Set SDK to Gradio"
echo "4. Upload all files from this directory"
echo "5. Set environment variables:"
echo " - SUPABASE_URL"
echo " - SUPABASE_ANON_KEY"
echo "6. Deploy and wait for build to complete"
echo "7. Test the API endpoints"
echo ""
echo "🔗 Your Space will be available at:"
echo " https://huggingface.co/spaces/$SPACE_NAME"
echo ""
echo "✅ Deployment preparation completed!"