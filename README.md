---
title: STA AI
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.47.2
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
license: mit
short_description: 'Lightweight Saem’s Tunes assistant — Phi-3.5-mini-instruct '
---

# 🎵 Saem's Tunes AI Assistant

Advanced AI-powered assistant for Saem's Tunes music platform, built with Microsoft Phi-3.5-mini-instruct and comprehensive monitoring.

## 🌟 Features

- **Smart FAQ System**: AI-powered responses with contextual understanding
- **Multi-Platform Deployment**: Hugging Face Spaces, Railway, and local deployment
- **Continuous Learning**: Improves over time with user feedback
- **Advanced RAG**: Semantic search through your music database
- **Real-time Monitoring**: Comprehensive performance analytics
- **Production Ready**: Security, rate limiting, and error handling

## 🚀 Quick Start

### Option 1: Hugging Face Spaces (Recommended - Free)
1. **Create a Space** at [huggingface.co/spaces](https://huggingface.co/spaces)
2. **Upload these files** to your Space:
   - `app.py` (main application)
   - `requirements.txt` (dependencies)
   - `models/phi3.5-mini.Q4_K_M.gguf` (download instructions below)
3. **Set environment variables** in Space settings:
   - `SUPABASE_URL`: Your Supabase project URL
   - `SUPABASE_ANON_KEY`: Your Supabase anon key
4. **Deploy** and your AI assistant will be live!

### Option 2: Railway Deployment
1. **Connect your GitHub repo** to [Railway](https://railway.app)
2. **Set environment variables** in Railway dashboard
3. **Deploy automatically** from your repository

### Option 3: Local Development
```bash`
# Clone and setup
git clone <your-repo>
cd saems-tunes-ai

# Install dependencies
pip install -r requirements.txt

# Download the model
mkdir -p models
cd models
wget https://huggingface.co/Thetima4/Phi-3.5-mini-instruct-Q4_K_M-GGUF/resolve/main/Phi-3.5-mini-instruct-q4_k_m.gguf

# Run locally
python app.py
📦 Model Download
The system uses Microsoft Phi-3.5-mini-instruct quantized to Q4_K_M for optimal performance.

# Download Command:
bash
wget -O models/phi3.5-mini.Q4_K_M.gguf \
    "https://huggingface.co/Thetima4/Phi-3.5-mini-instruct-Q4_K_M-GGUF/resolve/main/Phi-3.5-mini-instruct-q4_k_m.gguf"
Alternative Models:
Q4_0: Faster, slightly lower quality

Q5_K_M: Better quality, larger size

Q8_0: Best quality, largest size

🔧 Configuration
# Environment Variables:
bash
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key
HF_SPACE_URL=your_huggingface_space_url
MODEL_PATH=./models/phi3.5-mini.Q4_K_M.gguf
# Supabase Schema:
Your database should include these tables (see supabase_schema.sql):

songs - Music catalog

artists - Artist information

users - User profiles

ai_interactions - AI conversation logging

🎯 Integration with Your React App
Add the AI component to your existing React app:

javascript
// In your main App.js
import SaemsTunesAI from './components/SaemsTunesAI';

function App() {
  return (
    <div className="App">
      {/* Your existing components */}
      <SaemsTunesAI />
    </div>
  );
}
📊 Monitoring & Analytics
The system includes comprehensive monitoring:

Real-time Dashboard: Streamlit-based analytics

Performance Metrics: Response times, error rates, token usage

Alert System: Email/Slack notifications for issues

Usage Analytics: User behavior and model performance

Access the dashboard at /dashboard when running locally.

🔒 Security Features
Rate Limiting: Prevents API abuse

Input Sanitization: Protects against injection attacks

Audit Logging: Tracks all user interactions

Content Filtering: Detects suspicious queries

🔄 Continuous Learning
The system improves over time by:

Collecting feedback from user interactions

Fine-tuning on successful conversations

Automated model updates without downtime

🏗️ Architecture
text
Frontend (React) → AI API (FastAPI) → Phi-3.5 Model → Supabase Database
                     ↑
              Monitoring & Analytics
Components:
Frontend: React component with chat interface

Backend: FastAPI server with model inference

Database: Supabase for music data and analytics

Monitoring: Comprehensive metrics and alerts

🚨 Troubleshooting
Common Issues:
Model not loading:

Verify the model file exists in models/

Check file permissions

Ensure enough RAM (4GB+ recommended)

Supabase connection issues:

Verify environment variables

Check Supabase project status

Test database connection

High response times:

Use smaller quantization (Q4_0 instead of Q8_0)

Increase allocated resources

Enable GPU acceleration if available

Getting Help:
Check the Hugging Face discussion forum

Open an issue in this repository

Contact the Saem's Tunes development team

📈 Performance Benchmarks
Model	Size	Response Time	Quality	Use Case
Q4_K_M	2.4GB	1-3s	Excellent	Production
Q4_0	2.2GB	1-2s	Very Good	Fast responses
Q8_0	4.2GB	3-5s	Best	Maximum quality
🔮 Future Enhancements
Voice interface integration

Mobile app companion

Advanced music recommendation engine

Multi-language support (Swahili focus)

Band collaboration features

👥 Contributing
We welcome contributions! Please see:

Code of Conduct

Contributing Guidelines

Issue Templates

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Microsoft for the Phi-3.5 model

Hugging Face for model hosting and Spaces

Supabase for the database backend

Railway for deployment infrastructure

Built with ❤️ for the Saem's Tunes community

Visit Saem's Tunes | Report an Issue