
#!/bin/bash

set -e  # Exit on any error

echo "🔧 Setting up Saem's Tunes AI development environment"
echo "===================================================="

# Check Python version
echo "🐍 Checking Python version..."
python --version || { echo "❌ Python not found"; exit 1; }

# Create virtual environment
echo "📦 Creating Python virtual environment..."
python -m venv venv
source venv/bin/activate

# Upgrade pip
echo "📥 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt
pip install -r requirements-dev.txt

echo "✅ Dependencies installed"

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p models logs config tests/monitoring static

# Setup pre-commit hooks
echo "🔨 Setting up pre-commit hooks..."
pre-commit install || echo "⚠️  pre-commit not available, skipping"

# Create environment template if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating environment file..."
    cp .env.example .env
    echo "⚠️  Please update .env with your actual values"
fi

# Create test configuration
echo "🧪 Creating test configuration..."
cat > tests/test_config.py << 'EOF'
"""Test configuration for Saem's Tunes AI"""

TEST_CONFIG = {
    "supabase": {
        "url": "https://test.supabase.co",
        "key": "test_key"
    },
    "model": {
        "name": "microsoft/Phi-3.5-mini-instruct",
        "quantization": "Q4_K_M"
    },
    "security": {
        "rate_limit": 60,
        "max_input_length": 10000
    }
}
EOF

# Create development startup script
echo "🚀 Creating development startup script..."
cat > run_dev.sh << 'EOF'
#!/bin/bash

source venv/bin/activate
export LOG_LEVEL=DEBUG
export ENVIRONMENT=development
python app.py
EOF

chmod +x run_dev.sh

# Create production startup script
echo "🏭 Creating production startup script..."
cat > run_prod.sh << 'EOF'
#!/bin/bash

source venv/bin/activate
export LOG_LEVEL=INFO
export ENVIRONMENT=production
python railway_app.py
EOF

chmod +x run_prod.sh

# Download model script
echo "📥 Creating model download script..."
cat > download_model.py << 'EOF'
#!/usr/bin/env python3

import os
from huggingface_hub import hf_hub_download
import argparse

def main():
    parser = argparse.ArgumentParser(description="Download Phi-3.5-mini-instruct model")
    parser.add_argument("--repo", default="Thetima4/Phi-3.5-mini-instruct-Q4_K_M-GGUF",
                       help="Hugging Face repository ID")
    parser.add_argument("--file", default="Phi-3.5-mini-instruct-q4_k_m.gguf",
                       help="Model filename")
    parser.add_argument("--dir", default="./models",
                       help="Local directory to save the model")
    
    args = parser.parse_args()
    
    print(f"📥 Downloading model from {args.repo}")
    print(f"📄 Model file: {args.file}")
    print(f"💾 Local directory: {args.dir}")
    
    # Create local directory if it doesn't exist
    os.makedirs(args.dir, exist_ok=True)
    
    try:
        # Download the model
        model_path = hf_hub_download(
            repo_id=args.repo,
            filename=args.file,
            local_dir=args.dir,
            local_dir_use_symlinks=False
        )
        
        print(f"✅ Model downloaded successfully: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return None

if __name__ == "__main__":
    main()
EOF

chmod +x download_model.py

echo "✅ Environment setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Update .env with your actual environment variables"
echo "2. Run: source venv/bin/activate"
echo "3. Download model: python download_model.py"
echo "4. Test the system: python app.py"
echo "5. Run tests: pytest tests/"
echo ""
echo "🎯 Development commands:"
echo "   ./run_dev.sh    # Start development server"
echo "   ./run_prod.sh   # Start production server"
echo "   pytest tests/   # Run test suite"
echo "   black src/      # Format code"
echo "   flake8 src/     # Lint code"
echo ""
echo "🚀 Happy coding!"