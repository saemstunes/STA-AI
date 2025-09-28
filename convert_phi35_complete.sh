#!/bin/bash
set -e

echo "🚀 COMPREHENSIVE PHI-3.5-MINI CONVERSION & QUANTIZATION SYSTEM"
echo "================================================================"

# Configuration
MODEL_NAME="microsoft/Phi-3.5-mini-instruct"
OUTPUT_DIR="./phi3.5-mini-gguf-complete"
QUANT_TYPES=("Q2_K" "Q3_K_S" "Q3_K_M" "Q3_K_L" "Q4_0" "Q4_1" "Q4_K_S" "Q4_K_M" "Q5_0" "Q5_1" "Q5_K_S" "Q5_K_M" "Q6_K" "Q8_0" "F16" "F32")

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/artifacts
mkdir -p $OUTPUT_DIR/logs

# Install system dependencies
echo "📦 INSTALLING SYSTEM DEPENDENCIES"
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-venv \
    libcurl4-openssl-dev \
    git-lfs \
    wget \
    curl \
    pkg-config \
    libopenblas-dev \
    libatomic-dev

# Initialize Git LFS
git lfs install

# Create Python virtual environment
python3 -m venv phi3-convert
source phi3-convert/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers>=4.43.0
pip install safetensors
pip install huggingface-hub
pip install accelerate
pip install protobuf
pip install sentencepiece
pip install ninja
pip install requests
pip install tqdm
pip install psutil
pip install GPUtil

# Clone and build llama.cpp with all optimizations
echo "🔨 BUILDING LLAMA.CPP WITH FULL OPTIMIZATIONS"
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
fi

cd llama.cpp

# Clean build directory
rm -rf build
mkdir -p build
cd build

# Configure with maximum optimizations
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_NATIVE=ON \
    -DLLAMA_CURL=ON \
    -DLLAMA_OPENBLAS=ON \
    -DLLAMA_BLAS=ON \
    -DBLAS_LIBRARIES=/usr/lib/openblas-base/libopenblas.a \
    -DLLAMA_CUDA=OFF \
    -DLLAMA_METAL=OFF \
    -DLLAMA_MPI=OFF \
    -DLLAMA_CCACHE=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_EXAMPLES=ON

# Build with maximum parallelism
make -j$(nproc)

# Return to main directory
cd ../..

# Download the model
echo "📥 DOWNLOADING PHI-3.5-MINI-INSTRUCT MODEL"
if [ ! -d "Phi-3.5-mini-instruct" ]; then
    git clone https://huggingface.co/microsoft/Phi-3.5-mini-instruct
fi

# Verify model files
echo "🔍 VERIFYING MODEL FILES"
cd Phi-3.5-mini-instruct
ls -la
cd ..

# Create comprehensive conversion script
cat > complete_conversion.py << 'EOF'
import os
import sys
import subprocess
import json
import time
from datetime import datetime

class Phi3Converter:
    def __init__(self, model_path, output_dir):
        self.model_path = model_path
        self.output_dir = output_dir
        self.quant_types = ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", 
                           "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", 
                           "Q6_K", "Q8_0", "F16", "F32"]
        
    def run_command(self, cmd, description):
        print(f"🔄 {description}")
        print(f"   Command: {cmd}")
        
        start_time = time.time()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        end_time = time.time()
        
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr}")
            return False
        
        print(f"✅ Completed in {end_time - start_time:.2f}s")
        return True
    
    def convert_to_gguf(self, outtype="f16"):
        output_file = f"{self.output_dir}/phi3.5-mini.{outtype}.gguf"
        cmd = f"python3 llama.cpp/convert-hf-to-gguf.py {self.model_path} --outfile {output_file} --outtype {outtype}"
        
        return self.run_command(cmd, f"Converting to GGUF ({outtype})")
    
    def quantize_model(self, input_file, quant_type):
        output_file = f"{self.output_dir}/phi3.5-mini.{quant_type}.gguf"
        cmd = f"./llama.cpp/build/bin/quantize {input_file} {output_file} {quant_type}"
        
        return self.run_command(cmd, f"Quantizing to {quant_type}")
    
    def generate_model_card(self):
        model_card = {
            "model_name": "Phi-3.5-mini-instruct-GGUF-Comprehensive",
            "base_model": "microsoft/Phi-3.5-mini-instruct",
            "description": "Comprehensive GGUF quantization of Phi-3.5-mini-instruct for Saem's Tunes",
            "quantizations": self.quant_types,
            "conversion_date": datetime.now().isoformat(),
            "file_sizes": {},
            "recommended_use_cases": {
                "Q2_K": "Mobile devices, extremely low memory",
                "Q3_K_S": "Mobile devices, low memory",
                "Q4_0": "Good balance for most use cases",
                "Q4_K_M": "Recommended balance (quality/speed)",
                "Q5_K_M": "High quality with reasonable size",
                "Q8_0": "Maximum quality, larger size",
                "F16": "Original quality, large size"
            }
        }
        
        with open(f"{self.output_dir}/model_card.json", "w") as f:
            json.dump(model_card, f, indent=2)
    
    def run_comprehensive_conversion(self):
        print("🚀 STARTING COMPREHENSIVE CONVERSION PIPELINE")
        
        # Convert to F16 first
        if not self.convert_to_gguf("f16"):
            return False
        
        base_model = f"{self.output_dir}/phi3.5-mini.f16.gguf"
        
        # Create all quantization variants
        for quant_type in self.quant_types:
            if quant_type not in ["F16", "F32"]:
                self.quantize_model(base_model, quant_type)
        
        # Convert to F32 separately
        self.convert_to_gguf("f32")
        
        # Generate model card
        self.generate_model_card()
        
        print("🎉 COMPREHENSIVE CONVERSION COMPLETED!")
        return True

if __name__ == "__main__":
    converter = Phi3Converter("Phi-3.5-mini-instruct", "phi3.5-mini-gguf-complete")
    converter.run_comprehensive_conversion()
EOF

# Run the comprehensive conversion
echo "🏃 STARTING COMPREHENSIVE CONVERSION PROCESS"
python3 complete_conversion.py

# Generate file manifest
echo "📊 GENERATING FILE MANIFEST"
ls -la phi3.5-mini-gguf-complete/ > phi3.5-mini-gguf-complete/file_manifest.txt

# Create deployment package
echo "📦 CREATING DEPLOYMENT PACKAGE"
tar -czvf phi3.5-mini-complete-gguf-$(date +%Y%m%d).tar.gz phi3.5-mini-gguf-complete/

echo "🎉 COMPLETE CONVERSION SYSTEM FINISHED!"
echo "📍 Output directory: phi3.5-mini-gguf-complete/"
echo "📁 Total files generated: $(ls -1 phi3.5-mini-gguf-complete/*.gguf | wc -l)"