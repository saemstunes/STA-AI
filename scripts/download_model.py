#!/usr/bin/env python3

"""
Script to download the Phi-3.5-mini-instruct model for local development
"""

import os
import sys
from huggingface_hub import hf_hub_download, snapshot_download
import argparse
import requests
import hashlib
import time

def calculate_file_hash(filepath):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_with_progress(repo_id, filename, local_dir):
    """Download file with progress tracking"""
    print(f"📥 Downloading {filename} from {repo_id}")
    
    try:
        # Download with progress
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        # Verify download
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024 * 1024)  # GB
            file_hash = calculate_file_hash(model_path)
            
            print(f"✅ Download completed successfully!")
            print(f"📁 File: {model_path}")
            print(f"📊 Size: {file_size:.2f} GB")
            print(f"🔒 Hash: {file_hash}")
            
            return model_path
        else:
            print(f"❌ Downloaded file not found: {model_path}")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return None

def check_disk_space(required_gb=3):
    """Check if there's enough disk space"""
    try:
        stat = os.statvfs('/')
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        
        print(f"💾 Available disk space: {free_gb:.1f} GB")
        print(f"💾 Required disk space: {required_gb} GB")
        
        if free_gb < required_gb:
            print(f"❌ Not enough disk space. Need {required_gb} GB, have {free_gb:.1f} GB")
            return False
        return True
    except:
        print("⚠️  Could not check disk space")
        return True

def main():
    parser = argparse.ArgumentParser(description="Download TinyLlama-1.1B-Chat model")
    parser.add_argument(
        "--repo",
        default="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        help="Hugging Face repository ID"
    )
    parser.add_argument(
        "--file",
        default="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        help="Model filename"
    )
    parser.add_argument(
        "--dir",
        default="./models",
        help="Local directory to save the model"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if file exists"
    )
    
    args = parser.parse_args()
    
    print("🚀 Saem's Tunes AI Model Downloader")
    print("=" * 50)
    
    # Check disk space
    if not check_disk_space(3):
        sys.exit(1)
    
    # Create local directory if it doesn't exist
    os.makedirs(args.dir, exist_ok=True)
    
    # Check if file already exists
    local_path = os.path.join(args.dir, args.file)
    if os.path.exists(local_path) and not args.force:
        print(f"✅ Model already exists: {local_path}")
        file_size = os.path.getsize(local_path) / (1024 * 1024 * 1024)
        print(f"📊 Size: {file_size:.2f} GB")
        return local_path
    
    # Download the model
    start_time = time.time()
    model_path = download_with_progress(args.repo, args.file, args.dir)
    download_time = time.time() - start_time
    
    if model_path:
        print(f"⏱️  Download time: {download_time:.1f} seconds")
        print(f"🎉 Model ready for use!")
        
        # Create model info file
        info_file = os.path.join(args.dir, "model_info.txt")
        with open(info_file, 'w') as f:
            f.write(f"Model: {args.repo}\n")
            f.write(f"File: {args.file}\n")
            f.write(f"Downloaded: {time.ctime()}\n")
            f.write(f"Path: {model_path}\n")
            f.write(f"Size: {os.path.getsize(model_path) / (1024**3):.2f} GB\n")
        
        print(f"📄 Model info saved to: {info_file}")
    else:
        print("❌ Model download failed")
        sys.exit(1)
    
    return model_path

if __name__ == "__main__":
    main()
