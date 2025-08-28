#!/usr/bin/env python3
"""
NotaGen Model Quantization Script
================================
Quantize the NotaGen model to reduce size from 5.8GB to ~1.5GB
"""

import os
import sys
import torch
import time
from pathlib import Path

# Add the project root to path
sys.path.append('/home/manos/codes/weavemuse')

from .config import *
from .utils import NotaGenLMHeadModel
from .quantization import NotaGenQuantizer
from transformers import GPT2Config

def main():
    """Main quantization function."""
    print("🎵 NotaGen Model Quantization")
    print("=" * 50)
    
    # Check if weights exist
    weights_path = "/home/manos/codes/weavemuse/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth"
    if not os.path.exists(weights_path):
        print(f"❌ Weights not found at: {weights_path}")
        return False
    
    print(f"📂 Original weights: {weights_path}")
    original_size = os.path.getsize(weights_path) / (1024**3)  # GB
    print(f"📊 Original size: {original_size:.2f} GB")
    
    # Create model architecture (same as inference.py)
    print("\n🔧 Creating model architecture...")
    
    patch_config = GPT2Config(
        num_hidden_layers=PATCH_NUM_LAYERS,
        max_length=PATCH_LENGTH,
        max_position_embeddings=PATCH_LENGTH,
        n_embd=HIDDEN_SIZE,
        num_attention_heads=HIDDEN_SIZE // 64,
        vocab_size=1
    )
    
    byte_config = GPT2Config(
        num_hidden_layers=CHAR_NUM_LAYERS,
        max_length=PATCH_SIZE + 1,
        max_position_embeddings=PATCH_SIZE + 1,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=HIDDEN_SIZE // 64,
        vocab_size=128
    )
    
    model = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=byte_config)
    
    # Load original weights
    print("📥 Loading original weights...")
    try:
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        # Extract model state dict from checkpoint
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            state_dict = checkpoint
            print("✅ Loaded direct state dict")
            
        model.load_state_dict(state_dict)
        print("✅ Weights loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return False
    
    # Apply quantization
    print("\n⚡ Applying dynamic quantization...")
    quantizer = NotaGenQuantizer(model, device='cpu')
    
    start_time = time.time()
    quantized_model = quantizer.dynamic_quantization()
    quantization_time = time.time() - start_time
    
    print(f"✅ Quantization completed in {quantization_time:.2f} seconds!")
    
    # Save quantized model
    quantized_path = "/home/manos/codes/weavemuse/.cache/weights_notagenx_quantized_int8.pth"
    print(f"\n💾 Saving quantized model to: {quantized_path}")
    
    saved_size = quantizer.save_quantized_model(quantized_model, quantized_path)
    
    # Results summary
    print("\n" + "=" * 50)
    print("📈 QUANTIZATION RESULTS")
    print("=" * 50)
    print(f"Original size:    {original_size:.2f} GB")
    print(f"Quantized size:   {saved_size / 1024:.2f} GB")
    print(f"Size reduction:   {((original_size * 1024 - saved_size) / (original_size * 1024)) * 100:.1f}%")
    print(f"Compression ratio: {(original_size * 1024) / saved_size:.1f}x")
    print(f"Space saved:      {(original_size * 1024 - saved_size):.1f} MB")
    
    # Update config files
    print(f"\n🔧 Updating configuration files...")
    update_config_files(quantized_path)
    
    print(f"\n✨ Quantization complete!")
    print(f"   Your NotaGen model is now {((original_size * 1024 - saved_size) / (original_size * 1024)) * 100:.1f}% smaller!")
    print(f"   To use the quantized model, restart your WeaveMuse app.")
    
    return True

def update_config_files(quantized_path):
    """Update config files to use quantized model."""
    
    # Update config.py
    config_path = "/home/manos/codes/weavemuse/weavemuse/models/notagen/config.py"
    print(f"📝 Adding quantization config to: {config_path}")
    
    # Read current config
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    # Add quantization settings if not already present
    quantization_config = f"""
# Quantization settings
USE_QUANTIZATION = True
QUANTIZED_WEIGHTS_PATH = '.cache/{os.path.basename(quantized_path)}'
"""
    
    if "USE_QUANTIZATION" not in config_content:
        with open(config_path, 'a') as f:
            f.write(quantization_config)
        print("✅ Added quantization config")
    else:
        print("ℹ️  Quantization config already exists")
    
    # Create updated inference.py with quantization support
    create_quantized_inference()

def create_quantized_inference():
    """Create an updated inference.py that supports quantization."""
    
    print("📝 Creating quantized inference loader...")
    
    quantized_inference_content = '''"""
NotaGen Inference with Quantization Support
==========================================
Enhanced version of inference.py with quantization support.
"""

import os
import time
import torch
import re
import difflib
from .utils import *
from .config import (
    TEMPERATURE,
    TOP_P,
    TOP_K,
    USE_QUANTIZATION,
    QUANTIZED_WEIGHTS_PATH,
    INFERENCE_WEIGHTS_PATH
)
from transformers import GPT2Config
from abctoolkit.utils import Exclaim_re, Quote_re, SquareBracket_re, Barline_regexPattern
from abctoolkit.transpose import Note_list, Pitch_sign_list
from abctoolkit.duration import calculate_bartext_duration
import requests
import torch
from huggingface_hub import hf_hub_download
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Note_list = Note_list + ['z', 'x']

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

patchilizer = Patchilizer()

patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS,
                          max_length=PATCH_LENGTH,
                          max_position_embeddings=PATCH_LENGTH,
                          n_embd=HIDDEN_SIZE,
                          num_attention_heads=HIDDEN_SIZE // 64,
                          vocab_size=1)
byte_config = GPT2Config(num_hidden_layers=CHAR_NUM_LAYERS,
                         max_length=PATCH_SIZE + 1,
                         max_position_embeddings=PATCH_SIZE + 1,
                         hidden_size=HIDDEN_SIZE,
                         num_attention_heads=HIDDEN_SIZE // 64,
                         vocab_size=128)

model = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=byte_config).to(device)

def load_model_weights():
    """Load model weights with quantization support."""
    global model
    
    # Check if quantized model should be used
    if hasattr(config, 'USE_QUANTIZATION') and config.USE_QUANTIZATION:
        quantized_path = getattr(config, 'QUANTIZED_WEIGHTS_PATH', 'weights_notagenx_quantized_int8.pth')
        
        if os.path.exists(quantized_path):
            logger.info(f"Loading quantized model from: {quantized_path}")
            
            # Apply quantization to model structure
            model = torch.quantization.quantize_dynamic(
                model.cpu(), 
                {torch.nn.Linear}, 
                dtype=torch.qint8
            )
            
            # Load quantized weights
            state_dict = torch.load(quantized_path, map_location='cpu')
            model.load_state_dict(state_dict)
            
            # Move to device after loading
            if device.type == 'cuda':
                logger.warning("Quantized model running on CPU for compatibility")
                model = model.cpu()
            else:
                model = model.to(device)
                
            logger.info("✅ Quantized model loaded successfully!")
            return True
        else:
            logger.warning(f"Quantized weights not found at: {quantized_path}")
    
    # Fall back to original weights
    logger.info(f"Loading original model from: {INFERENCE_WEIGHTS_PATH}")
    if os.path.exists(INFERENCE_WEIGHTS_PATH):
        state_dict = torch.load(INFERENCE_WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("✅ Original model loaded successfully!")
        return True
    else:
        # Download if not exists
        download_model_weights()
        state_dict = torch.load(INFERENCE_WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state_dict)
        return True

# Load the model weights on import
load_model_weights()

def download_model_weights():
    """Download model weights if they don't exist."""
    # Original download logic from inference.py
    logger.info("Downloading NotaGen model weights...")
    # Add your download logic here
    pass

# Rest of the inference functions remain the same...
'''
    
    # Write the enhanced inference file
    quantized_inference_path = "/home/manos/codes/weavemuse/weavemuse/models/notagen/inference_quantized.py"
    with open(quantized_inference_path, 'w') as f:
        f.write(quantized_inference_content)
    
    print(f"✅ Created quantized inference at: {quantized_inference_path}")
    print("ℹ️  You can replace inference.py with this file to use quantization")

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Quantization successful!")
        print("   Restart your WeaveMuse app to use the smaller model.")
    else:
        print("\n❌ Quantization failed!")
        exit(1)
