"""
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
