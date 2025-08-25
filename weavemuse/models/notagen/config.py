import os

# Configurations for inference
INFERENCE_WEIGHTS_PATH = 'weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth'               # Path to weights for inference# Folder to save output files
TOP_K = 9                                                       # Top k for sampling
TOP_P = 0.9                                                      # Top p for sampling
TEMPERATURE = 1.2                                                 # Temperature for sampling

# Configurations for model
PATCH_STREAM = True                                             # Stream training / inference
PATCH_SIZE = 16                                                # Patch Size
PATCH_LENGTH = 1024                                             # Patch Length
CHAR_NUM_LAYERS = 6                                             # Number of layers in the decoder
PATCH_NUM_LAYERS = 20                                           # Number of layers in the encoder
HIDDEN_SIZE = 1280                                               # Hidden Size