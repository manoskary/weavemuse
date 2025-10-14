import os
import time
import torch
import re
import difflib
from .utils import *
from .model_manager import get_optimal_notagen_model, model_manager
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


def get_system_memory_gb():
    """Get system RAM in GB"""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        return 8.0  # Default assumption

def get_gpu_memory_gb():
    """Get GPU memory in GB"""
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            return 0.0
    return 0.0

def should_use_quantization():
    """
    Automatically determine if quantization should be used based on system resources.
    
    Returns:
        bool: True if quantization should be used
    """
    # Check if user explicitly set quantization
    if 'NOTAGEN_USE_QUANTIZATION' in os.environ:
        return os.environ['NOTAGEN_USE_QUANTIZATION'].lower() in ['true', '1', 'yes']
    
    # Auto-detect based on system resources
    system_ram = get_system_memory_gb()
    gpu_ram = get_gpu_memory_gb()
    
    logger.info(f"System RAM: {system_ram:.1f}GB, GPU RAM: {gpu_ram:.1f}GB")
    
    # Always prefer quantized model for better performance and efficiency
    # Use quantization if:
    # 1. System has less than 16GB RAM, OR
    # 2. GPU has less than 8GB VRAM, OR
    # 3. No GPU available
    # 4. Or by default (quantized is faster and smaller)
    if system_ram < 16.0 or gpu_ram < 8.0 or not torch.cuda.is_available():
        logger.info("Using quantized model due to limited resources")
        return True
    else:
        # Default to quantized for better performance
        logger.info("Using quantized model (default for optimal performance)")
        return True

def load_model_weights(model_id=None):
    """Load model weights with intelligent quantization support."""
    global model
    
    # Determine quantization strategy
    use_quantization = should_use_quantization()
    
    # Try loading quantized model first (recommended)
    if use_quantization and (model_id is None or model_id=="manoskary/NotaGenX-Quantized"):
        # Download quantized weights if needed
        try:
            quantized_path = download_model_weights(repo_id="manoskary/NotaGenX-Quantized")
            
            logger.info(f"Loading quantized model from: {quantized_path}")
            
            # Load the quantized state dict
            state_dict = torch.load(quantized_path, map_location='cpu', weights_only=False)
            
            # The weights from HF are already quantized INT8, load them directly
            model.load_state_dict(state_dict, strict=False)
            
            # Move to appropriate device
            # Quantized model can work on both CPU and CUDA
            if torch.cuda.is_available():
                model = model.to(device)
                logger.info(f"✅ Quantized model loaded successfully on {device}!")
            else:
                model = model.to('cpu')
                logger.info("✅ Quantized model loaded successfully on CPU!")
            
            # Ensure model is in eval mode
            model.eval()
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load quantized model: {e}")
            logger.info("Falling back to original model...")
            import traceback
            logger.debug(traceback.format_exc())
    
    # Fall back to original weights
    try:
        logger.info("Loading original full-precision model...")
        original_path = download_model_weights(repo_id="ElectricAlexis/NotaGen")
        
        checkpoint = torch.load(original_path, map_location=device, weights_only=False)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        logger.info("✅ Original model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def get_model_device_and_dtype():
    """Get the appropriate device and dtype for the current model."""
    global model, device
    if model is not None:
        # Get device from the model's parameters
        try:
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            return model_device, model_dtype
        except (StopIteration, AttributeError):
            # Fallback if model has no parameters
            pass
    
    # Fallback based on quantization settings and global device
    if USE_QUANTIZATION:
        return device, torch.float32  # Use the global device variable
    else:
        return device, torch.float32


def create_tensor_for_model(data, **kwargs):
    """Create a tensor with appropriate device and dtype for the current model."""
    model_device, model_dtype = get_model_device_and_dtype()
    return torch.tensor(data, device=model_device, dtype=model_dtype, **kwargs)


def download_model_weights(repo_id="manoskary/NotaGenX-Quantized"):
    """
    Download model weights from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repo ID (default: quantized model)
        
    Returns:
        str: Path to downloaded weights file
    """
    # Determine filename based on repo
    if repo_id == "manoskary/NotaGenX-Quantized":
        weights_filename = "pytorch_model.bin"
        cache_subdir = "notagen-quantized"
    elif repo_id == "ElectricAlexis/NotaGen":
        weights_filename = "weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth"
        cache_subdir = "notagen-original"
    else:
        # Fallback for other repos
        weights_filename = "weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth"
        cache_subdir = "notagen-custom"
    
    # Create cache directory
    cache_dir = os.path.join(os.getcwd(), ".cache", cache_subdir)
    os.makedirs(cache_dir, exist_ok=True)
    
    local_weights_path = os.path.join(cache_dir, weights_filename)

    # Check if weights already exist locally
    if os.path.exists(local_weights_path):
        logger.info(f"Model weights already cached at {local_weights_path}")
        return local_weights_path

    logger.info(f"Downloading model weights from {repo_id}...")
    try:
        # Download from HuggingFace Hub
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=weights_filename,
            cache_dir=cache_dir,
            local_dir=cache_dir,
            local_dir_use_symlinks=False
        )
        logger.info(f"✅ Model weights downloaded to {downloaded_path}")
        return downloaded_path
    except Exception as e:
        logger.error(f"❌ Error downloading model weights: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to download model weights from {repo_id}: {str(e)}")


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    """
    Prepare model for k-bit training.
    Features include:
    1. Convert model to mixed precision (FP16).
    2. Disable unnecessary gradient computations.
    3. Enable gradient checkpointing (optional).
    """
    # Convert model to mixed precision
    model = model.to(dtype=torch.float16)

    # Disable gradients for embedding layers
    for param in model.parameters():
        if param.dtype == torch.float32:
            param.requires_grad = False

    # Enable gradient checkpointing
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model


model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=False
)

print("Parameter Number: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))


def postprocess_inst_names(abc_text):
    
    # try to load standard instrument names and mapping from local files otherwise directly from the raw GitHub
    try:
        with open(os.path.join(os.path.dirname(__file__), 'standard_inst_names.txt'), 'r', encoding='utf-8') as f:
            standard_instruments_list = [line.strip() for line in f if line.strip()]

        with open(os.path.join(os.path.dirname(__file__), 'instrument_mapping.json'), 'r', encoding='utf-8') as f:
            instrument_mapping = json.load(f)
    except:
        logger.warning("Failed to load local instrument files, fetching from GitHub...")
        raw_gh_path = "https://raw.githubusercontent.com/manoskary/weavemuse/refs/heads/main/weavemuse/models/notagen/"        
        url_txt = raw_gh_path + "standard_inst_names.txt"
        response_txt = requests.get(url_txt)
        response_txt.raise_for_status()
        standard_instruments_list = [line.strip() for line in response_txt.text.splitlines() if line.strip()]

        url_json = raw_gh_path + "instrument_mapping.json"
        response_json = requests.get(url_json)
        response_json.raise_for_status()
        instrument_mapping = response_json.json()
        

    abc_lines = abc_text.split('\n')
    abc_lines = list(filter(None, abc_lines))
    abc_lines = [line + '\n' for line in abc_lines]

    for i, line in enumerate(abc_lines):
        if line.startswith('V:') and 'nm=' in line:
            match = re.search(r'nm="([^"]*)"', line)
            if match:
                inst_name = match.group(1)

                # Check if the instrument name is already standard
                if inst_name in standard_instruments_list:
                    continue
                else:
                    inst_name = "Piano"

                # Find the most similar key in instrument_mapping
                matching_key = difflib.get_close_matches(inst_name, list(instrument_mapping.keys()), n=1, cutoff=0.6)

                if matching_key:
                    # Replace the instrument name with the standardized version
                    replacement = instrument_mapping[matching_key[0]]
                    new_line = line.replace(f'nm="{inst_name}"', f'nm="{replacement}"')
                    abc_lines[i] = new_line

    # Combine the lines back into a single string
    processed_abc_text = ''.join(abc_lines)
    return processed_abc_text


def complete_brackets(s):
    stack = []
    bracket_map = {'{': '}', '[': ']', '(': ')'}

    # Iterate through each character, handle bracket matching
    for char in s:
        if char in bracket_map:
            stack.append(char)
        elif char in bracket_map.values():
            # Find the corresponding left bracket
            for key, value in bracket_map.items():
                if value == char:
                    if stack and stack[-1] == key:
                        stack.pop()
                    break  # Found matching right bracket, process next character

    # Complete missing right brackets (in reverse order of remaining left brackets in stack)
    completion = ''.join(bracket_map[c] for c in reversed(stack))
    return s + completion


def rest_unreduce(abc_lines):
    tunebody_index = None
    for i in range(len(abc_lines)):
        if abc_lines[i].startswith('%%score'):
            abc_lines[i] = complete_brackets(abc_lines[i])
        if '[V:' in abc_lines[i]:
            tunebody_index = i
            break

    metadata_lines = abc_lines[: tunebody_index]
    tunebody_lines = abc_lines[tunebody_index:]

    part_symbol_list = []
    voice_group_list = []
    for line in metadata_lines:
        if line.startswith('%%score'):
            for round_bracket_match in re.findall(r'\((.*?)\)', line):
                voice_group_list.append(round_bracket_match.split())
            existed_voices = [item for sublist in voice_group_list for item in sublist]
        if line.startswith('V:'):
            symbol = line.split()[0]
            part_symbol_list.append(symbol)
            if symbol[2:] not in existed_voices:
                voice_group_list.append([symbol[2:]])
    z_symbol_list = []  # voices that use z as rest
    x_symbol_list = []  # voices that use x as rest
    for voice_group in voice_group_list:
        z_symbol_list.append('V:' + voice_group[0])
        for j in range(1, len(voice_group)):
            x_symbol_list.append('V:' + voice_group[j])

    part_symbol_list.sort(key=lambda x: int(x[2:]))

    unreduced_tunebody_lines = []

    for i, line in enumerate(tunebody_lines):
        unreduced_line = ''

        line = re.sub(r'^\[r:[^\]]*\]', '', line)

        pattern = r'\[V:(\d+)\](.*?)(?=\[V:|$)'
        matches = re.findall(pattern, line)

        line_bar_dict = {}
        for match in matches:
            key = f'V:{match[0]}'
            value = match[1]
            line_bar_dict[key] = value

        # calculate duration and collect barline
        dur_dict = {}
        for symbol, bartext in line_bar_dict.items():
            right_barline = ''.join(re.split(Barline_regexPattern, bartext)[-2:])
            bartext = bartext[:-len(right_barline)]
            try:
                bar_dur = calculate_bartext_duration(bartext)
            except:
                bar_dur = None
            if bar_dur is not None:
                if bar_dur not in dur_dict.keys():
                    dur_dict[bar_dur] = 1
                else:
                    dur_dict[bar_dur] += 1

        try:
            ref_dur = max(dur_dict, key=dur_dict.get)
        except:
            pass  # use last ref_dur

        if i == 0:
            prefix_left_barline = line.split('[V:')[0]
        else:
            prefix_left_barline = ''

        for symbol in part_symbol_list:
            if symbol in line_bar_dict.keys():
                symbol_bartext = line_bar_dict[symbol]
            else:
                if symbol in z_symbol_list:
                    symbol_bartext = prefix_left_barline + 'z' + str(ref_dur) + right_barline
                elif symbol in x_symbol_list:
                    symbol_bartext = prefix_left_barline + 'x' + str(ref_dur) + right_barline
            unreduced_line += '[' + symbol + ']' + symbol_bartext

        unreduced_tunebody_lines.append(unreduced_line + '\n')

    unreduced_lines = metadata_lines + unreduced_tunebody_lines

    return unreduced_lines


def inference_patch(period, composer, instrumentation):
    prompt_lines = [
        '%' + period + '\n',
        '%' + composer + '\n',
        '%' + instrumentation + '\n']

    while True:

        failure_flag = False

        bos_patch = [patchilizer.bos_token_id] * (PATCH_SIZE - 1) + [patchilizer.eos_token_id]

        start_time = time.time()

        prompt_patches = patchilizer.patchilize_metadata(prompt_lines)
        byte_list = list(''.join(prompt_lines))
        context_tunebody_byte_list = []
        metadata_byte_list = []

        print(''.join(byte_list), end='')

        prompt_patches = [[ord(c) for c in patch] + [patchilizer.special_token_id] * (PATCH_SIZE - len(patch)) for patch
                          in prompt_patches]
        prompt_patches.insert(0, bos_patch)

        input_patches = create_tensor_for_model(prompt_patches).reshape(1, -1)

        end_flag = False
        cut_index = None

        tunebody_flag = False
        
        # Get device and dtype for current model
        model_device, model_dtype = get_model_device_and_dtype()

        with torch.inference_mode():

            while True:
                # Only use autocast for CUDA, not for CPU quantized model
                if model_device == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        predicted_patch = model.generate(input_patches.unsqueeze(0),
                                                         top_k=TOP_K,
                                                         top_p=TOP_P,
                                                         temperature=TEMPERATURE)
                else:
                    predicted_patch = model.generate(input_patches.unsqueeze(0),
                                                     top_k=TOP_K,
                                                     top_p=TOP_P,
                                                     temperature=TEMPERATURE)
                    
                if not tunebody_flag and patchilizer.decode([predicted_patch]).startswith(
                        '[r:'):  # 初次进入tunebody，必须以[r:0/开头
                    tunebody_flag = True
                    r0_patch = create_tensor_for_model([ord(c) for c in '[r:0/']).unsqueeze(0)
                    temp_input_patches = torch.concat([input_patches, r0_patch], axis=-1)
                    
                    if model_device == 'cuda':
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            predicted_patch = model.generate(temp_input_patches.unsqueeze(0),
                                                             top_k=TOP_K,
                                                             top_p=TOP_P,
                                                             temperature=TEMPERATURE)
                    else:
                        predicted_patch = model.generate(temp_input_patches.unsqueeze(0),
                                                         top_k=TOP_K,
                                                         top_p=TOP_P,
                                                         temperature=TEMPERATURE)
                    predicted_patch = [ord(c) for c in '[r:0/'] + predicted_patch
                if predicted_patch[0] == patchilizer.bos_token_id and predicted_patch[1] == patchilizer.eos_token_id:
                    end_flag = True
                    break
                next_patch = patchilizer.decode([predicted_patch])

                for char in next_patch:
                    byte_list.append(char)
                    if tunebody_flag:
                        context_tunebody_byte_list.append(char)
                    else:
                        metadata_byte_list.append(char)
                    print(char, end='')

                patch_end_flag = False
                for j in range(len(predicted_patch)):
                    if patch_end_flag:
                        predicted_patch[j] = patchilizer.special_token_id
                    if predicted_patch[j] == patchilizer.eos_token_id:
                        patch_end_flag = True

                predicted_patch = create_tensor_for_model([predicted_patch])  # (1, 16)
                input_patches = torch.cat([input_patches, predicted_patch], dim=1)  # (1, 16 * patch_len)

                if len(byte_list) > 102400:
                    failure_flag = True
                    break
                if time.time() - start_time > 10 * 60:
                    failure_flag = True
                    break

                if input_patches.shape[1] >= PATCH_LENGTH * PATCH_SIZE and not end_flag:
                    print('Stream generating...')

                    metadata = ''.join(metadata_byte_list)
                    context_tunebody = ''.join(context_tunebody_byte_list)

                    if '\n' not in context_tunebody:
                        break  # Generated content is all metadata, abandon

                    context_tunebody_lines = context_tunebody.strip().split('\n')

                    if not context_tunebody.endswith('\n'):
                        context_tunebody_lines = [context_tunebody_lines[i] + '\n' for i in
                                                  range(len(context_tunebody_lines) - 1)] + [context_tunebody_lines[-1]]
                    else:
                        context_tunebody_lines = [context_tunebody_lines[i] + '\n' for i in
                                                  range(len(context_tunebody_lines))]

                    cut_index = len(context_tunebody_lines) // 2
                    abc_code_slice = metadata + ''.join(context_tunebody_lines[-cut_index:])

                    input_patches = patchilizer.encode_generate(abc_code_slice)

                    input_patches = [item for sublist in input_patches for item in sublist]
                    input_patches = create_tensor_for_model([input_patches])
                    input_patches = input_patches.reshape(1, -1)

                    context_tunebody_byte_list = list(''.join(context_tunebody_lines[-cut_index:]))

            if not failure_flag:
                abc_text = ''.join(byte_list)

                # unreduce
                abc_lines = abc_text.split('\n')
                abc_lines = list(filter(None, abc_lines))
                abc_lines = [line + '\n' for line in abc_lines]
                try:
                    unreduced_abc_lines = rest_unreduce(abc_lines)
                except:
                    failure_flag = True
                    pass
                else:
                    unreduced_abc_lines = [line for line in unreduced_abc_lines if
                                           not (line.startswith('%') and not line.startswith('%%'))]
                    unreduced_abc_lines = ['X:1\n'] + unreduced_abc_lines
                    unreduced_abc_text = ''.join(unreduced_abc_lines)
                    return unreduced_abc_text


if __name__ == '__main__':
    inference_patch('Classical', 'Beethoven, Ludwig van', 'Orchestral')
# else:
#     # Load the model weights when imported (but not when run as main)
#     try:
#         load_model_weights()
#     except Exception as e:
#         logger.warning(f"Failed to load model weights on import: {e}")
#         logger.info("Model will be loaded on first use.")
