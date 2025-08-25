import os
import time
import torch
import re
import difflib
from .utils import *
from .config import (
    TEMPERATURE,
    TOP_P,
    TOP_K
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


def download_model_weights():
    weights_path = "weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth"
    local_weights_path = os.path.join(os.getcwd(), weights_path)

    # Check if weights already exist locally
    if os.path.exists(local_weights_path):
        logger.info(f"Model weights already exist at {local_weights_path}")
        return local_weights_path

    logger.info("Downloading model weights from HuggingFace Hub...")
    try:
        # Download from HuggingFace
        downloaded_path = hf_hub_download(
            repo_id="ElectricAlexis/NotaGen",
            filename=weights_path,
            local_dir=os.getcwd(),
            local_dir_use_symlinks=False
        )
        logger.info(f"Model weights downloaded successfully to {downloaded_path}")
        return downloaded_path
    except Exception as e:
        logger.error(f"Error downloading model weights: {str(e)}")
        raise RuntimeError(f"Failed to download model weights: {str(e)}")


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

# Download weights at startup
model_weights_path = download_model_weights()
checkpoint = torch.load(model_weights_path, weights_only=True, map_location=torch.device(device))
model.load_state_dict(checkpoint['model'], strict=False)

model = model.to(device)
model.eval()


def postprocess_inst_names(abc_text):
    
    with open(os.path.join(os.path.dirname(__file__), 'standard_inst_names.txt'), 'r', encoding='utf-8') as f:
        standard_instruments_list = [line.strip() for line in f if line.strip()]

    with open(os.path.join(os.path.dirname(__file__), 'instrument_mapping.json'), 'r', encoding='utf-8') as f:
        instrument_mapping = json.load(f)

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

        input_patches = torch.tensor(prompt_patches, device=device).reshape(1, -1)

        end_flag = False
        cut_index = None

        tunebody_flag = False

        with torch.inference_mode():

            while True:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    predicted_patch = model.generate(input_patches.unsqueeze(0),
                                                     top_k=TOP_K,
                                                     top_p=TOP_P,
                                                     temperature=TEMPERATURE)
                if not tunebody_flag and patchilizer.decode([predicted_patch]).startswith(
                        '[r:'):  # 初次进入tunebody，必须以[r:0/开头
                    tunebody_flag = True
                    r0_patch = torch.tensor([ord(c) for c in '[r:0/']).unsqueeze(0).to(device)
                    temp_input_patches = torch.concat([input_patches, r0_patch], axis=-1)
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

                predicted_patch = torch.tensor([predicted_patch], device=device)  # (1, 16)
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
                    input_patches = torch.tensor([input_patches], device=device)
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
