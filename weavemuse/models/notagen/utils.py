import torch
import random
import bisect
import json
import re
from .config import *
from transformers import GPT2Model, GPT2LMHeadModel, PreTrainedModel, BitsAndBytesConfig
from samplings import top_p_sampling, top_k_sampling, temperature_sampling
from tokenizers import Tokenizer


quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_modules=["patch_embedding"]  # 跳过可能不兼容的模块
)

class Patchilizer:
    def __init__(self, stream=PATCH_STREAM):
        self.stream = stream
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = '(' + '|'.join(map(re.escape, self.delimiters)) + ')'
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.special_token_id = 0

    def split_bars(self, body_lines):
        """
        Split a body of music into individual bars.
        """
        new_bars = []
        try:
            for line in body_lines:
                line_bars = re.split(self.regexPattern, line)
                line_bars = list(filter(None, line_bars))
                new_line_bars = []

                if len(line_bars) == 1:
                    new_line_bars = line_bars
                else:
                    if line_bars[0] in self.delimiters:
                        new_line_bars = [line_bars[i] + line_bars[i + 1] for i in range(0, len(line_bars), 2)]
                    else:
                        new_line_bars = [line_bars[0]] + [line_bars[i] + line_bars[i + 1] for i in range(1, len(line_bars), 2)]
                    if 'V' not in new_line_bars[-1]:
                        new_line_bars[-2] += new_line_bars[-1]  # 吸收最后一个 小节线+\n 的组合
                        new_line_bars = new_line_bars[:-1]
                new_bars += new_line_bars
        except:
            pass

        return new_bars

    def split_patches(self, abc_text, patch_size=PATCH_SIZE, generate_last=False):
        if not generate_last and len(abc_text) % patch_size != 0:
            abc_text += chr(self.eos_token_id)
        patches = [abc_text[i : i + patch_size] for i in range(0, len(abc_text), patch_size)]
        return patches

    def patch2chars(self, patch):
        """
        Convert a patch into a bar.
        """
        bytes = ''
        for idx in patch:
            if idx == self.eos_token_id:
                break
            if idx < self.eos_token_id:
                pass
            bytes += chr(idx)
        return bytes
        

    def patchilize_metadata(self, metadata_lines):

        metadata_patches = []
        for line in metadata_lines:
            metadata_patches += self.split_patches(line)

        return metadata_patches
    
    def patchilize_tunebody(self, tunebody_lines, encode_mode='train'):

        tunebody_patches = []
        bars = self.split_bars(tunebody_lines)
        if encode_mode == 'train':
            for bar in bars:
                tunebody_patches += self.split_patches(bar)
        elif encode_mode == 'generate':
            for bar in bars[:-1]:
                tunebody_patches += self.split_patches(bar)
            tunebody_patches += self.split_patches(bars[-1], generate_last=True)
       
        return tunebody_patches

    def encode(self, abc_text, patch_length=PATCH_LENGTH, patch_size=PATCH_SIZE, add_special_patches=True, cut=True):

        lines = abc_text.split('\n')
        lines = list(filter(None, lines))
        lines = [line + '\n' for line in lines]

        tunebody_index = -1
        for i, line in enumerate(lines):
            if line.startswith('[r:'):
                tunebody_index = i
                break

        metadata_lines = lines[: tunebody_index]
        tunebody_lines = lines[tunebody_index:]

        metadata_patches = self.patchilize_metadata(metadata_lines)
        tunebody_patches = self.patchilize_tunebody(tunebody_lines, encode_mode='train')

        if add_special_patches:
            bos_patch = chr(self.bos_token_id) * (patch_size - 1) + chr(self.eos_token_id)
            eos_patch = chr(self.bos_token_id) + chr(self.eos_token_id) * (patch_size - 1)

            metadata_patches = [bos_patch] + metadata_patches
            tunebody_patches = tunebody_patches + [eos_patch]

        if self.stream:
            if len(metadata_patches) + len(tunebody_patches) > patch_length:
                available_cut_indexes = [0] + [index + 1 for index, patch in enumerate(tunebody_patches) if
                                               '\n' in patch]
                line_index_for_cut_index = list(range(len(available_cut_indexes)))  # 每个cut_index对应tunebody的哪一行
                end_index = len(metadata_patches) + len(tunebody_patches) - patch_length
                biggest_index = bisect.bisect_left(available_cut_indexes, end_index)  # biggest index 在 end_index 右面一位
                available_cut_indexes = available_cut_indexes[:biggest_index + 1]

                if len(available_cut_indexes) == 1:
                    choices = ['head']
                elif len(available_cut_indexes) == 2:
                    choices = ['head', 'tail']
                else:
                    choices = ['head', 'tail', 'middle']
                choice = random.choice(choices)
                if choice == 'head':
                    patches = metadata_patches + tunebody_patches[0:]
                else:
                    if choice == 'tail':
                        cut_index = len(available_cut_indexes) - 1
                    else:
                        cut_index = random.choice(range(1, len(available_cut_indexes) - 1))

                    line_index = line_index_for_cut_index[cut_index]
                    stream_tunebody_lines = tunebody_lines[line_index:]

                    stream_tunebody_patches = self.patchilize_tunebody(stream_tunebody_lines, encode_mode='train')
                    if add_special_patches:
                        stream_tunebody_patches = stream_tunebody_patches + [eos_patch]
                    patches = metadata_patches + stream_tunebody_patches
            else:
                patches = metadata_patches + tunebody_patches
        else:
            patches = metadata_patches + tunebody_patches

        patches = patches[: patch_length]

        # encode to ids
        id_patches = []
        for patch in patches:
            id_patch = [ord(c) for c in patch] + [self.special_token_id] * (patch_size - len(patch))
            id_patches.append(id_patch)

        return id_patches

    def encode_generate(self, abc_code, patch_length=PATCH_LENGTH, patch_size=PATCH_SIZE, add_special_patches=True):

        lines = abc_code.split('\n')
        lines = list(filter(None, lines))
    
        tunebody_index = None
        for i, line in enumerate(lines):
            if line.startswith('[V:') or line.startswith('[r:'):
                tunebody_index = i
                break
    
        metadata_lines = lines[ : tunebody_index]
        tunebody_lines = lines[tunebody_index : ]   # 备份未省略前的tunebody_lines
    
        metadata_lines = [line + '\n' for line in metadata_lines]
        if self.stream:
            if not abc_code.endswith('\n'): # 如果生成结果最后一行未完结
                tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines) - 1)] + [tunebody_lines[-1]]
            else:
                tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines))]
        else:
            tunebody_lines = [line + '\n' for line in tunebody_lines]
    
        metadata_patches = self.patchilize_metadata(metadata_lines)
        tunebody_patches = self.patchilize_tunebody(tunebody_lines, encode_mode='generate')
    
        if add_special_patches:
            bos_patch = chr(self.bos_token_id) * (patch_size - 1) + chr(self.eos_token_id)

            metadata_patches = [bos_patch] + metadata_patches
    
        patches = metadata_patches + tunebody_patches
        patches = patches[ : patch_length]

        # encode to ids
        id_patches = []
        for patch in patches:
            if len(patch) < PATCH_SIZE and patch[-1] != chr(self.eos_token_id):
                id_patch = [ord(c) for c in patch]
            else:
                id_patch = [ord(c) for c in patch] + [self.special_token_id] * (patch_size - len(patch))
            id_patches.append(id_patch)
        
        return id_patches

    def decode(self, patches):
        """
        Decode patches into music.
        """
        return ''.join(self.patch2chars(patch) for patch in patches)

        


class PatchLevelDecoder(PreTrainedModel):
    """
    A Patch-level Decoder model for generating patch features in an auto-regressive manner. 
    It inherits PreTrainedModel from transformers.
    """
    def __init__(self, config):
        super().__init__(config)
        self.patch_embedding = torch.nn.Linear(PATCH_SIZE * 128, config.n_embd).to(torch.float16)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.base = GPT2Model(config)

    def forward(self,
                patches: torch.Tensor,
                masks=None) -> torch.Tensor:
        """
        The forward pass of the patch-level decoder model.
        :param patches: the patches to be encoded
        :param masks: the masks for the patches
        :return: the encoded patches
        """
        # Get the device of the model's parameters
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        # Ensure patches are on the correct device first
        patches = patches.to(device)
        
        # Ensure patches tensor is LongTensor for one_hot compatibility
        # This is crucial for quantized models where tensor types can change
        if patches.dtype not in [torch.long, torch.int64]:
            patches = patches.long()
        
        # Ensure patches are in valid range for one_hot encoding
        patches = torch.clamp(patches, 0, 127)
        
        # Perform one_hot encoding and ensure result is on correct device
        patches_one_hot = torch.nn.functional.one_hot(patches, num_classes=128)
        patches_one_hot = patches_one_hot.to(device=device, dtype=dtype)
        
        # Reshape and pass through embedding layer
        patches = patches_one_hot.reshape(len(patches), -1, PATCH_SIZE * (128))
        patches = self.patch_embedding(patches)

        if masks==None:
            return self.base(inputs_embeds=patches)
        else:
            return self.base(inputs_embeds=patches,
                             attention_mask=masks)


class CharLevelDecoder(PreTrainedModel):
    """
    A Char-level Decoder model for generating the chars within each patch in an auto-regressive manner
    based on the encoded patch features. It inherits PreTrainedModel from transformers.
    """
    def __init__(self, config):
        super().__init__(config)
        self.special_token_id = 0
        self.bos_token_id = 1

        self.base = GPT2LMHeadModel(config)

    def forward(self,
                encoded_patches: torch.Tensor,
                target_patches: torch.Tensor):
        """
        The forward pass of the char-level decoder model.
        :param encoded_patches: the encoded patches
        :param target_patches: the target patches
        :return: the output of the model
        """
        # Ensure target_patches is LongTensor for embedding layers
        target_patches = target_patches.long()
        
        target_patches = torch.cat((torch.ones_like(target_patches[:, 0:1]) * self.bos_token_id,
                                        target_patches), dim=1)  # [patch_len, patch_size + 1]

        target_masks = target_patches == self.special_token_id  # [patch_len, patch_size + 1]
        labels = target_patches.clone().masked_fill_(target_masks, -100)

        target_masks = torch.ones_like(labels)
        target_masks = target_masks.masked_fill_(labels == -100, 0)

        input_embeds = torch.nn.functional.embedding(target_patches, self.base.transformer.wte.weight)
        input_embeds = torch.cat((encoded_patches.unsqueeze(1), input_embeds[:, 1:, :]), dim=1)
        logits = self.base(inputs_embeds=input_embeds,
                           attention_mask=target_masks).logits  # [patch_len, patch_size + 1, vocab_size]
        logits = logits[:, :-1, :]
        token_logps = torch.gather(logits.log_softmax(-1), dim=-1, index=target_patches[:, 1:].unsqueeze(-1)).squeeze(-1)   # [patch_len, patch_size]
        token_logps = token_logps[target_masks[:, 1:] == 1]
        all_logps = token_logps.sum()

        return all_logps

    def generate(self,
                 encoded_patch: torch.Tensor,   # [hidden_size]
                 tokens: torch.Tensor): # [1]
        """
        The generate function for generating a patch based on the encoded patch and already generated tokens.
        :param encoded_patch: the encoded patch
        :param tokens: already generated tokens in the patch
        :return: the probability distribution of next token
        """
        encoded_patch = encoded_patch.reshape(1, 1, -1) # [1, 1, hidden_size]
        tokens = tokens.reshape(1, -1)
        
        # Ensure tokens is LongTensor for embedding layer
        tokens = tokens.long()

        # Get input embeddings
        tokens = torch.nn.functional.embedding(tokens, self.base.transformer.wte.weight)

        # Concatenate the encoded patch with the input embeddings
        tokens = torch.cat((encoded_patch, tokens[:,1:,:]), dim=1)
        
        # Get output from model
        outputs = self.base(inputs_embeds=tokens)
        
        # Get probabilities of next token
        probs = torch.nn.functional.softmax(outputs.logits.squeeze(0)[-1], dim=-1)

        return probs

class NotaGenLMHeadModel(PreTrainedModel):
    """
    NotaGen is a language model with a hierarchical structure.
    It includes a patch-level decoder and a char-level decoder.
    The patch-level decoder is used to generate patch features in an auto-regressive manner.
    The char-level decoder is used to generate the chars within each patch in an auto-regressive manner.
    It inherits PreTrainedModel from transformers.
    """
    def __init__(self, encoder_config, decoder_config):
        super().__init__(encoder_config)
        self.special_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.patch_level_decoder = PatchLevelDecoder(encoder_config)
        self.char_level_decoder = CharLevelDecoder(decoder_config)

    def forward(self,
                patches: torch.Tensor,
                masks: torch.Tensor):
        """
        The forward pass of the bGPT model.
        :param patches: the patches to be encoded
        :param masks: the masks for the patches
        :return: the decoded patches
        """
        patches = patches.reshape(len(patches), -1, PATCH_SIZE)
        encoded_patches = self.patch_level_decoder(patches, masks)["last_hidden_state"]

        left_shift_masks = masks * (masks.flip(1).cumsum(1).flip(1) > 1)
        masks[:, 0] = 0

        encoded_patches = encoded_patches[left_shift_masks == 1]
        patches = patches[masks == 1]

        return self.char_level_decoder(encoded_patches, patches)

    def generate(self,
                 patches: torch.Tensor,
                 top_k:int=0,
                 top_p:float=1.0,
                 temperature:float=1.0):
        """
        The generate function for generating patches based on patches.
        :param patches: the patches to be encoded
        :param top_k: the top k for sampling
        :param top_p: the top p for sampling
        :param temperature: the temperature for sampling
        :return: the generated patches
        """
        if patches.shape[-1] % PATCH_SIZE != 0:
            tokens = patches[:,:,-(patches.shape[-1]%PATCH_SIZE):].squeeze(0, 1)
            tokens = torch.cat((torch.tensor([self.bos_token_id], device=self.device), tokens), dim=-1)
            patches = patches[:,:,:-(patches.shape[-1]%PATCH_SIZE)]
        else:
            tokens =  torch.tensor([self.bos_token_id], device=self.device)

        patches = patches.reshape(len(patches), -1, PATCH_SIZE) # [bs, seq, patch_size]
        encoded_patches = self.patch_level_decoder(patches)["last_hidden_state"]    # [bs, seq, hidden_size]
        generated_patch = []            

        while True:
            prob = self.char_level_decoder.generate(encoded_patches[0][-1], tokens).cpu().detach().numpy()  # [128]
            prob = top_k_sampling(prob, top_k=top_k, return_probs=True) # [128]
            prob = top_p_sampling(prob, top_p=top_p, return_probs=True) # [128]
            token = temperature_sampling(prob, temperature=temperature) # int
            char = chr(token)
            generated_patch.append(token)

            if len(tokens) >= PATCH_SIZE:# or token == self.eos_token_id:
                break
            else:
                tokens = torch.cat((tokens, torch.tensor([token], device=self.device)), dim=0)
        
        return generated_patch


