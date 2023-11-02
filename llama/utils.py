"""
Utilities for loading the Llama model and tokenizer from a checkpoint directory
and a tokenizer model file.
"""

import json
import re

from pathlib import Path
from typing import Dict, Any, Tuple

import torch

from .model import LlamaArgs, Llama
from .tokenizer import Tokenizer

ModuleParams = Dict[str, Any]

def _load_model_from_checkpoint(checkpoint_dir: str) \
        -> Tuple[LlamaArgs, ModuleParams]:
    """
    Load the Llama model from a given checkpoint directory.

    Args:
        checkpoint_dir (str): The path to the directory containing the Llama
        model checkpoint.

    Returns:
        Tuple[LlamaArgs, ModuleParams]: A tuple containing:
            - LlamaArgs: Arguments used for initializing the Llama model.
            - ModuleParams: PyTorch state dictionary for the Llama model.
    """

    checkpoint_path = Path(checkpoint_dir)
    with open(checkpoint_path / "params.json", "r", encoding='utf-8') as f:
        args = json.loads(f.read())

    args = LlamaArgs(**args)

    checkpoint_paths = list(checkpoint_path.glob('*.pth'))
    checkpoint_paths = sorted(checkpoint_paths)

    checkpoint = torch.load(checkpoint_paths[0], map_location="cpu")

    return args, checkpoint

# pylint: disable=locally-disabled, R0912
def _transform_params(params: ModuleParams) -> ModuleParams:
    """
    Map the state dictionary keys from the official Llama model to the keys
    used in this implementation.

    Args:
        params (ModuleParams): The state dictionary from the official Llama
        model.

    Returns:
        ModuleParams: The modified state dictionary to match the keys used in
        this implementation.
    """

    new_params = {}

    for label, param in params.items():

        if label == 'tok_embeddings.weight':
            label = 'embeddings.embedding.weight'
        elif label == 'norm.weight':
            label = 'output_norm.gain'
        elif label == 'output.weight':
            label = 'vocab_map.weight'
        else:

            if label in { 'rope.freqs' }:
                continue

            regex = re.compile(r'layers\.(\d+)\.(.*)')

            m = regex.match(label)

            assert m is not None

            layer_num = m.group(1)
            sub_label = m.group(2)

            label = f'decoder_stack.decoders.{layer_num}.'

            if sub_label == 'attention.wq.weight':
                label += 'gqa.wq.weight'
            elif sub_label == 'attention.wk.weight':
                label += 'gqa.wk.weight'
            elif sub_label == 'attention.wv.weight':
                label += 'gqa.wv.weight'
            elif sub_label == 'attention.wo.weight':
                label += 'gqa.wo.weight'
            elif sub_label == 'feed_forward.w1.weight':
                label += 'feed_forward.w.weight'
            elif sub_label == 'feed_forward.w2.weight':
                label += 'feed_forward.w2.weight'
            elif sub_label == 'feed_forward.w3.weight':
                label += 'feed_forward.v.weight'
            elif sub_label == 'attention_norm.weight':
                label += 'attention_norm.gain'
            elif sub_label == 'ffn_norm.weight':
                label += 'forward_norm.gain'
            else:
                assert False, "Key not found"

        new_params[label] = param

    return new_params

def load_llama_from_checkpoint(checkpoint_dir: str, tokenizer_path: str,
                               max_batch_size: int = 1, max_context_len: int = 2048) \
        -> Tuple[Llama, Tokenizer]:
    """
    Load the Llama model and the tokenizer from specified paths.

    Args:
        checkpoint_dir (str): Path to the directory containing the model
                              checkpoint.
        tokenizer_path (str): Path to the tokenizer model file.
        max_batch_size (int, optional): Maximum batch size the model can accept.
                                        Affects cache size. Default is 1.
        max_context_len (int, optional): Maximum context length the model can
                                        handle. Affects cache size. Default is
                                        2048.

    Returns:
        Tuple[Llama, Tokenizer]: A tuple containing:
            - Llama: The Llama model loaded from the checkpoint.
            - Tokenizer: The tokenizer model loaded from the given path.
    """

    args, checkpoint = _load_model_from_checkpoint(checkpoint_dir)

    tokenizer = Tokenizer(tokenizer_path)

    args.vocab_size = tokenizer.vocab_size
    args.max_batch_size = max_batch_size
    args.max_ctx_len = max_context_len
    args.padding_idx = tokenizer.pad_id

    checkpoint = _transform_params(checkpoint)

    llama_model = Llama(args)
    llama_model.load_state_dict(checkpoint)

    return llama_model, tokenizer

__all__ = [
    'load_llama_from_checkpoint'
]
