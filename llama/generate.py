"""
This module provides an interface for generating and sampling token sequences from a language model.
It allows for the controlled generation of text by specifying parameters such as temperature, top-k,
and top-p, which influence the randomness and quality of the generated sequences.
"""

# pylint: disable=locally-disabled, R0913, R0914

from collections import OrderedDict
from typing import List, Optional, Generator, cast

import torch

from .llm import LLM

TokenList = OrderedDict[int, float]

def _sample_internal(llm: LLM, context: torch.Tensor) -> torch.Tensor:
    """
    Sample a tensor of logits from the language model (LLM) based on the input context.
    """

    batch_size, seq_len = context.shape

    assert seq_len <= llm.context_length
    assert batch_size <= llm.max_batch_size

    with torch.inference_mode():
        return llm(context)

def _load_context(tokens: List[List[int]], pad_id: int,
                  pad_to_length: Optional[int] = None) -> torch.Tensor:
    """
    Load a batch of token lists into a padded tensor suitable for input to a language model.
    """
    batch_size = len(tokens)

    max_token_len = max((len(tok) for tok in tokens))

    pad_to_length = max_token_len if pad_to_length is None else pad_to_length

    context = torch.full(
        (batch_size, pad_to_length), pad_id, dtype=torch.long
    )

    for dim, toks in enumerate(tokens):
        context[dim, :len(toks)] = torch.tensor(toks, dtype=torch.long)

    return context

def batched_token_probabilities(llm: LLM,
                                tokens: List[List[int]],
                                temperature: float = 1.0) -> List[TokenList]:
    """
    Calculate the probabilities of the next token sequence across a batch of sequences.

    Args:
    - llm (LLM): An instance of the language model.
    - tokens (List[List[int]]): A list of tokenized input sequences.
    - temperature (float): A temperature parameter to scale the logits before
                           applying softmax. Default is 1.0.

    Returns:
    - List[TokenList]: A list of ordered dictionaries where each dictionary maps
                       token ids to their corresponding probabilities for each
                       sequence in the batch.
    """

    context = _load_context(tokens, llm.padding_idx)

    token_logprobs = _sample_internal(llm, context)

    token_probs = torch.softmax(token_logprobs / temperature, dim=-1)

    samples: List[TokenList] = [OrderedDict() for _ in range(len(tokens))]
    for i, p in enumerate(token_probs):
        for _id in torch.argsort(p, descending=True):
            samples[i][int(_id)] = float(p[_id])

    return samples

def token_probabilities(llm: LLM, tokens: List[int]) -> TokenList:
    """
    Calculate the probabilities of the next token sequence.
    See batched_token_probabilities.
    """

    return batched_token_probabilities(llm, [ tokens ])[0]

def sample_batched_token(
    token_logprobs: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    sample_eps: float = 1e-6) -> torch.Tensor:
    """
    Sample a token from a batch of token logits with optional top-k and top-p filtering.


    Args:
    - token_logprobs (torch.Tensor): A tensor of token log probabilities.
    - temperature (float): A scaling factor for logits before sampling. Default
                           is 1.0.
    - top_k (Optional[int]): If set, the sampling is restricted to the top k
                             tokens. Default is None (no restriction).
    - top_p (float): If set, the sampling is restricted to the smallest set
                     of tokens with cumulative probability exceeding top_p.
                     Default is 1.0 (no restriction).
    - sample_eps (float): An epsilon value to avoid precision errors during
                          cumulative probability calculation. Default is 1e-6.

    Returns:
    - torch.Tensor: A tensor of sampled token ids for each item in the batch.

    Implements both top-k sampling, top-p sampling, and beam search.

    See:
      - https://arxiv.org/pdf/1805.04833.pdf for top-k sampling
      - https://arxiv.org/pdf/1904.09751.pdf for top-p sampling
    """

    batch_size = token_logprobs.shape[0]

    token_probs = torch.softmax(token_logprobs / temperature, dim=-1)


    selected_tokens = torch.zeros(batch_size, dtype=torch.long)

    sorted_tokens = torch.argsort(token_probs, descending=True)
    sorted_probs = torch.gather(token_probs, 1, sorted_tokens)
    nucleus_mask = sorted_probs.cumsum(dim=-1) < top_p + sample_eps
    nucleus_mask[:,0] = True

    for i, (tokens, mask, probs) in enumerate(zip(sorted_tokens, nucleus_mask, sorted_probs)):
        nucleus = tokens[mask]
        p = probs[mask]

        if top_k is not None and len(nucleus) > top_k:
            nucleus = nucleus[:top_k]
            p = p[:top_k]

        p /= p.sum(axis=0)

        token = nucleus[torch.multinomial(p, 1)]

        selected_tokens[i] = token

    return selected_tokens

def generate_batched_token_sequence(llm: LLM,
                    prompts: List[List[int]],
                    max_generation_length: Optional[int] = None,
                    temperature: float = 1.0,
                    top_k: Optional[int] = None,
                    top_p: float = 1.0) -> Generator[List[Optional[int]], None, None]:
    """
    Generate a sequence of tokens for each prompt across a sequence of batches.

    Args:
    - llm (LLM): An instance of the language model.
    - prompts (List[List[int]]): A list of tokenized input sequences (prompts).
    - max_generation_length (Optional[int]): The maximum number of tokens to
      generate for each prompt. If None, generate up to the model's maximum
      context length. Default is None.
    - temperature (float): A scaling factor for logits before sampling. Default
                           is 1.0.
    - top_k (Optional[int]): If set, restricts sampling to the top k most
                             likely tokens. Default is None (no restriction).
    - top_p (float): If set, restricts sampling to a subset of tokens with a
      cumulative probability greater than top_p. Default is 1.0
      (no restriction).

    Yields:
    - Generator[List[Optional[int]], None, None]: A generator that yields lists
      of token ids, where each list corresponds to one prompt in the batch.
      Yields none if a token was not generated during an iteration of inference.

    Raises:
    - AssertionError: If batch size exceeds the maximum allowed by the LLM, or
      if the requested generation length is too long.
    """

    batch_size = len(prompts)
    assert batch_size <= llm.max_batch_size, \
        "Batch size exceeded the maximum batch size of the LLM"

    prompt_lens = torch.tensor([len(p) for p in prompts], dtype=torch.long)
    max_prompt_len = max(prompt_lens)

    remaining_context = llm.context_length - max_prompt_len
    if max_generation_length is None:
        max_generation_length = remaining_context
    else:
        assert max_generation_length <= remaining_context, \
            "Cannot generate more tokens than exist in the context"

    eos = torch.zeros(batch_size, dtype=torch.long)
    last_pos = 0

    end_pos = max_prompt_len + max_generation_length
    context = _load_context(prompts, llm.padding_idx, pad_to_length=end_pos)

    start_pos = max(prompt_lens)

    for pos in range(start_pos, end_pos):
        log_probs = llm(context[:, last_pos:pos], last_pos)

        sampled = sample_batched_token(
            log_probs,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        in_prompt = pos < prompt_lens
        should_replace_mask = (eos == 0) & (~in_prompt)

        yield [int(sampled[i]) if should_replace_mask[i] else None for i in range(batch_size)]

        context[should_replace_mask, pos] = sampled[should_replace_mask]
        eos[(eos > 0) & (sampled == llm.eos_token)] = pos + 1
        last_pos = pos

        if (eos > 0).all():
            break

def generate_token_sequence(llm: LLM,
                            prompt: List[int],
                            max_generation_length: Optional[int] = None,
                            temperature: float = 1.0,
                            top_k: Optional[int] = None,
                            top_p: float = 1.0) -> Generator[int, None, None]:
    """
    Generate a sequence of tokens for a single prompt.
    See generate_batched_token_sequence.
    """

    for tokens in generate_batched_token_sequence(llm,
                                                  [ prompt ],
                                                  max_generation_length=max_generation_length,
                                                  temperature=temperature,
                                                  top_k=top_k,
                                                  top_p=top_p):
        yield cast(int, tokens[0])

def sample_batched_sequence(llm: LLM,
                            prompts: List[List[int]],
                            max_generation_length: Optional[int] = None,
                            temperature: float = 1.0,
                            top_k: Optional[int] = None,
                            top_p: float = 1.0,
                            include_prompt: bool = False) -> List[List[int]]:
    """
    Generate and sample a sequence of tokens for each input sequence in a batch.

    Args:
    - llm (LLM): An instance of the language model.
    - prompts (List[List[int]]): A list of tokenized input sequences (prompts).
    - max_generation_length (Optional[int]): The maximum number of tokens to
      generate for each prompt. Defaults to None, which allows the generation
      up to the model's maximum context length.
    - temperature (float): A scaling factor for logits before sampling,
      affecting the randomness of the output. Default is 1.0, with lower values
      leading to less random samples.
    - top_k (Optional[int]): Limits the sampling pool to the top k tokens
      according to the probability distribution. Default is None, indicating no
      limit.
    - top_p (float): The cumulative probability threshold for nucleus sampling;
      allows sampling from a set of high-probability tokens whose cumulative
      probability exceeds this threshold. Default is 1.0, indicating no limit.
    - include_prompt (bool): If True, includes the input prompt at the beginning
      of the generated sequence. Default is False.

    Returns:
    - List[List[int]]: A list of lists containing the sampled token sequences
      for each input prompt. The sequences include the generated tokens and,
      if include_prompt is True, the original input prompt tokens.
    """

    sampled_seqs: List[List[int]] = [[] for _ in range(len(prompts))]

    if include_prompt:
        for i, prompt in enumerate(prompts):
            sampled_seqs[i].extend(prompt)

    for generated_tokens in generate_batched_token_sequence(llm,
                                            prompts,
                                            max_generation_length,
                                            temperature,
                                            top_k,
                                            top_p):
        for i, token in enumerate(generated_tokens):
            if token is not None:
                sampled_seqs[i].append(token)

    return sampled_seqs

def sample_sequence(llm: LLM,
                    prompts: List[int],
                    max_generation_length: Optional[int] = None,
                    temperature: float = 1.0,
                    top_k: Optional[int] = None,
                    top_p: float = 1.0,
                    include_prompt: bool = False) -> List[int]:
    """
    Generate and sample a sequence of tokens for a single input sequence.
    See sample_batched_sequence for reference.
    """

    return sample_batched_sequence(
        llm, [prompts], max_generation_length, temperature,
        top_k, top_p, include_prompt
    )[0]

__all__ = [
    'token_probabilities',
    'sample_batched_token',
    'sample_sequence',
    'sample_batched_sequence',
    'generate_token_sequence',
    'generate_batched_token_sequence',
]
