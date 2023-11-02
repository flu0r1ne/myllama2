"""
The core model for the Llama family of LLMs
"""

import math
import copy

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn.functional as F

from torch import nn
from .llm import LLM

# pylint: disable=locally-disabled, R0902, R0913

def _round_up_to_multiple(n: int, m: int) -> int:
    """
    Round n up to an integer multiple of m
    """
    return math.ceil(n / m) * m

@dataclass
class LlamaArgs:
    """
    Arguments class for configuring a LLAMA model.

    Attributes:
       dim (int): The model dimension, typically referred to as d_model in
        "Attention is All You Need" paper.
       n_layers (int): The number of layers in the model.
       n_heads (int): The number of attention heads in each layer.
       vocab_size (int): The size of the model's vocabulary.
       multiple_of (int): Ensures the feed-forward network dimension (d_ff)
       is a multiple of this factor.
       norm_eps (float): The epsilon value for RMS normalization, avoiding
       division by zero.
       max_ctx_len (int, optional): The maximum context length the model can
       handle. Defaults to 2048.
       max_batch_size (int, optional): The maximum batch size supported by the
       model's cache. Defaults to 1.
       n_groups (Optional[int], optional): The number of key-value groups in
       grouped query-attention (GQA), if applicable. Defaults to None.
       padding_idx (int): The index used for padding in embeddings. Defaults
       to -1.
    """

    dim: int
    n_layers: int
    n_heads: int
    vocab_size: int
    multiple_of: int
    norm_eps: float
    max_ctx_len: int = 2048
    max_batch_size: int = 1
    n_groups: Optional[int] = None
    padding_idx: int = -1

class RMSNorm(nn.Module):
    """
    Implements an unbiased Root Mean Square (RMS) Layer Normalization.

    Reference:
    See the paper "Root Mean Square Layer Normalization" at
    https://arxiv.org/pdf/1910.07467.pdf for more details.

    Attributes:
        eps (float): A small epsilon value added to the denominator for
        numerical stability.
        gain (nn.Parameter): A learnable gain parameter applied after
        normalization.
    """

    def __init__(self, d: int, eps: float = 1e-6, dtype: torch.dtype = torch.float):
        """
        Initializes the RMSNorm layer.

        Args:
            d (int): The dimensionality of the input feature space.
            eps (float, optional): A small epsilon value to add to the
            denominator for numerical stability. Defaults to 1e-6.
            dtype (torch.dtype, optional): The data type of the learnable gain
            parameter. Defaults to torch.float.
        """
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d, dtype=dtype))


    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS normalization to the input tensor.

        Args:
            a (torch.Tensor): The input tensor to be normalized.

        Returns:
            torch.Tensor: The normalized tensor with the same shape as the input.
        """

        inverse_rms = torch.rsqrt(self.eps + torch.mean(a ** 2, dim=-1, keepdim=True))

        return a * inverse_rms * self.gain


class SwiGLU(nn.Module):
    """
    Implements the SwiGLU variant of the Gated Linear Unit (GLU) as part of the
    FFN layer of a transformer. SwiGLU is a variant of the Gated Linear Unit
    where the gating mechanism is controlled by a Swish activation function.

    Reference:
    The SwiGLU activation function is detailed in the paper "GLU Variants Improve Transformer"
    which can be accessed at https://arxiv.org/pdf/2002.05202.pdf.
    """

    def __init__(self, dim : int, dim_ff: int, dtype: torch.dtype = torch.float):
        """
        Initializes the SwiGLU module.

        Arguments:
            dim (int): The dimensionality of the input and output tensors.
            dim_ff (int): The reduced dimensionality of the hidden layer.
            dtype (torch.dtype, optional): The data type for the weights of
            the linear transformations. Defaults to torch.float.
        """
        super().__init__()

        self.w = nn.Linear(dim, dim_ff, bias=False, dtype=dtype)
        self.v = nn.Linear(dim, dim_ff, bias=False, dtype=dtype)
        self.w2 = nn.Linear(dim_ff, dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the SwiGLU feed-forward layer

        Arguments:
            x (torch.Tensor): The input tensor to the SwiGLU module.

        Returns:
            torch.Tensor: The output tensor after applying the SwiGLU operation.
        """
        return self.w2(F.silu(self.w(x)) * self.v(x))

class RotaryEmbeddings(nn.Module):
    """
    Implementation of rotary position embeddings.

    Rotary embeddings are a mechanism for injecting positional information into
    transformer models. These embeddings apply a rotation to the key and value
    vectors in the attention mechanism based on their position, with different
    "dampening" factors applied based on the relative distance between two tokens.

    Args:
        - dim (int): The dimension of the embeddings.
        - max_ctx_len (int): The maximum length of the context for which to compute
        the embeddings.
        - theta (float, optional): The frequency parameter for computing the rotary
        embeddings. Defaults to 10000.0.

    Raises:
        AssertionError: If the dimension is not even.

    References:
        - RoFormer paper: https://arxiv.org/pdf/2104.09864.pdf
    """

    embedding_cache: Dict[int, torch.Tensor] = {}

    def __init__(self, dim: int, max_ctx_len: int, theta: float = 10000.0):
        """
        Initialize the RotaryEmbeddings module.

        Args:
            - dim (int): The dimension of the embeddings.
            - max_ctx_len (int): The maximum length of the context for which
            to compute the embeddings.
            - theta (float, optional): The frequency parameter for computing
            the rotary embeddings. Defaults to 10000.0.

        Raises:
            AssertionError: If the dimension is not even.
        """
        super().__init__()

        assert dim % 2 == 0, "Model dimension should be a multiple of two"

        self.n_coord_pairs = dim // 2
        self.rots = RotaryEmbeddings.get_embeddings(dim, max_ctx_len, theta)

    @staticmethod
    def compute_angles(dim: int, max_ctx_len: int, theta: float) -> torch.Tensor:
        """
        Compute the rotation angles for the embeddings.

        Arguments:
            dim (int): The dimension of the embeddings.
            max_ctx_len (int): The maximum context length.
            theta (float): The frequency parameter for the embeddings.

        Returns:
            torch.Tensor: A tensor of shape (max_ctx_len, dim // 2) containing the
            rotation angles.
        """

        freqs = theta ** (-torch.arange(0, dim, 2, dtype=torch.float) / dim)

        m = torch.arange(max_ctx_len)

        angles = torch.outer(m, freqs)

        return torch.polar(torch.ones((max_ctx_len, dim // 2)), angles)

    @staticmethod
    def get_embeddings(dim: int, max_ctx_len: int, theta: float) -> torch.Tensor:
        """
        Retrieve or compute and cache the rotary embeddings.

        Args:
            - dim (int): The dimension of the embeddings.
            - max_ctx_len (int): The maximum context length.
            - theta (float): The frequency parameter for the embeddings.

        Returns:
            - torch.Tensor: A tensor containing the precomputed embeddings.
        """

        cache = RotaryEmbeddings.embedding_cache

        if dim not in cache:

            cache[dim] = \
                    RotaryEmbeddings.compute_angles(dim, max_ctx_len, theta)

        return cache[dim]

    def forward(self, x: torch.Tensor, cur_pos: int = 0) -> torch.Tensor:
        """
        Apply the rotary embeddings to the input tensor.

        Arguments:
            - x (torch.Tensor): A tensor of shape (batch_size, ctx_len, ..., dim)
            representing input features.
            - cur_pos (int, optional): The current position index from which to
            apply rotations. Defaults to 0.

        Returns:
             - torch.Tensor: The rotated tensor with the same shape as the input.
        """

        _batch_size, ctx_len, *dup_dims, dim = x.shape

        rotated = x.view(*x.shape[:-1], self.n_coord_pairs, 2)
        rotated = torch.view_as_complex(rotated.float())

        broad_shape = [1, ctx_len] + [1] * len(dup_dims) + [ dim // 2 ]

        rotated *= self.rots[cur_pos : cur_pos + ctx_len].view(*broad_shape)

        rotated = torch.view_as_real(rotated)
        rotated = rotated.view(*x.shape[:-1], dim).type_as(x)

        return rotated

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              mask: Optional[torch.Tensor] = None) \
              -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the scaled dot product attention.

    This function takes as input the query (Q), key (K), value (V) tensors,
    and an optional mask, and returns the attention output and attention
    weights.

    Arguments:
    - q (torch.Tensor): The query tensor of shape (..., seq_len, d_k).
    - k (torch.Tensor): The key tensor of shape (..., seq_len, d_k).
    - v (torch.Tensor): The value tensor of shape (..., seq_len, d_v).
    - mask (Optional[torch.Tensor]): An optional mask tensor to apply to
      the scores before softmax.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor]: A tuple consisting of the attention
      output tensor and the attention weights tensor.

    References:
    - "Attention Is All You Need": https://arxiv.org/pdf/1706.03762.pdf
    """

    d_k = q.size(-1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn = F.softmax(scores, dim=-1)

    return torch.matmul(attn, v), attn

class LinearCache:
    """
        A simple linear-cache. This is used to cache the attention
        keys and values.
    """

    def __init__(self, max_batch_size: int, max_context_len: int,
                 tensor_dims: Tuple, dtype: torch.dtype = torch.float):
        """Initializes the LinearCache with given dimensions and data type."""

        self.max_batch_size = max_batch_size
        self.max_context_len = max_context_len
        self.cache = torch.zeros(
            (max_batch_size, max_context_len, *tensor_dims),
            dtype=dtype,
        )
        self.cached_batch_size = 0

    def get(self, pos: int) -> torch.Tensor:
        """Retrieves the cached values up to a given sequence position."""
        return self.cache[:self.cached_batch_size, :pos]

    def set(self, current_pos: int, seq: torch.Tensor) -> None:
        """Updates the cache with new sequences at the specified position."""

        batch_size, ctx_len, *_ = seq.shape

        self.cache[:batch_size, current_pos:current_pos+ctx_len] = seq

        self.cached_batch_size = batch_size

class GQA(nn.Module):
    """
        Group-Query Attention (GQA) module for transformer architectures.

        References:
        - See "GQA: Training Generalized Multi-Query Transformer Models from
        Multi-Head Checkpoints" at https://arxiv.org/pdf/2305.13245.pdf
    """

    def __init__(self, dim: int, n_heads: int,
                 n_groups: Optional[int] = None,
                 query_embedding: Optional[nn.Module] = None,
                 key_embedding: Optional[nn.Module] = None,
                 apply_decoder_mask: bool = False,
                 kv_caches: Optional[Tuple[LinearCache, LinearCache]] = None,
                 dtype: torch.dtype = torch.float):
        """
        Initializes the Group-Query Attention (GQA) module.

        Parameters:
            dim (int): The dimensionality of the input features and the last dimension of
                       the output tensor.
            n_heads (int): The number of attention heads to use.
            n_groups (Optional[int]): The number of groups to divide the attention heads
                                      into. If not specified, defaults to the number of heads.
                                      Must divide `n_heads` evenly.
            query_embedding (Optional[nn.Module]): An optional module to embed the query
                                                   vectors, e.g., a positional encoding module.
            key_embedding (Optional[nn.Module]): An optional module to embed the key vectors,
                                                 similar to `query_embedding`.
            apply_decoder_mask (bool): Whether to apply a causal mask to the attention mechanism,
                                       useful for decoder self-attention.
            kv_caches (Optional[Tuple[LinearCache, LinearCache]]): Optional tuple of
                                                                   `LinearCache` instances for
                                                                   caching key and value projections
                                                                   in an autoregressive setting.
            dtype (torch.dtype): The data type of the module's parameters, e.g., `torch.float32`.
                                 The cache tensors should also use this data type.
        """

        n_groups = n_groups if n_groups else n_heads

        assert dim % n_heads == 0, \
            "Model dimension should be a multiple of n_heads"
        assert n_heads % n_groups == 0, \
            "n_heads should be a multiple of n_groups"

        super().__init__()

        head_dim = dim // n_heads

        self.n_heads = n_heads
        self.n_groups = n_groups
        self.head_dim = head_dim
        self.apply_decoder_mask = apply_decoder_mask

        self.query_embedding = query_embedding
        self.key_embedding = key_embedding

        self.wq = nn.Linear(
            dim,
            n_heads * head_dim,
            bias=False,
            dtype=dtype,
        )

        self.wk = nn.Linear(
            dim,
            n_groups * head_dim,
            bias=False,
            dtype=dtype,
        )

        self.wv = nn.Linear(
            dim,
            n_groups * head_dim,
            bias=False,
            dtype=dtype,
        )

        self.wo = nn.Linear(
            n_heads * head_dim,
            dim,
            bias=False,
            dtype=dtype,
        )

        if kv_caches is not None:
            self.key_cache = kv_caches[0]
            self.value_cache = kv_caches[1]
            self.has_cache = True
        else:
            self.has_cache = False

    def forward(self, x: torch.Tensor, cur_pos: int):
        """
        Processes the input tensor with Group-Query Attention.

        Arguments:
            - x (torch.Tensor): The input tensor of shape
            (batch_size, context_length, dim).
            - cur_pos (int): The current position in the sequence for which
            to compute attention. This is relevant when using key-value caches,
            as it determines the part of the cache to update and utilize.

        Returns:
            - torch.Tensor: The output tensor after applying Group-Query Attention.
        """

        batch_size, ctx_len, dim = x.shape

        # Perform key, query, and value projections

        # wq(x) performs all n_heads projections at once, then the result
        # is reshaped such that the first head_dim results are part of the first
        # head, the second head_dim results are part of the second head, and so
        # on.
        q = self.wq(x).view(batch_size, ctx_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch_size, ctx_len, self.n_groups, self.head_dim)
        v = self.wv(x).view(batch_size, ctx_len, self.n_groups, self.head_dim)

        # Apply embeddings to the key and query matrices
        if self.query_embedding:
            q = self.query_embedding(q, cur_pos)

        if self.key_embedding:
            k = self.key_embedding(k, cur_pos)

        if self.has_cache:
            # Add the new embeddings to the cache
            self.key_cache.set(cur_pos, k)
            self.value_cache.set(cur_pos, v)

            # Get all the previous embedding from the cache.

            # Note if cur_pos != 0, ctx_len is the length of
            # the new sequence. In reality, the whole sequence
            # is cur_pos + ctx_len and both cached results will
            # be of size (batch_size, ctx_len + cur_pos, n_groups, head_dim)
            k = self.key_cache.get(cur_pos + ctx_len)
            v = self.value_cache.get(cur_pos + ctx_len)

        # Avoid copy if multi-head attention MHA is used. This is true in the
        # 7B and 13B models.
        if self.n_groups != self.n_heads:

            repeats = self.n_heads // self.n_groups

            # Duplicate grouped attention heads:

            # From: { G_0, G_1, ... G_{k - 1} }
            # To: { G_0, G_0, ... G_0, G_1, ..., G_{k - 1}, G_{k - 1}, ..., G_{k - 1}
            k = torch.repeat_interleave(k, dim=2, repeats=repeats)
            v = torch.repeat_interleave(v, dim=2, repeats=repeats)

        # Transpose to parallelize attention across heads during batched-matrix
        # multiplication
        q = q.transpose(1, 2) # (batch_size, n_heads, ctx_len, head_dim)
        k = k.transpose(1, 2) # (batch_size, n_heads, ctx_len, head_dim)
        v = v.transpose(1, 2) # (batch_size, n_heads, ctx_len, head_dim)

        if self.apply_decoder_mask:
            # Construct attention mask

            # In the decoder architecture, the mask is a lower triangular matrix that prevents
            # previous tokens from attending to subsequent ones. More concretely for attention
            # scores (i, j), token i cannot attend to token j if j > i.

            # When key-value caching is enabled, we are only computing the attention scores
            # for the new sequence. Thus, the matrix of scores is of size (ctx_len, total_len)
            # and the only masked entries are (i, j) for j > cached_len + i since row i really
            # represents token cached_len + i.
            mask = torch.hstack([
                torch.ones((ctx_len, cur_pos)),
                torch.tril(torch.ones((ctx_len, ctx_len))),
            ])
        else:
            mask = None

        # Perform attention
        x, _ = attention(q, k, v, mask)

        # Concatenate heads
        x = x.transpose(1, 2) # (batch_size, ctx_len, n_heads, head_dim)
        x = x.reshape((batch_size, ctx_len, dim))

        # Final linear layer
        x = self.wo(x)

        return x

class LlamaTransformerLayer(nn.Module):
    """
      This constitutes a single transformer block within Meta's Llama architecture.

      The transformer architecture combines group-query attention (GQA) and key-value caching.

      It also utilizes RMSNorm to decrease co-variance shifts during training and skip connections
      which make training easier.
    """

    def __init__(self, dim: int, n_heads: int, n_groups: Optional[int], max_context_len: int,
                 max_batch_size: int, round_ff_to_multiple: int, eps: float = 1e-6):
        """Initializes a layer of the Lamma transformer."""

        super().__init__()

        head_dim = dim // n_heads

        self.query_embedding = RotaryEmbeddings(head_dim, max_context_len)
        self.key_embedding = RotaryEmbeddings(head_dim, max_context_len)

        cache_size = n_groups if n_groups else n_heads

        self.key_cache = LinearCache(
            max_batch_size, max_context_len, (cache_size, head_dim), dtype=torch.bfloat16
        )
        self.value_cache = LinearCache(
            max_batch_size, max_context_len, (cache_size, head_dim), dtype=torch.bfloat16
        )

        self.gqa = GQA(
            dim, n_heads, n_groups,
            query_embedding=self.query_embedding,
            key_embedding=self.key_embedding,
            kv_caches=(self.key_cache, self.value_cache),
            dtype=torch.bfloat16,
            apply_decoder_mask=True,
        )

        # It might have been better to specify the inner "hidden" feed-forward
        # dimension directly as a hyper parameter. It seems that FAIR chose
        # this odd ratio from the [SwiGLU paper](https://arxiv.org/pdf/2002.05202.pdf)
        # directly. This seems slightly odd as this ratio was initially used only for
        # the purposes of enabling a fair comparison across different feed-forward
        # configurations.
        dim_ff = _round_up_to_multiple(4 * int(2 * dim / 3), round_ff_to_multiple)

        self.feed_forward = SwiGLU(dim, dim_ff, dtype=torch.bfloat16)

        self.attention_norm = RMSNorm(dim, eps, dtype=torch.bfloat16)
        self.forward_norm = RMSNorm(dim, eps, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor, cur_pos: int = 0) -> torch.Tensor:
        """
        Takes as an input the input embeddings or previous decoder output
        and produces the output of this decoder
        """

        # RMS Norm
        x_norm = self.attention_norm(x)
        # GQA with a skip connection
        # See ResNet at https://arxiv.org/pdf/1512.03385.pdf for skip connections
        h = x + self.gqa(x_norm, cur_pos=cur_pos)
        # RMS Norm
        h_norm = self.forward_norm(h)
        # SwiGLU feed-forward with a skip connection
        h = h + self.feed_forward(h_norm)

        return h

class LlamaDecoderStack(nn.Module):
    """
    The decoder stack is a stack of n_layers of decoders.
    """

    def __init__(self, args: LlamaArgs):
        """Initializes the decoder stack"""

        super().__init__()

        layer = LlamaTransformerLayer(
            args.dim, args.n_heads, args.n_groups, args.max_ctx_len,
            args.max_batch_size, args.multiple_of, args.norm_eps
        )

        self.decoders = nn.ModuleList([
            copy.deepcopy(layer) for _ in range(args.n_layers)
        ])

    def forward(self, embedding: torch.Tensor, cur_pos: int = 0) -> torch.Tensor:
        """Apply all encoders, obtaining the outputs of the last decoder"""

        h = embedding

        for decoder in self.decoders:
            h = decoder(h, cur_pos)

        return h

class LlamaEmbeddings(nn.Module):
    """
        LlamaEmbeddings transform a tensor of token ids into embedding vectors of size dim
    """

    def __init__(self, vocab_size: int, dim: int, padding_idx: int):
        """Initializes the LlamaEmbeddings"""

        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(self.vocab_size, self.dim, dtype=torch.bfloat16)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Retrieve the embeddings for a token sequence"""

        # The original Llama implementation employs parallel embeddings. This
        # implicitly produces zero embeddings for padding_idx = -1. This behavior
        # is seemingly undefined and relies on implementation details within
        # the parallel embeddings.

        # Since nn.Embedding does not handle negative indices, we must manually
        # zero out the padded parts of the context.
        padding_mask = context == torch.tensor(self.padding_idx, dtype=torch.long)
        context[padding_mask] = torch.tensor(0, dtype=torch.long)

        embeddings = self.embedding(context)

        embeddings[padding_mask] = torch.zeros((self.dim,), dtype=embeddings.dtype)

        return embeddings

class Llama(LLM):
    """An class representing the Llama family of LLMs"""

    def __init__(self, args : LlamaArgs):
        """Initialize the Llama model"""

        super().__init__()

        self.context_length = args.max_ctx_len
        self.max_batch_size = args.max_batch_size

        self.embeddings = LlamaEmbeddings(
            args.vocab_size, args.dim, args.padding_idx
        )

        self.decoder_stack = LlamaDecoderStack(args)

        self.output_norm = RMSNorm(
            args.dim, eps=args.norm_eps, dtype=torch.bfloat16
        )

        self.vocab_map = nn.Linear(
            args.dim, args.vocab_size, bias=False, dtype=torch.bfloat16
        )

    def forward(self, context: torch.Tensor, cur_pos: int = 0) -> torch.Tensor:
        """
        Computes the log probabilities of the next token given a sequence of
        tokens as context.

        Args:
            context (torch.Tensor): A tensor of shape (batch_size, context_length)
                                   containing token ids. These tokens serve as the
                                   context for predicting the next token.

            cur_pos (int, optional): The position at which to start the
                                       prediction. If cur_pos is not zero,
                                       the internal cache (if available) will
                                       be used to speed up predictions.
                                       Defaults to 0.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, vocab_size) containing
                          the log probabilities of the next token given the
                          context.

        Examples:
            # Predict the next token for a sequence [1, 2, 3]
            log_probs = llm(torch.tensor([[1, 2, 3]], dtype=torch.long), 0)

            # Predict the next token for a sequence [1, 2, 3, 4, 5] using the
            # cache starting at position 3
            log_probs = llm(torch.tensor([[4, 5]], dtype=torch.long), 3)
        """

        embeddings = self.embeddings(context) # ( n_batches, n_embeddings, dim )

        h = self.decoder_stack(embeddings, cur_pos)

        h = self.output_norm(h)

        vocab_logits = self.vocab_map(h)

        return vocab_logits[:,-1]
