"""
LLM provides a generalized interface for autoregressive
next-word prediction models. The class can be utilized for tasks such as text
sampling and probability prediction over a vocabulary.
"""

import torch
from torch import nn

class LLM(nn.Module):
    """
    LLM provides a generalized interface for autoregressive
    next-word prediction models. The class can be utilized for tasks such as text
    sampling and probability prediction over a vocabulary.

    Attributes:
        context_length (int): Length of the context window for the
                              autoregressive model.  Default is -1, which
                              indicates that this needs to be set.

        max_batch_size (int): The maximum size of a batch that can be processed.
                              Default is -1, which indicates that this needs to
                              be set.

        vocab_size (int): The size of the vocabulary used in the model.
                          Default is -1, which indicates that this needs to
                          be set.

        padding_idx (int): The index used for padding in mixed-length batches.
                           Default is -1, which indicates that this needs to be
                           set.

        eos_token (int): Token index that signifies the end of a sequence during
                         auto-regressive generation. Default is -1, which
                         indicates that this needs to be set.
    """

    context_length = -1
    max_batch_size = -1
    vocab_size = -1
    padding_idx = -1
    eos_token = -1

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

        raise NotImplementedError()
