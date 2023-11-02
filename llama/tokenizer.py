"""
Llama Tokenizer
===============
This module contains the Tokenizer class that wraps the SentencePiece tokenizer.
"""

from typing import List
from sentencepiece import SentencePieceProcessor  # type: ignore

class Tokenizer:
    """
    Llama Tokenizer Class
    ---------------------
    This class provides a wrapper around the SentencePiece tokenizer.
    It adds some utility functions for easier encoding and decoding.

    Attributes:
        bos_id (int): The id representing the "beginning of sentence" token.
        eos_id (int): The id representing the "end of sentence" token.
        pad_id (int): The id representing the padding token.
        vocab_size (int): The size of the vocabulary.
    """

    def __init__(self, model_path: str):
        """
        Initialize the Tokenizer.

        Args:
            model_path (str): The path to the SentencePiece model file.

        Returns:
            None
        """
        sp = SentencePieceProcessor(model_file=model_path)

        self.bos_id: int = sp.bos_id()
        self.eos_id: int = sp.eos_id()
        self.pad_id: int = sp.pad_id()
        self.vocab_size: int = sp.vocab_size()

        self.sp = sp

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        """
        Encode a string as a sequence of token IDs.

        Args:
            s (str): The string to be encoded.
            bos (bool, optional): Whether to add a "beginning of sentence" token. Defaults to False.
            eos (bool, optional): Whether to add an "end of sentence" token. Defaults to False.

        Returns:
            List[int]: The list of token IDs.
        """
        tokens = []

        if bos:
            tokens.append(self.bos_id)

        tokens.extend(self.sp.encode(s))

        if eos:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """
        Decode a sequence of token IDs to a string.

        Args:
            tokens (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.sp.decode(tokens)

    def id_to_piece(self, token: int) -> str:
        """
        Convert a token ID to its corresponding token string.

        Args:
            token (int): The token ID.

        Returns:
            str: The token string, with SentencePiece's '▁' character replaced by a space.
        """
        return self.sp.id_to_piece(token).replace('▁', ' ')
