# File: /home/admin/locked/SimpleRNATokenizer.py
"""
Simple RNA Tokenizer - Character-Level Baseline

Single-nucleotide tokenization with fixed vocabulary of 11 tokens:
- 7 special tokens: <pad>, <start>, <end>, <mask>, <unk>, <msa_sep>, <gap>
- 4 nucleotides: A, U, G, C

This tokenizer requires NO training - vocabulary is fixed.
"""

import os
import json
from typing import List, Dict, Optional


class MinimalRNATokenizer:
    """
    Character-level RNA tokenizer with fixed vocabulary.
    
    Vocabulary (11 tokens):
        0: <pad>    - Padding token
        1: <start>  - Sequence start
        2: <end>    - Sequence end
        3: <mask>   - MLM mask token
        4: <unk>    - Unknown token
        5: <msa_sep> - MSA separator
        6: <gap>    - Gap character
        7: A        - Adenine
        8: U        - Uracil
        9: G        - Guanine
        10: C       - Cytosine
    """
    
    # - Fixed vocabulary
    SPECIAL_TOKENS = {
        '<pad>': 0,
        '<start>': 1,
        '<end>': 2,
        '<mask>': 3,
        '<unk>': 4,
        '<msa_sep>': 5,
        '<gap>': 6,
    }
    
    NUCLEOTIDES = {
        'A': 7,
        'U': 8,
        'G': 9,
        'C': 10,
    }
    
    def __init__(self):
        """Initialize tokenizer with fixed vocabulary."""
        # Build vocabulary
        self.vocab = {}
        self.vocab.update(self.SPECIAL_TOKENS)
        self.vocab.update(self.NUCLEOTIDES)
        
        # Reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Token IDs for convenience
        self.pad_token_id = self.SPECIAL_TOKENS['<pad>']
        self.start_token_id = self.SPECIAL_TOKENS['<start>']
        self.end_token_id = self.SPECIAL_TOKENS['<end>']
        self.mask_token_id = self.SPECIAL_TOKENS['<mask>']
        self.unk_token_id = self.SPECIAL_TOKENS['<unk>']
        self.msa_sep_token_id = self.SPECIAL_TOKENS['<msa_sep>']
        self.gap_token_id = self.SPECIAL_TOKENS['<gap>']
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size (always 11)."""
        return len(self.vocab)
    
    def _normalize(self, sequence: str) -> str:
        """Normalize sequence: uppercase, T->U."""
        return sequence.upper().replace('T', 'U')
    
    def tokenize(self, sequence: str, apply_length_policy: bool = False) -> List[str]:
        """
        Tokenize RNA sequence into single nucleotides.
        
        Args:
            sequence: RNA sequence string
            apply_length_policy: Ignored (for API compatibility)
        
        Returns:
            List of single-character tokens
        """
        sequence = self._normalize(sequence)
        tokens = []
        
        for char in sequence:
            if char in self.NUCLEOTIDES:
                tokens.append(char)
            elif char == '-':
                tokens.append('<gap>')
            elif char == 'N':
                # Handle ambiguous nucleotide as unknown
                tokens.append('<unk>')
            else:
                # Unknown character
                tokens.append('<unk>')
        
        return tokens
    
    def encode_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to IDs.
        
        Args:
            tokens: List of token strings
        
        Returns:
            List of token IDs
        """
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]
    
    def encode(self, sequence: str, add_special: bool = False) -> List[int]:
        """
        Encode sequence to token IDs.
        
        Args:
            sequence: RNA sequence string
            add_special: If True, add <start> and <end> tokens
        
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(sequence)
        ids = self.encode_to_ids(tokens)
        
        if add_special:
            ids = [self.start_token_id] + ids + [self.end_token_id]
        
        return ids
    
    def encode_sequence(self, sequence: str, add_special: bool = True) -> List[int]:
        """Alias for encode() with add_special=True by default."""
        return self.encode(sequence, add_special=add_special)
    
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs back to sequence.
        
        Args:
            ids: List of token IDs
            skip_special: If True, skip special tokens in output
        
        Returns:
            Decoded RNA sequence string
        """
        tokens = []
        special_ids = set(self.SPECIAL_TOKENS.values())
        
        for id in ids:
            if skip_special and id in special_ids:
                continue
            token = self.id_to_token.get(id, '<unk>')
            if token not in self.SPECIAL_TOKENS:
                tokens.append(token)
        
        return ''.join(tokens)
    
    def decode_from_ids(self, ids: List[int]) -> str:
        """Alias for decode()."""
        return self.decode(ids, skip_special=True)
    
    def get_stats(self) -> Dict:
        """Get tokenizer statistics."""
        return {
            'vocab_size': self.vocab_size,
            'num_special_tokens': len(self.SPECIAL_TOKENS),
            'num_nucleotides': len(self.NUCLEOTIDES),
            'tokenizer_type': 'simple_character_level',
        }
    
    def save(self, save_dir: str):
        """Save tokenizer configuration."""
        os.makedirs(save_dir, exist_ok=True)
        
        config = {
            'tokenizer_type': 'SimpleRNATokenizer',
            'vocab_size': self.vocab_size,
            'vocab': self.vocab,
            'special_tokens': self.SPECIAL_TOKENS,
            'nucleotides': self.NUCLEOTIDES,
        }
        
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, load_dir: str) -> 'SimpleRNATokenizer':
        """Load tokenizer (returns new instance - vocab is fixed)."""
        return cls()
    
    @classmethod
    def load_checkpoint(cls, checkpoint_dir: str) -> 'SimpleRNATokenizer':
        """Alias for load() - API compatibility with Nu tokenizer."""
        return cls.load(checkpoint_dir)
