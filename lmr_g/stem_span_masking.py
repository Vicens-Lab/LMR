# File: data/stem_span_masking.py
"""
Stem-Span Masking for RNA Foundation Model Training

Implements the "killer app" masking strategy that forces long-range dependency learning
by masking two distant spans that may be complementary (base-pairing candidates).

This module extends the existing RNAMLMCollator with stem-span masking capability.

Key concept: Mask pairs of regions that are:
1. Separated by >= min_dist nucleotides
2. Have similar GC content (potential stem regions)
3. Show weak complementarity signal

This forces the model to learn that distant positions are structurally related.
"""

import torch
import random
import numpy as np
from typing import List, Tuple, Set, Optional, Dict, Any


class StemSpanMasker:
    """
    Generates stem-span masking patterns for RNA sequences.
    
    A "stem" in RNA is where two distant regions pair together (like a hairpin stem).
    By masking both regions simultaneously, we force the model to use the surrounding
    context to predict complementary bases - learning structural relationships.
    
    Args:
        min_dist: Minimum distance between masked spans (default: 100nt)
        span_len: Length of each masked span (default: 8nt)
        gc_threshold: Minimum GC content for candidate regions (default: 0.5)
        complementarity_threshold: Minimum complementarity for pair selection (default: 0.4)
        max_attempts: Maximum attempts to find valid pairs (default: 100)
    """
    
    # Token mappings (adjust based on your tokenizer)
    # These match MinimalRNATokenizer: A=7, U=8, G=9, C=10
    A_TOKEN = 7
    U_TOKEN = 8
    G_TOKEN = 9
    C_TOKEN = 10
    MASK_TOKEN = 3
    
    # Complementarity mapping
    COMPLEMENT_MAP = {
        7: 8,   # A-U
        8: 7,   # U-A
        9: 10,  # G-C
        10: 9,  # C-G
    }
    
    # Wobble pairs (weaker but valid)
    WOBBLE_MAP = {
        9: 8,   # G-U (wobble)
        8: 9,   # U-G (wobble)
    }
    
    def __init__(
        self,
        min_dist: int = 100,
        span_len: int = 8,
        gc_threshold: float = 0.5,
        complementarity_threshold: float = 0.4,
        max_attempts: int = 100,
        include_wobble: bool = True
    ):
        self.min_dist = min_dist
        self.span_len = span_len
        self.gc_threshold = gc_threshold
        self.complementarity_threshold = complementarity_threshold
        self.max_attempts = max_attempts
        self.include_wobble = include_wobble
    
    def compute_gc_content(
        self, 
        tokens: torch.Tensor, 
        start: int, 
        length: int
    ) -> float:
        """Compute GC content for a region."""
        region = tokens[start:start+length]
        gc_count = ((region == self.G_TOKEN) | (region == self.C_TOKEN)).sum().item()
        return gc_count / length if length > 0 else 0.0
    
    def check_complementarity(
        self, 
        tokens: torch.Tensor, 
        left_start: int, 
        right_start: int,
        span_len: int
    ) -> float:
        """
        Check complementarity between two regions.
        
        The right region is checked in reverse (5'->3' vs 3'->5' pairing).
        
        Returns:
            Fraction of positions that are complementary (0-1)
        """
        left_span = tokens[left_start:left_start+span_len]
        right_span = tokens[right_start:right_start+span_len]
        
        # Reverse the right span for antiparallel pairing
        right_span_rev = right_span.flip(0)
        
        matches = 0
        for l_tok, r_tok in zip(left_span, right_span_rev):
            l_tok = l_tok.item()
            r_tok = r_tok.item()
            
            # Check Watson-Crick complement
            if self.COMPLEMENT_MAP.get(l_tok) == r_tok:
                matches += 1
            # Check wobble (if enabled)
            elif self.include_wobble and self.WOBBLE_MAP.get(l_tok) == r_tok:
                matches += 0.5  # Count wobble as half match
        
        return matches / span_len
    
    def find_gc_rich_regions(
        self, 
        tokens: torch.Tensor, 
        window: int = 20
    ) -> List[int]:
        """Find positions with high local GC content."""
        L = len(tokens)
        gc_rich = []
        
        for i in range(L - window):
            gc = self.compute_gc_content(tokens, i, window)
            if gc >= self.gc_threshold:
                gc_rich.append(i)
        
        return gc_rich
    
    def find_stem_candidates(
        self, 
        tokens: torch.Tensor
    ) -> List[Tuple[int, int]]:
        """
        Find candidate stem pairs in a sequence.
        
        Returns:
            List of (left_start, right_start) tuples
        """
        L = len(tokens)
        
        # Can't find stems in short sequences
        if L < self.min_dist + 2 * self.span_len:
            return []
        
        candidates = []
        
        # Find GC-rich regions
        gc_rich = self.find_gc_rich_regions(tokens)
        
        if len(gc_rich) < 2:
            return []
        
        # Pair distant GC-rich regions
        for i in gc_rich:
            for j in gc_rich:
                # Check distance constraint
                if j - i < self.min_dist:
                    continue
                
                # Check bounds
                if j + self.span_len > L:
                    continue
                
                # Check complementarity
                comp = self.check_complementarity(tokens, i, j, self.span_len)
                
                if comp >= self.complementarity_threshold:
                    candidates.append((i, j))
        
        return candidates
    
    def generate_stem_span_mask(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Generate stem-span masking for a single sequence.
        
        Args:
            tokens: [length] token IDs
            attention_mask: Optional [length] mask (1=valid, 0=padding)
            
        Returns:
            masked_tokens: Tokens with stem spans masked
            labels: Target labels (-100 for non-masked positions)
            success: Whether a valid stem pair was found
        """
        L = len(tokens)
        
        # Initialize labels as -100 (ignore)
        labels = torch.full_like(tokens, -100)
        masked_tokens = tokens.clone()
        
        # Find stem candidates
        candidates = self.find_stem_candidates(tokens)
        
        if not candidates:
            # No valid candidates - fall back to random span masking
            return masked_tokens, labels, False
        
        # Randomly select one pair
        left_start, right_start = random.choice(candidates)
        
        # Mask left span
        for i in range(self.span_len):
            pos = left_start + i
            if pos < L and (attention_mask is None or attention_mask[pos] == 1):
                labels[pos] = tokens[pos]
                masked_tokens[pos] = self.MASK_TOKEN
        
        # Mask right span
        for i in range(self.span_len):
            pos = right_start + i
            if pos < L and (attention_mask is None or attention_mask[pos] == 1):
                labels[pos] = tokens[pos]
                masked_tokens[pos] = self.MASK_TOKEN
        
        return masked_tokens, labels, True


class StemSpanCollatorMixin:
    """
    Mixin class to add stem-span masking to existing collator.
    
    Usage:
        class EnhancedCollator(RNAMLMCollator, StemSpanCollatorMixin):
            def __init__(self, ...):
                super().__init__(...)
                self.init_stem_span_masking(...)
    """
    
    def init_stem_span_masking(
        self,
        use_stem_span: bool = False,
        stem_span_min_dist: int = 100,
        stem_span_length: int = 8,
        stem_span_gc_threshold: float = 0.5,
        stem_span_complementarity: float = 0.4
    ):
        """Initialize stem-span masking components."""
        self.use_stem_span = use_stem_span
        
        if use_stem_span:
            self.stem_masker = StemSpanMasker(
                min_dist=stem_span_min_dist,
                span_len=stem_span_length,
                gc_threshold=stem_span_gc_threshold,
                complementarity_threshold=stem_span_complementarity
            )
            print(f"\n✓ Stem-span masking enabled:")
            print(f"    Min distance: {stem_span_min_dist} nt")
            print(f"    Span length: {stem_span_length} nt")
            print(f"    GC threshold: {stem_span_gc_threshold}")
            print(f"    Complementarity threshold: {stem_span_complementarity}")
    
    def apply_stem_span_masking(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply stem-span masking to a batch.
        
        Args:
            input_ids: [batch, length] token IDs
            attention_mask: [batch, length] attention mask
            
        Returns:
            masked_input_ids: Tokens with stems masked
            labels: Target labels
        """
        batch_size, seq_len = input_ids.shape
        
        masked_ids = input_ids.clone()
        labels = torch.full_like(input_ids, -100)
        
        stem_successes = 0
        
        for b in range(batch_size):
            tokens = input_ids[b]
            mask = attention_mask[b]
            
            m_tokens, m_labels, success = self.stem_masker.generate_stem_span_mask(
                tokens, mask
            )
            
            masked_ids[b] = m_tokens
            labels[b] = m_labels
            
            if success:
                stem_successes += 1
        
        return masked_ids, labels


def create_stem_span_aware_collator(base_collator, config_dict: Dict[str, Any]):
    """
    Factory function to enhance an existing collator with stem-span masking.
    
    Args:
        base_collator: Existing RNAMLMCollator instance
        config_dict: Configuration dictionary
        
    Returns:
        Enhanced collator with stem-span capability
    """
    data_config = config_dict.get('data', {})
    
    # Add stem-span masking capability
    base_collator.use_stem_span = data_config.get('use_stem_span_masking', False)
    
    if base_collator.use_stem_span:
        base_collator.stem_masker = StemSpanMasker(
            min_dist=data_config.get('stem_span_min_dist', 100),
            span_len=data_config.get('stem_span_length', 8),
            gc_threshold=data_config.get('stem_span_gc_threshold', 0.5),
            complementarity_threshold=data_config.get('stem_span_complementarity', 0.4)
        )
        
        # Monkey-patch the apply method
        original_call = base_collator.__call__
        
        def enhanced_call(batch):
            result = original_call(batch)
            
            # If stem-span is enabled and curriculum says to use it
            if base_collator.use_stem_span:
                masked_ids, stem_labels = base_collator.apply_stem_span_masking(
                    result['input_ids'],
                    result['attention_mask']
                )
                result['stem_span_input_ids'] = masked_ids
                result['stem_span_labels'] = stem_labels
            
            return result
        
        base_collator.__call__ = enhanced_call
        base_collator.apply_stem_span_masking = lambda ids, mask: \
            StemSpanCollatorMixin.apply_stem_span_masking(base_collator, ids, mask)
        base_collator.apply_stem_span_masking.__self__ = base_collator
    
    return base_collator


# =============================================================================
# Standalone Testing
# =============================================================================
if __name__ == "__main__":
    # Test stem-span masking
    print("Testing StemSpanMasker...")
    
    masker = StemSpanMasker(
        min_dist=20,  # Shorter for testing
        span_len=4,
        gc_threshold=0.3
    )
    
    # Create a test sequence with a potential stem
    # This mimics a hairpin: 5'-GGCGAU...AUCGCC-3' (complementary ends)
    # Using token IDs: A=7, U=8, G=9, C=10
    test_tokens = torch.tensor([
        1,  # START
        9, 9, 10, 9, 7, 8,  # GGCGAU (GC-rich)
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7,  # AAAAAAAAAA (loop)
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7,  # more loop
        8, 7, 10, 9, 10, 10,  # UACGCC (complementary, reversed)
        2   # END
    ])
    
    print(f"Test sequence length: {len(test_tokens)}")
    
    # Find candidates
    candidates = masker.find_stem_candidates(test_tokens)
    print(f"Found {len(candidates)} stem candidates")
    
    if candidates:
        # Generate mask
        masked, labels, success = masker.generate_stem_span_mask(test_tokens)
        print(f"Masking success: {success}")
        print(f"Masked positions: {(masked == masker.MASK_TOKEN).nonzero().squeeze().tolist()}")
        print(f"Label positions: {(labels != -100).nonzero().squeeze().tolist()}")
    else:
        print("No candidates found - sequence may be too short or lack complementary regions")
    
    print("\n✓ StemSpanMasker test complete")