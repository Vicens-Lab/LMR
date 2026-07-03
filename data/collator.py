# File: data/collator_enhanced.py
# ============================================================================
# ENHANCED RNA MLM COLLATOR - ADVANCED DATA AUGMENTATION
# ============================================================================
# New features vs. original collator.py:
#   1. Span masking: Mask contiguous 2-5 nucleotide spans (biologically meaningful)
#   2. Reverse complement: Randomly reverse complement 50% of sequences
#   3. Nucleotide substitution noise: 1-2% random substitutions (simulates errors)
#   4. Dynamic masking rate: Vary mask probability 15-30% per batch
# 
# All features are configurable via config file flags.
# Falls back to standard masking if augmentation disabled.
# ============================================================================

import torch
import random
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class RNAMLMCollatorConfig:
    """Enhanced configuration for RNA MLM collator with augmentation"""
    # Standard MLM parameters
    mask_probability: float = 0.15
    mask_replace_prob: float = 0.8
    pad_token_id: int = 0
    mask_token_id: int = 3
    vocab_size: int = 11
    max_seq_len: int = 2048
    
    # Advanced augmentation flags (NEW)
    use_span_masking: bool = False
    span_length_min: int = 2
    span_length_max: int = 5
    
    use_reverse_complement: bool = False
    
    use_nucleotide_noise: bool = False
    noise_probability: float = 0.02
    
    dynamic_masking: bool = False
    dynamic_mask_min: float = 0.15
    dynamic_mask_max: float = 0.30


class RNAMLMCollator:
    """
    Enhanced MLM collator with advanced biological augmentation.
    
    Augmentation strategies:
    1. Span Masking: Masks contiguous spans of 2-5 nucleotides
       - Biologically meaningful (k-mer dependencies)
       - Harder task → better generalization
    
    2. Reverse Complement: Randomly flips sequence direction
       - Exploits biological symmetry of RNA
       - Doubles effective dataset size
    
    3. Nucleotide Noise: Random substitutions (1-2%)
       - Simulates sequencing errors
       - Forces robustness to mutations
    
    4. Dynamic Masking: Varies mask rate per batch (15-30%)
       - Prevents overfitting to fixed rate
       - Encourages diverse learning
    """
    
    # Nucleotide complement mapping for reverse complement
    COMPLEMENT_MAP = {
        'A': 'U',
        'U': 'A',
        'G': 'C',
        'C': 'G',
        '<pad>': '<pad>',
        '<start>': '<start>',
        '<end>': '<end>',
        '<mask>': '<mask>',
        '<unk>': '<unk>',
        '<msa_sep>': '<msa_sep>',
        '<gap>': '<gap>'
    }
    
    # Token ID complement mapping (for reverse complement)
    COMPLEMENT_ID_MAP = {
        7: 8,   # A → U
        8: 7,   # U → A
        9: 10,  # G → C
        10: 9,  # C → G
        # Special tokens map to themselves
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6
    }
    
    def __init__(self, tokenizer, config: RNAMLMCollatorConfig):
        self.tokenizer = tokenizer
        self.config = config
        
        self.mask_prob = config.mask_probability
        self.mask_replace_prob = config.mask_replace_prob
        self.pad_token_id = config.pad_token_id
        self.mask_token_id = config.mask_token_id
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        
        # Augmentation settings
        self.use_span_masking = config.use_span_masking
        self.span_length_min = config.span_length_min
        self.span_length_max = config.span_length_max
        
        self.use_reverse_complement = config.use_reverse_complement
        
        self.use_nucleotide_noise = config.use_nucleotide_noise
        self.noise_probability = config.noise_probability
        
        self.dynamic_masking = config.dynamic_masking
        self.dynamic_mask_min = config.dynamic_mask_min
        self.dynamic_mask_max = config.dynamic_mask_max
        
        # Nucleotide token IDs (for random replacement)
        self.nucleotide_ids = torch.tensor([7, 8, 9, 10], dtype=torch.long)
        
        print("\n" + "="*70)
        print("Enhanced RNAMLMCollator")
        print("="*70)
        print(f"Tokenizer: SimpleRNATokenizer (vocab_size={self.vocab_size})")
        print(f"Base mask probability: {self.mask_prob}")
        print(f"Max sequence length: {self.max_seq_len}")
        print(f"\nAugmentation Features:")
        print(f"  Span masking: {'ENABLED' if self.use_span_masking else 'disabled'}")
        if self.use_span_masking:
            print(f"    Span length: {self.span_length_min}-{self.span_length_max} nucleotides")
        print(f"  Reverse complement: {'ENABLED' if self.use_reverse_complement else 'disabled'}")
        print(f"  Nucleotide noise: {'ENABLED' if self.use_nucleotide_noise else 'disabled'}")
        if self.use_nucleotide_noise:
            print(f"    Noise rate: {self.noise_probability:.1%}")
        print(f"  Dynamic masking: {'ENABLED' if self.dynamic_masking else 'disabled'}")
        if self.dynamic_masking:
            print(f"    Mask rate range: {self.dynamic_mask_min:.1%}-{self.dynamic_mask_max:.1%}")
        print("="*70 + "\n")
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch with optional augmentation.
        
        Augmentation pipeline:
        1. Reverse complement (if enabled) - 50% probability
        2. Encode to token IDs
        3. Add nucleotide noise (if enabled) - before masking
        4. Pad sequences
        5. Apply MLM masking (span or random)
        
        Args:
            batch: List of dicts with 'sequence' key
        
        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        sequences = [item['sequence'] for item in batch]
        
        # Dynamic masking: vary mask probability per batch
        if self.dynamic_masking:
            current_mask_prob = random.uniform(self.dynamic_mask_min, self.dynamic_mask_max)
        else:
            current_mask_prob = self.mask_prob
        
        # Augmentation: Reverse complement (before encoding)
        if self.use_reverse_complement:
            sequences = [self._maybe_reverse_complement(seq) for seq in sequences]
        
        # Encode sequences
        encoded_batch = []
        for seq in sequences:
            ids = self.tokenizer.encode_sequence(seq, add_special=True)
            if len(ids) > self.max_seq_len:
                ids = ids[:self.max_seq_len]
            encoded_batch.append(ids)
        
        # Find max length in batch
        max_len = max(len(ids) for ids in encoded_batch)
        
        # Pad sequences and create tensors
        batch_size = len(encoded_batch)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        
        for i, ids in enumerate(encoded_batch):
            seq_len = len(ids)
            input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
        
        # Augmentation: Add nucleotide noise (before masking)
        if self.use_nucleotide_noise:
            input_ids = self._add_nucleotide_noise(input_ids, attention_mask)
        
        # Apply MLM masking (span or random)
        if self.use_span_masking:
            input_ids, labels = self._apply_span_masking(
                input_ids, attention_mask, current_mask_prob
            )
        else:
            input_ids, labels = self._apply_mlm_masking_vectorized(
                input_ids, attention_mask, current_mask_prob
            )
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
    
    def _maybe_reverse_complement(self, sequence: str) -> str:
        """
        Randomly reverse complement sequence with 50% probability.
        
        RNA reverse complement:
        - Reverse the sequence order
        - Replace each nucleotide with its complement (A↔U, G↔C)
        
        Example:
          Original:  AUGC → [A, U, G, C]
          Reversed:  CGUA → [C, G, U, A]
          Complement: GCAU → [G, C, A, U]
        
        Args:
            sequence: RNA sequence string
        
        Returns:
            Original or reverse complemented sequence
        """
        if random.random() < 0.5:
            # Reverse complement
            reversed_seq = sequence[::-1]  # Reverse
            complement = ''.join(self.COMPLEMENT_MAP.get(nt, nt) for nt in reversed_seq)
            return complement
        return sequence
    
    def _add_nucleotide_noise(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Add random nucleotide substitutions to simulate sequencing errors.
        
        Only affects actual nucleotides (IDs 7-10), not special tokens.
        Probability is low (default 2%) to avoid corrupting too much data.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            input_ids with noise added
        """
        # Clone to avoid modifying original
        noisy_ids = input_ids.clone()
        
        # Identify nucleotide positions (IDs 7-10)
        is_nucleotide = (input_ids >= 7) & (input_ids <= 10)
        is_valid = (attention_mask == 1)
        can_noise = is_nucleotide & is_valid
        
        # Randomly select positions to add noise
        noise_mask = torch.rand_like(input_ids, dtype=torch.float32) < self.noise_probability
        noise_mask = noise_mask & can_noise
        
        # Replace selected positions with random nucleotides
        n_noise = noise_mask.sum().item()
        if n_noise > 0:
            random_nucs = self.nucleotide_ids[torch.randint(0, 4, (n_noise,))]
            noisy_ids[noise_mask] = random_nucs
        
        return noisy_ids
    
    def _apply_span_masking(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_probability: float
    ) -> tuple:
        """
        Apply span masking: mask contiguous spans of 2-5 nucleotides.
        
        This is more biologically meaningful than random token masking:
        - RNA k-mers (2-5 nucleotide motifs) are functional units
        - Forces model to learn dependencies between adjacent nucleotides
        - Harder task → better generalization
        
        Algorithm:
        1. Identify maskable positions (nucleotides only)
        2. Randomly select span start positions
        3. Mask spans of random length (2-5 nucleotides)
        4. Continue until target mask probability reached
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            mask_probability: Target fraction of tokens to mask
        
        Returns:
            Masked input_ids and labels
        """
        batch_size, seq_len = input_ids.shape
        
        # Create labels (copy before masking)
        labels = input_ids.clone()
        
        # Special tokens that should never be masked
        special_mask = input_ids < 7  # True for special tokens
        can_mask = (attention_mask == 1) & (~special_mask)
        
        # Calculate target number of tokens to mask per sequence
        target_masked = int(seq_len * mask_probability)
        
        # Process each sequence in batch
        for b in range(batch_size):
            maskable_positions = torch.where(can_mask[b])[0]
            
            if len(maskable_positions) == 0:
                # No positions to mask
                labels[b] = -100
                continue
            
            # Shuffle maskable positions
            perm = torch.randperm(len(maskable_positions))
            shuffled_positions = maskable_positions[perm]
            
            masked_count = 0
            span_starts = []
            
            # Generate spans until target reached
            i = 0
            while masked_count < target_masked and i < len(shuffled_positions):
                start_pos = shuffled_positions[i].item()
                
                # Random span length (2-5 nucleotides)
                span_len = random.randint(self.span_length_min, self.span_length_max)
                
                # Ensure span doesn't exceed sequence or target
                remaining = min(
                    seq_len - start_pos,
                    target_masked - masked_count
                )
                span_len = min(span_len, remaining)
                
                # Mark span for masking
                for offset in range(span_len):
                    pos = start_pos + offset
                    if pos < seq_len and can_mask[b, pos]:
                        span_starts.append(pos)
                        masked_count += 1
                
                i += 1
            
            # Convert to mask tensor
            if span_starts:
                mask_indices = torch.tensor(span_starts, dtype=torch.long)
                
                # Set labels: -100 for unmasked positions
                seq_labels = labels[b].clone()
                seq_labels[:] = -100
                seq_labels[mask_indices] = input_ids[b, mask_indices]
                labels[b] = seq_labels
                
                # Apply masking strategy (80% mask, 10% random, 10% keep)
                rand_action = torch.rand(len(mask_indices))
                
                # 80%: Replace with <mask>
                mask_with_mask = mask_indices[rand_action < self.mask_replace_prob]
                input_ids[b, mask_with_mask] = self.mask_token_id
                
                # 10%: Replace with random nucleotide
                mask_with_random = mask_indices[
                    (rand_action >= self.mask_replace_prob) & 
                    (rand_action < self.mask_replace_prob + 0.1)
                ]
                if len(mask_with_random) > 0:
                    random_nucs = self.nucleotide_ids[torch.randint(0, 4, (len(mask_with_random),))]
                    input_ids[b, mask_with_random] = random_nucs
                
                # 10%: Keep original (do nothing)
            else:
                # No spans created - mark all as unmasked
                labels[b] = -100
        
        return input_ids, labels
    
    def _apply_mlm_masking_vectorized(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_probability: float
    ) -> tuple:
        """
        Apply standard MLM masking (vectorized, fast).
        
        This is the fallback when span masking is disabled.
        Masks random individual tokens instead of spans.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            mask_probability: Fraction of tokens to mask
        
        Returns:
            Masked input_ids and labels
        """
        # Create labels (copy before masking)
        labels = input_ids.clone()
        
        # Special tokens that should never be masked
        special_mask = input_ids < 7  # True for special tokens
        can_mask = (attention_mask == 1) & (~special_mask)
        
        # Randomly select positions to mask
        rand_mask = torch.rand_like(input_ids, dtype=torch.float32) < mask_probability
        mask_indices = rand_mask & can_mask
        
        # Set labels: -100 for positions we're NOT predicting
        labels[~mask_indices] = -100
        
        # For masked positions, determine action:
        # 80% → <mask> token
        # 10% → random nucleotide
        # 10% → keep original
        
        rand_action = torch.rand_like(input_ids, dtype=torch.float32)
        
        # 80%: Replace with <mask>
        mask_with_mask_token = mask_indices & (rand_action < self.mask_replace_prob)
        input_ids[mask_with_mask_token] = self.mask_token_id
        
        # 10%: Replace with random nucleotide
        mask_with_random = mask_indices & \
                          (rand_action >= self.mask_replace_prob) & \
                          (rand_action < self.mask_replace_prob + 0.1)
        n_random = mask_with_random.sum().item()
        if n_random > 0:
            random_nucs = self.nucleotide_ids[torch.randint(0, 4, (n_random,))]
            input_ids[mask_with_random] = random_nucs
        
        # 10%: Keep original (do nothing)
        
        return input_ids, labels


# ============================================================================
# Factory function - for train.py compatibility
# ============================================================================
def get_collator_enhanced(tokenizer, config_dict: dict) -> RNAMLMCollator:
    """
    Factory function to create enhanced collator from config dictionary.
    
    Reads augmentation flags from config file.
    """
    data_config = config_dict.get('data', {})
    model_config = config_dict.get('model', {})
    
    collator_config = RNAMLMCollatorConfig(
        # Standard MLM parameters
        mask_probability=data_config.get('mask_probability', 0.15),
        mask_replace_prob=data_config.get('mask_replace_prob', 0.8),
        pad_token_id=model_config.get('pad_token_id', 0),
        mask_token_id=model_config.get('mask_token_id', 3),
        vocab_size=model_config.get('vocab_size', 11),
        max_seq_len=data_config.get('max_seq_len', 2048),
        
        # Augmentation parameters (NEW)
        use_span_masking=data_config.get('use_span_masking', False),
        span_length_min=data_config.get('span_length_min', 2),
        span_length_max=data_config.get('span_length_max', 5),
        
        use_reverse_complement=data_config.get('use_reverse_complement', False),
        
        use_nucleotide_noise=data_config.get('use_nucleotide_noise', False),
        noise_probability=data_config.get('noise_probability', 0.02),
        
        dynamic_masking=data_config.get('dynamic_masking', False),
        dynamic_mask_min=data_config.get('dynamic_mask_min', 0.15),
        dynamic_mask_max=data_config.get('dynamic_mask_max', 0.30),
    )
    
    return RNAMLMCollator(tokenizer, collator_config)


# ============================================================================
# Backward compatibility - original collator still available
# ============================================================================
def get_collator(tokenizer, config_dict: dict) -> RNAMLMCollator:
    """Original collator (no augmentation) - for backward compatibility"""
    data_config = config_dict.get('data', {})
    model_config = config_dict.get('model', {})
    
    collator_config = RNAMLMCollatorConfig(
        mask_probability=data_config.get('mask_probability', 0.15),
        mask_replace_prob=data_config.get('mask_replace_prob', 0.8),
        pad_token_id=model_config.get('pad_token_id', 0),
        mask_token_id=model_config.get('mask_token_id', 3),
        vocab_size=model_config.get('vocab_size', 11),
        max_seq_len=data_config.get('max_seq_len', 2048),
        # All augmentation flags default to False
    )
    
    return RNAMLMCollator(tokenizer, collator_config)