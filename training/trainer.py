# File: training/trainer.py
"""
LMR v0 Trainer with Proper Epoch Continuation

Key fixes from previous version:
1. Epoch boundaries now based on global_step, not per-epoch counter
2. Checkpoints save/restore steps_in_epoch for mid-epoch resumption
3. When resuming mid-epoch, remaining steps are calculated correctly
4. Epoch transitions properly increment epoch counter

This ensures that when you resume training, it continues exactly where
it left off rather than re-running the entire epoch.
"""

import os
import gc
import json
import time
import math
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from training.metrics_logger import MetricsLogger


@dataclass
class EarlyStopping:
    """Early stopping handler"""
    patience: int = 10
    min_delta: float = 0.0001
    mode: str = 'min'
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.should_stop = False
    
    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class Trainer:
    """
    Trainer for LMR with proper checkpoint resumption.
    
    Key features:
    - Epoch boundaries based on global_step
    - Mid-epoch checkpoint resumption
    - Distributed training support (DDP)
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Metrics logging to CSV
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        config: Dict[str, Any],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)
        
        # =====================================================================
        # Training state - these are saved/restored from checkpoints
        # =====================================================================
        self.epoch = 0
        self.global_step = 0
        self.steps_in_epoch = 0  # NEW: Track progress within current epoch
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.train_start_time = None
        
        # Loss function
        label_smoothing = config['training'].get('label_smoothing', 0.0)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=label_smoothing
        )
        
        if self.is_main_process:
            print(f"\n{'='*70}")
            print("Loss Function Configuration")
            print(f"{'='*70}")
            print(f"  Label smoothing: {label_smoothing}")
            print(f"  Ignore index: -100")
            print(f"{'='*70}\n")
        
        # Mixed precision
        self.use_amp = config['training'].get('use_amp', True) and self.device.type == 'cuda'
        amp_dtype_str = config['training'].get('amp_dtype', 'bfloat16')
        self.amp_dtype = torch.bfloat16 if amp_dtype_str == 'bfloat16' else torch.float16

        use_scaler = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = GradScaler(enabled=use_scaler)
        
        # Training parameters
        self.num_training_steps = config['training'].get('num_training_steps', 100000)
        self.steps_per_epoch = config['training'].get('steps_per_epoch', 1000)
        
        # Early stopping
        early_stop_config = config.get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=early_stop_config.get('patience', 10),
            min_delta=early_stop_config.get('min_delta', 0.0001),
            mode='min'
        ) if early_stop_config.get('enabled', False) else None
        
        # Gradient clipping
        self.max_grad_norm = config['training'].get('max_grad_norm', 1.0)
        
        # Logging intervals
        self.logging_steps = config['logging'].get('log_steps', 50)
        self.eval_steps = config['checkpointing'].get('eval_steps', 1000)
        self.save_steps = config['checkpointing'].get('save_steps', 2500)
        
        # Gradient accumulation
        self.grad_accum_steps = config['training'].get('gradient_accumulation_steps', 1)
        
        # Metrics logger
        checkpoint_dir = config['checkpointing']['output_dir']
        self.metrics_logger = MetricsLogger(checkpoint_dir, rank=rank)
        
        # Running metrics for logging
        self.running_train_loss = 0.0
        self.running_train_correct = 0
        self.running_train_total = 0
        
        # WandB
        self.use_wandb = config['logging'].get('use_wandb', False) and self.is_main_process
        if self.use_wandb:
            import wandb
            wandb.init(
                project=config['logging'].get('wandb_project', 'lmr'),
                config=config,
                name=f"run_{time.strftime('%Y%m%d_%H%M%S')}"
            )
    
    def train(self):
        """Main training loop with proper epoch handling."""
        num_epochs = self.config['training']['num_epochs']
        
        self.train_start_time = time.time()
        
        if self.is_main_process:
            print("\n" + "="*70)
            print("Starting Training")
            print("="*70)
            print(f"Epochs: {num_epochs}")
            print(f"Steps per epoch: {self.steps_per_epoch}")
            print(f"Gradient accumulation: {self.grad_accum_steps}")
            print(f"Mixed precision: {self.use_amp} ({self.amp_dtype})")
            print(f"Effective batch size: {self.config['training']['batch_size_per_gpu'] * self.grad_accum_steps * self.world_size}")
            
            # Show resume info
            if self.global_step > 0:
                print(f"\n*** RESUMED from step {self.global_step}, epoch {self.epoch+1} ***")
                print(f"    Steps completed in epoch {self.epoch+1}: {self.steps_in_epoch}")
                print(f"    Steps remaining in epoch {self.epoch+1}: {self.steps_per_epoch - self.steps_in_epoch}")
            
            print("="*70 + "\n")
        
        # Resume from self.epoch (loaded from checkpoint in load_checkpoint())
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            if self.is_main_process:
                print(f"\n{'='*70}")
                print(f"Epoch {epoch+1}/{num_epochs}")
                if self.steps_in_epoch > 0:
                    print(f"  (Resuming from step {self.steps_in_epoch}/{self.steps_per_epoch})")
                print(f"{'='*70}")
            
            should_stop = self.train_epoch(epoch)
            
            if should_stop:
                if self.is_main_process:
                    print("\n" + "="*70)
                    print("Early stopping triggered")
                    print("="*70)
                break
            
            # Reset steps_in_epoch for next epoch
            self.steps_in_epoch = 0
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Final summary
        if self.is_main_process:
            total_time = time.time() - self.train_start_time
            
            print("\n" + "="*70)
            print("Training Complete!")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print(f"Best validation accuracy: {self.best_val_acc:.2%}")
            print(f"Total time: {total_time/3600:.2f} hours")
            print("="*70)
            
            self.metrics_logger.save_summary(
                best_val_loss=self.best_val_loss,
                best_val_acc=self.best_val_acc,
                total_steps=self.global_step,
                total_time=total_time,
                config=self.config
            )
            
            if self.use_wandb:
                import wandb
                wandb.finish()
    
    def train_epoch(self, epoch: int) -> bool:
        """
        Train one epoch with proper resumption handling.
        
        Key fix: Use steps_in_epoch to track progress and skip already-completed
        steps when resuming mid-epoch.
        """
        self.model.train()
        
        # Calculate how many steps remain in this epoch
        steps_remaining = self.steps_per_epoch - self.steps_in_epoch
        
        if self.is_main_process:
            pbar = tqdm(
                enumerate(self.train_dataloader),
                total=self.steps_per_epoch,
                initial=self.steps_in_epoch,  # Start progress bar from current position
                desc=f"Epoch {epoch+1}"
            )
        else:
            pbar = enumerate(self.train_dataloader)
        
        epoch_loss = 0.0
        num_batches = 0
        step_start_time = time.time()
        
        # Reset running metrics
        self.running_train_loss = 0.0
        self.running_train_correct = 0
        self.running_train_total = 0
        
        # Counter for skipping already-processed batches when resuming
        batches_to_skip = self.steps_in_epoch * self.grad_accum_steps
        skipped = 0
        
        for step, batch in pbar:
            # ================================================================
            # Skip batches if resuming mid-epoch
            # ================================================================
            if skipped < batches_to_skip:
                skipped += 1
                continue
            
            # ================================================================
            # Check if epoch is complete
            # ================================================================
            if num_batches >= steps_remaining:
                break
            
            is_accumulating = (step + 1) % self.grad_accum_steps != 0
            
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Forward pass
            with autocast(device_type='cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                loss = loss / self.grad_accum_steps
            
            # Track training accuracy
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                mask = labels != -100
                correct = ((predictions == labels) & mask).sum().item()
                total = mask.sum().item()
                self.running_train_correct += correct
                self.running_train_total += total
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Optimizer step (only when not accumulating)
            if not is_accumulating:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Update counters
                self.global_step += 1
                self.steps_in_epoch += 1
                epoch_loss += loss.item() * self.grad_accum_steps
                self.running_train_loss += loss.item() * self.grad_accum_steps
                num_batches += 1
                
                # Throughput calculation
                step_time = time.time() - step_start_time
                throughput = 1.0 / step_time if step_time > 0 else 0
                step_start_time = time.time()
                
                # Update progress bar
                if self.is_main_process:
                    train_acc = self.running_train_correct / max(1, self.running_train_total)
                    pbar.set_postfix({
                        'loss': f"{loss.item() * self.grad_accum_steps:.4f}",
                        'train_acc': f"{train_acc:.2%}",
                        'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                        'it/s': f"{throughput:.2f}"
                    })
                
                # ============================================================
                # Logging
                # ============================================================
                if self.global_step % self.logging_steps == 0:
                    avg_loss = self.running_train_loss / max(1, num_batches)
                    train_acc = self.running_train_correct / max(1, self.running_train_total)
                    current_lr = self.scheduler.get_last_lr()[0]
                    
                    if self.is_main_process:
                        print(f"\nStep {self.global_step} (Epoch {epoch+1}, {self.steps_in_epoch}/{self.steps_per_epoch})")
                        print(f"  Train Loss: {avg_loss:.4f}")
                        print(f"  Train Acc:  {train_acc:.2%}")
                        print(f"  LR: {current_lr:.2e}")
                        print(f"  Throughput: {throughput:.2f} it/s")
                        
                        self.metrics_logger.log_train_step(
                            step=self.global_step,
                            epoch=epoch,
                            loss=avg_loss,
                            learning_rate=current_lr,
                            throughput=throughput,
                            log_memory=(self.global_step % (self.logging_steps * 10) == 0)
                        )
                        
                        if self.use_wandb:
                            import wandb
                            wandb.log({
                                'train/loss': avg_loss,
                                'train/accuracy': train_acc,
                                'train/lr': current_lr,
                                'train/step': self.global_step,
                                'train/epoch': epoch,
                                'train/throughput': throughput
                            })
                
                # ============================================================
                # Evaluation
                # ============================================================
                if self.global_step % self.eval_steps == 0:
                    val_metrics = self.evaluate_detailed()
                    
                    if self.is_main_process:
                        train_acc = self.running_train_correct / max(1, self.running_train_total)
                        
                        print(f"\n{'='*70}")
                        print(f"Evaluation at step {self.global_step} (Epoch {epoch+1})")
                        print(f"{'='*70}")
                        print(f"  Train Acc:  {train_acc:.2%}")
                        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
                        print(f"  Val Acc:    {val_metrics['accuracy']:.2%}")
                        print(f"  Val PPL:    {val_metrics['perplexity']:.2f}")
                        print(f"  Train-Val Gap: {train_acc / max(0.01, val_metrics['accuracy']):.1f}x")
                        
                        # Per-class accuracy
                        if 'per_class_acc' in val_metrics:
                            print(f"\n  Per-Nucleotide Accuracy:")
                            for token, acc in val_metrics['per_class_acc'].items():
                                print(f"    {token}: {acc:.2%}")
                        
                        print(f"{'='*70}\n")
                        
                        self.metrics_logger.log_validation(
                            step=self.global_step,
                            epoch=epoch,
                            val_loss=val_metrics['loss'],
                            val_accuracy=val_metrics['accuracy']
                        )
                        
                        if self.use_wandb:
                            import wandb
                            wandb.log({
                                'val/loss': val_metrics['loss'],
                                'val/accuracy': val_metrics['accuracy'],
                                'val/perplexity': val_metrics['perplexity'],
                                'val/step': self.global_step
                            })
                        
                        # Check for best model (by loss)
                        if val_metrics['loss'] < self.best_val_loss:
                            self.best_val_loss = val_metrics['loss']
                            best_path = os.path.join(
                                self.config['checkpointing']['output_dir'],
                                "checkpoint_best.pt"
                            )
                            self._save_checkpoint(best_path, is_best=True)
                            print(f"✓ New best model (loss: {val_metrics['loss']:.4f})")
                        
                        # Check for best model (by accuracy)
                        if val_metrics['accuracy'] > self.best_val_acc:
                            self.best_val_acc = val_metrics['accuracy']
                            best_acc_path = os.path.join(
                                self.config['checkpointing']['output_dir'],
                                "checkpoint_best_acc.pt"
                            )
                            self._save_checkpoint(best_acc_path, is_best=True)
                            print(f"✓ New best accuracy ({val_metrics['accuracy']:.2%})")
                        
                        # Early stopping check
                        if self.early_stopping:
                            if self.early_stopping(val_metrics['loss'], epoch):
                                return True
                    
                    self.model.train()
                
                # ============================================================
                # Checkpointing
                # ============================================================
                if self.global_step % self.save_steps == 0:
                    if self.is_main_process:
                        checkpoint_path = os.path.join(
                            self.config['checkpointing']['output_dir'],
                            f"checkpoint_step_{self.global_step}.pt"
                        )
                        self._save_checkpoint(checkpoint_path, is_best=False)
                        print(f"✓ Checkpoint saved at step {self.global_step}")
        
        # Epoch completed successfully
        if self.is_main_process:
            print(f"\n✓ Epoch {epoch+1} complete (steps: {self.steps_in_epoch}/{self.steps_per_epoch})")
        
        return False
    
    def evaluate_detailed(self) -> Dict[str, Any]:
        """Detailed evaluation with per-class metrics"""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0
        
        # Per-class tracking (for nucleotide tokens 7-10)
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        # Token ID to name mapping (for simple tokenizer)
        token_names = {7: 'A', 8: 'U', 9: 'G', 10: 'C'}
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                with autocast(device_type='cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                    logits = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )
                
                total_loss += loss.item()
                
                # Overall accuracy
                predictions = torch.argmax(logits, dim=-1)
                mask = labels != -100
                correct = (predictions == labels) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
                
                # Per-class accuracy
                for token_id in token_names.keys():
                    token_mask = (labels == token_id)
                    class_total[token_id] += token_mask.sum().item()
                    class_correct[token_id] += ((predictions == labels) & token_mask).sum().item()
                
                num_batches += 1
                if num_batches >= 200:
                    break
        
        avg_loss = total_loss / max(1, num_batches)
        accuracy = total_correct / max(1, total_tokens)
        perplexity = math.exp(min(avg_loss, 10))  # Cap to prevent overflow
        
        # Per-class accuracy
        per_class_acc = {}
        for token_id, name in token_names.items():
            if class_total[token_id] > 0:
                per_class_acc[name] = class_correct[token_id] / class_total[token_id]
            else:
                per_class_acc[name] = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'perplexity': perplexity,
            'per_class_acc': per_class_acc,
            'total_tokens': total_tokens
        }
    
    def evaluate(self) -> Tuple[float, float]:
        """Simple evaluation (for compatibility)"""
        metrics = self.evaluate_detailed()
        return metrics['loss'], metrics['accuracy']
    
    def _save_checkpoint(self, path: str, is_best: bool = False):
        """
        Save checkpoint with all state needed for resumption.
        
        Key addition: saves steps_in_epoch for mid-epoch resumption.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            # Training state
            'epoch': self.epoch,
            'global_step': self.global_step,
            'steps_in_epoch': self.steps_in_epoch,  # NEW: For mid-epoch resumption
            
            # Model and optimizer
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            
            # Best metrics
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            
            # Config for reference
            'config': self.config
        }
        
        if self.early_stopping:
            checkpoint['early_stopping_state'] = {
                'counter': self.early_stopping.counter,
                'best_score': self.early_stopping.best_score,
                'best_epoch': self.early_stopping.best_epoch
            }
        
        torch.save(checkpoint, path)
        
        if self.is_main_process:
            print(f"  [Checkpoint: epoch={self.epoch+1}, step={self.global_step}, steps_in_epoch={self.steps_in_epoch}]")
    
    def load_checkpoint(self, path: str):
        """
        Load checkpoint and restore all training state.
        
        Key addition: restores steps_in_epoch for mid-epoch resumption.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Restore model
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore training state
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.steps_in_epoch = checkpoint.get('steps_in_epoch', 0)  # NEW: Mid-epoch state
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        # Restore early stopping state
        if self.early_stopping and 'early_stopping_state' in checkpoint:
            es_state = checkpoint['early_stopping_state']
            self.early_stopping.counter = es_state['counter']
            self.early_stopping.best_score = es_state['best_score']
            self.early_stopping.best_epoch = es_state['best_epoch']
        
        if self.is_main_process:
            print(f"✓ Checkpoint loaded:")
            print(f"    Global step: {self.global_step}")
            print(f"    Epoch: {self.epoch + 1}")
            print(f"    Steps in epoch: {self.steps_in_epoch}/{self.steps_per_epoch}")
            print(f"    Best val loss: {self.best_val_loss:.4f}")
            print(f"    Best val acc: {self.best_val_acc:.2%}")