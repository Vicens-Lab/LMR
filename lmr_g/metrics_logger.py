# File: training/metrics_logger.py
"""
Comprehensive metrics logging for LMR training.

Tracks and saves:
- Training loss (per step)
- Validation loss and accuracy (per eval)
- Learning rate (per step)
- GPU memory usage (per step)
- Throughput (iterations/second)
- Curriculum metrics (gamma, phase)

Saves to CSV for easy analysis and plotting.
"""

import os
import csv
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import torch


@dataclass
class TrainingMetrics:
    # - Metrics for a single training step
    step: int
    epoch: int
    loss: float
    learning_rate: float
    throughput: Optional[float] = None  # iterations/second
    gpu_memory_mb: Optional[float] = None
    gamma: Optional[float] = None
    phase: Optional[str] = None
    timestamp: Optional[float] = None


@dataclass
class ValidationMetrics:
    # - Metrics for a validation run
    step: int
    epoch: int
    val_loss: float
    val_accuracy: float
    timestamp: Optional[float] = None


class MetricsLogger:
    """
    Logger for training and validation metrics.
    
    Saves metrics to CSV files for later analysis and plotting.
    """
    def __init__(self, output_dir: str, rank: int = 0):
        """
        Initialize metrics logger.
        
        Args:
            output_dir: Directory to save metrics
            rank: Process rank (only rank 0 logs to files)
        """
        self.output_dir = output_dir
        self.rank = rank
        self.is_main_process = (rank == 0)
        
        # Metrics buffers (in memory)
        self.train_metrics: List[TrainingMetrics] = []
        self.val_metrics: List[ValidationMetrics] = []
        
        # File paths
        self.train_csv = os.path.join(output_dir, "train_metrics.csv")
        self.val_csv = os.path.join(output_dir, "val_metrics.csv")
        self.summary_txt = os.path.join(output_dir, "training_summary.txt")
        
        # Create output directory
        if self.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            
            # Initialize CSV files with headers
            self._init_train_csv()
            self._init_val_csv()
            
            print(f"\n{'='*70}")
            print("MetricsLogger initialized")
            print(f"{'='*70}")
            print(f"Output directory: {output_dir}")
            print(f"Training metrics: {self.train_csv}")
            print(f"Validation metrics: {self.val_csv}")
            print(f"Summary: {self.summary_txt}")
            print(f"{'='*70}\n")
    
    def _init_train_csv(self):
        # - Initialize training metrics CSV with headers
        with open(self.train_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'epoch', 'loss', 'learning_rate', 
                'throughput_it_per_s', 'gpu_memory_mb', 'gamma', 'phase', 'timestamp'
            ])
    
    def _init_val_csv(self):
        # - Initialize validation metrics CSV with headers
        with open(self.val_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'epoch', 'val_loss', 'val_accuracy', 'timestamp'
            ])
    
    def log_train_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        learning_rate: float,
        throughput: Optional[float] = None,
        log_memory: bool = True,
        gamma: Optional[float] = None,
        phase: Optional[str] = None
    ):
        """
        Log metrics for a training step.
        
        Args:
            step: Global step number
            epoch: Current epoch
            loss: Training loss
            learning_rate: Current learning rate
            throughput: Iterations per second
            log_memory: Whether to log GPU memory usage
            gamma: Current curriculum gamma value (optional)
            phase: Current curriculum phase name (optional)
        """
        if not self.is_main_process:
            return
        
        # Get GPU memory if requested
        gpu_memory_mb = None
        if log_memory and torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # Create metrics object
        metrics = TrainingMetrics(
            step=step,
            epoch=epoch,
            loss=loss,
            learning_rate=learning_rate,
            throughput=throughput,
            gpu_memory_mb=gpu_memory_mb,
            gamma=gamma,
            phase=phase,
            timestamp=time.time()
        )
        
        # Store in buffer
        self.train_metrics.append(metrics)
        
        # Write to CSV
        with open(self.train_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.step,
                metrics.epoch,
                f"{metrics.loss:.6f}",
                f"{metrics.learning_rate:.8f}",
                f"{metrics.throughput:.4f}" if metrics.throughput else "",
                f"{metrics.gpu_memory_mb:.2f}" if metrics.gpu_memory_mb else "",
                f"{metrics.gamma:.6f}" if metrics.gamma is not None else "",
                f"{metrics.phase}" if metrics.phase else "",
                f"{metrics.timestamp:.0f}" if metrics.timestamp else ""
            ])
    
    def log_validation(
        self,
        step: int,
        epoch: int,
        val_loss: float,
        val_accuracy: float
    ):
        """
        Log validation metrics.
        
        Args:
            step: Global step number
            epoch: Current epoch
            val_loss: Validation loss
            val_accuracy: Validation accuracy
        """
        if not self.is_main_process:
            return
        
        # Create metrics object
        metrics = ValidationMetrics(
            step=step,
            epoch=epoch,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            timestamp=time.time()
        )
        
        # Store in buffer
        self.val_metrics.append(metrics)
        
        # Write to CSV
        with open(self.val_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.step,
                metrics.epoch,
                f"{metrics.val_loss:.6f}",
                f"{metrics.val_accuracy:.6f}",
                f"{metrics.timestamp:.0f}" if metrics.timestamp else ""
            ])
    
    def save_summary(
        self,
        best_val_loss: float,
        best_val_acc: float,
        total_steps: int,
        total_time: float,
        config: Optional[Dict] = None
    ):
        """Save training summary to text file."""
        if not self.is_main_process:
            return
        
        with open(self.summary_txt, 'w') as f:
            f.write("="*70 + "\n")
            f.write("LMR Training Summary\n")
            f.write("="*70 + "\n\n")
            
            f.write("Performance:\n")
            f.write(f"  Best validation loss: {best_val_loss:.6f}\n")
            f.write(f"  Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)\n")
            f.write(f"  Total training steps: {total_steps:,}\n")
            f.write(f"  Total training time: {total_time/3600:.2f} hours\n")
            f.write(f"  Average throughput: {total_steps/max(total_time, 1):.2f} steps/sec\n")
            f.write("\n")
            
            if self.train_metrics:
                f.write("Training Metrics:\n")
                f.write(f"  Steps logged: {len(self.train_metrics):,}\n")
                f.write(f"  Final loss: {self.train_metrics[-1].loss:.6f}\n")
                f.write(f"  Final LR: {self.train_metrics[-1].learning_rate:.8f}\n")
                if self.train_metrics[-1].gpu_memory_mb:
                    f.write(f"  Max GPU memory: {self.train_metrics[-1].gpu_memory_mb:.0f} MB\n")
                f.write("\n")
            
            if self.val_metrics:
                f.write("Validation Metrics:\n")
                f.write(f"  Evaluations: {len(self.val_metrics)}\n")
                
                # Find step with best accuracy
                best_val = max(self.val_metrics, key=lambda x: x.val_accuracy)
                f.write(f"  Best accuracy at step: {best_val.step:,}\n")
                f.write(f"    Loss: {best_val.val_loss:.6f}\n")
                f.write(f"    Accuracy: {best_val.val_accuracy:.4f} ({best_val.val_accuracy*100:.2f}%)\n")
                f.write("\n")
                
                # Validation history (last 10)
                f.write("  Recent validation history:\n")
                for vm in self.val_metrics[-10:]:
                    f.write(f"    Step {vm.step:>7,}: loss={vm.val_loss:.4f}, acc={vm.val_accuracy:.4f}\n")
                f.write("\n")
            
            f.write("\n")
            f.write("="*70 + "\n")
            f.write(f"Metrics saved to: {self.output_dir}\n")
            f.write("="*70 + "\n")
        
        print(f"\n✓ Training summary saved to: {self.summary_txt}\n")
    
    def get_best_metrics(self) -> Dict:
        """Get best metrics achieved during training."""
        if not self.val_metrics:
            return {'best_val_loss': float('inf'), 'best_val_acc': 0.0}
        
        best_by_loss = min(self.val_metrics, key=lambda x: x.val_loss)
        best_by_acc = max(self.val_metrics, key=lambda x: x.val_accuracy)
        
        return {
            'best_val_loss': best_by_loss.val_loss,
            'best_val_loss_step': best_by_loss.step,
            'best_val_acc': best_by_acc.val_accuracy,
            'best_val_acc_step': best_by_acc.step
        }