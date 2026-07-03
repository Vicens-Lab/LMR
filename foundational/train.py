#!/usr/bin/env python3
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import yaml
import os
import argparse
import sys
import time
from datetime import timedelta
from typing import Dict, Any, Optional
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.pytorch_wrapper import get_datasets
from data.collator import RNAMLMCollator, RNAMLMCollatorConfig
from training.utils import set_seed, get_lr_scheduler
from tokenizer import MinimalRNATokenizer
from lmr_g.config_foundation import LMRFoundationConfig, estimate_foundation_params
from lmr_g.architecture_foundation import LMRFoundation, create_lmr_foundation
from lmr_g.diagnostics import run_full_diagnostics, DiagnosticsTracker
from lmr_g.schedulers import CombinedScheduler
from lmr_g.metrics_logger import MetricsLogger

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_distributed():
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        return (0, 1, 0)
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=1800), world_size=world_size, rank=rank)
    assert dist.is_initialized(), f'Rank {rank}: Process group not initialized!'
    return (rank, world_size, local_rank)

def staggered_print(rank, world_size, message, delay=0.1):
    time.sleep(rank * delay)
    print(f'[Rank {rank}/{world_size}] {message}', flush=True)
    time.sleep((world_size - rank) * delay)

class FoundationTrainer:

    def __init__(self, model, train_dataloader, val_dataloader, optimizer, scheduler, config: Dict[str, Any], model_config: LMRFoundationConfig, device: torch.device, rank: int=0, world_size: int=1):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.config = config
        self.model_config = model_config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.use_amp = config['training'].get('use_amp', True) and device.type == 'cuda'
        amp_dtype_str = config['training'].get('amp_dtype', 'bfloat16')
        self.amp_dtype = torch.bfloat16 if amp_dtype_str == 'bfloat16' else torch.float16
        self.scaler = GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        total_steps = config['training'].get('max_steps', 100000)
        self.foundation_scheduler = CombinedScheduler(model.module if hasattr(model, 'module') else model, model_config, total_steps)
        self.diagnostics_tracker = DiagnosticsTracker() if self.is_main_process else None
        self.log_steps = config['logging'].get('log_steps', 50)
        self.diagnostic_interval = getattr(model_config, 'diagnostic_interval', 100)
        self.max_grad_norm = config['training'].get('max_grad_norm', 1.0)
        self.grad_accum_steps = config['training'].get('gradient_accumulation_steps', 1)
        self.save_steps = config['checkpointing'].get('save_steps', 5000)
        self.eval_steps = config['checkpointing'].get('eval_steps', 2500)
        self.output_dir = config['checkpointing']['output_dir']
        if self.is_main_process:
            try:
                self.metrics_logger = MetricsLogger(self.output_dir, rank=rank)
            except:
                self.metrics_logger = None
        else:
            self.metrics_logger = None

    def train(self):
        max_steps = self.config['training'].get('max_steps', 100000)
        if self.is_main_process:
            print('\n' + '=' * 70)
            print('Starting LMR-Foundation Training')
            print('=' * 70)
            print(f'Max steps: {max_steps:,}')
            print(f"Plücker bias: {('ENABLED' if self.model_config.use_plucker_bias else 'disabled')}")
            print(f"Grassmann: {('ENABLED' if self.model_config.use_grassmann else 'disabled')}")
            print(f"Curriculum: {('ENABLED' if self.model_config.use_curriculum else 'disabled')}")
            print('=' * 70 + '\n')
        if self.world_size > 1:
            dist.barrier()
        self.model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        step_start_time = time.time()
        train_iter = iter(self.train_dataloader)
        for step in range(self.global_step, max_steps):
            self.global_step = step
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)
                self.epoch += 1
                if self.is_main_process:
                    print(f'\n>>> Epoch {self.epoch} complete <<<\n')
            (loss, metrics) = self._training_step(batch, step)
            running_loss += loss
            running_correct += metrics.get('correct', 0)
            running_total += metrics.get('total', 1)
            self.foundation_scheduler.step(step)
            if step > 0 and step % self.log_steps == 0:
                if self.is_main_process:
                    avg_loss = running_loss / self.log_steps
                    accuracy = running_correct / max(running_total, 1)
                    lr = self.lr_scheduler.get_last_lr()[0]
                    gamma = self.foundation_scheduler.get_gamma()
                    phase = self.foundation_scheduler.get_phase_name()
                    elapsed = time.time() - step_start_time
                    steps_per_sec = self.log_steps / elapsed
                    print(f'Step {step:6d} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2%} | LR: {lr:.2e} | γ: {gamma:.4f} | Phase: {phase} | {steps_per_sec:.2f} it/s')
                    if self.metrics_logger:
                        self.metrics_logger.log_train_step(step=step, epoch=self.epoch, loss=avg_loss, learning_rate=lr, throughput=steps_per_sec, gamma=gamma, phase=phase)
                running_loss = 0.0
                running_correct = 0
                running_total = 0
                step_start_time = time.time()
            if step > 0 and step % self.diagnostic_interval == 0:
                self._run_diagnostics(step, batch)
            if step > 0 and step % self.eval_steps == 0:
                (val_loss, val_acc) = self._evaluate()
                if self.is_main_process:
                    print(f'\n  [Eval] Step {step} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}')
                    if self.metrics_logger:
                        self.metrics_logger.log_validation(step=step, epoch=self.epoch, val_loss=val_loss, val_accuracy=val_acc)
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                        self.best_val_acc = val_acc
                        print(f'  [Eval] New best! Saving checkpoint...')
                        self._save_checkpoint(os.path.join(self.output_dir, 'checkpoint_best.pt'), is_best=True)
                    print()
                if self.world_size > 1:
                    dist.barrier()
                self.model.train()
            if step > 0 and step % self.save_steps == 0:
                if self.is_main_process:
                    self._save_checkpoint(os.path.join(self.output_dir, f'checkpoint_step_{step}.pt'))
        if self.is_main_process:
            self._save_checkpoint(os.path.join(self.output_dir, 'checkpoint_final.pt'))
            print('\n' + '=' * 70)
            print('Training Complete!')
            print(f'Best val loss: {self.best_val_loss:.4f}')
            print(f'Best val acc: {self.best_val_acc:.2%}')
            print('=' * 70)

    def _training_step(self, batch, step):
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            logits = self.model(input_ids, attention_mask=attention_mask)
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            if self.model_config.use_grassmann:
                raw_model = self.model.module if hasattr(self.model, 'module') else self.model
                ortho_penalty = raw_model.get_orthogonality_penalty()
                ortho_weight = self.foundation_scheduler.get_ortho_weight()
                loss = loss + ortho_weight * ortho_penalty
            loss = loss / self.grad_accum_steps
        if self.use_amp and self.amp_dtype == torch.float16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step + 1) % self.grad_accum_steps == 0:
            if self.use_amp and self.amp_dtype == torch.float16:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if self.use_amp and self.amp_dtype == torch.float16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        with torch.no_grad():
            mask = labels != -100
            preds = logits.argmax(dim=-1)
            correct = ((preds == labels) & mask).sum().item()
            total = mask.sum().item()
        return (loss.item() * self.grad_accum_steps, {'correct': correct, 'total': total})

    def _run_diagnostics(self, step, batch):
        if not self.is_main_process:
            return
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        input_ids = batch['input_ids'][:4].to(self.device)
        try:
            diagnostics = run_full_diagnostics(raw_model, input_ids, step)
            if self.diagnostics_tracker:
                self.diagnostics_tracker.update(diagnostics)
            max_spread = diagnostics.get('orth/max_eigenvalue_spread', 0)
            if max_spread > 1000:
                print(f'\n🔴 CRITICAL: Eigenvalue spread = {max_spread:.1f} > 1000')
                print('   Consider switching to Cayley parameterization!\n')
        except Exception as e:
            if self.is_main_process:
                print(f'  [Diagnostics] Warning: {e}')

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0
        max_eval_batches = 100
        for batch in self.val_dataloader:
            if num_batches >= max_eval_batches:
                break
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model(input_ids, attention_mask=attention_mask)
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
            mask = labels != -100
            preds = logits.argmax(dim=-1)
            total_correct += ((preds == labels) & mask).sum().item()
            total_tokens += mask.sum().item()
            num_batches += 1
        avg_loss = total_loss / max(num_batches, 1)
        accuracy = total_correct / max(total_tokens, 1)
        if self.world_size > 1:
            metrics = torch.tensor([avg_loss, accuracy, float(num_batches)], device=self.device, dtype=torch.float32)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            avg_loss = metrics[0].item() / self.world_size
            accuracy = metrics[1].item() / self.world_size
        return (avg_loss, accuracy)

    def _save_checkpoint(self, path, is_best=False):
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        checkpoint = {'model_state_dict': raw_model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.lr_scheduler.state_dict(), 'scaler_state_dict': self.scaler.state_dict(), 'epoch': self.epoch, 'global_step': self.global_step, 'best_val_loss': self.best_val_loss, 'best_val_acc': self.best_val_acc, 'config': self.config, 'model_config': vars(self.model_config)}
        torch.save(checkpoint, path)
        print(f'  ✓ Checkpoint saved: {os.path.basename(path)}')

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        raw_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        if self.is_main_process:
            print(f'✓ Checkpoint loaded: step {self.global_step}, epoch {self.epoch}')

def main():
    parser = argparse.ArgumentParser(description='Train LMR-Foundation v3.0')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--no-plucker', action='store_true', help='Disable Plücker bias (ablation)')
    parser.add_argument('--no-grassmann', action='store_true', help='Disable Grassmann layers (ablation)')
    parser.add_argument('--mlm-only', action='store_true', help='MLM only (no curriculum)')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size per GPU')
    args = parser.parse_args()
    (rank, world_size, local_rank) = setup_distributed()
    if world_size > 1:
        device = torch.device(f'cuda:{local_rank}')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    is_main_process = rank == 0
    config = load_config(args.config)
    model_config = LMRFoundationConfig.from_yaml(args.config)
    if args.batch_size is not None:
        if is_main_process:
            print(f"Overriding batch size: {config['training']['batch_size_per_gpu']} -> {args.batch_size}")
        config['training']['batch_size_per_gpu'] = args.batch_size
    if args.no_plucker:
        model_config.use_plucker_bias = False
    if args.no_grassmann:
        model_config.use_grassmann = False
    if args.mlm_only:
        model_config.use_curriculum = False
    set_seed(config['seed'] + rank)
    if world_size > 1:
        dist.barrier()
        staggered_print(rank, world_size, 'Process initialized')
        dist.barrier()
    if is_main_process:
        print('\n' + '=' * 80)
        print(' LMR-Foundation v3.0: Geometric RNA Language Model')
        print('=' * 80)
        print(f'Distributed training: {world_size} GPUs')
        print(f'Features:')
        print(f"  - Plücker bias: {('ENABLED' if model_config.use_plucker_bias else 'disabled')}")
        print(f"  - Grassmann: {('ENABLED' if model_config.use_grassmann else 'disabled')}")
        print(f"  - Curriculum: {('ENABLED' if model_config.use_curriculum else 'disabled')}")
        if args.resume:
            print(f'\n*** RESUMING FROM: {args.resume} ***')
        print('=' * 80 + '\n')
    if is_main_process:
        print('Creating model...')
        param_stats = estimate_foundation_params(model_config)
        print(f'\nModel Configuration:')
        print(f'  vocab_size: {model_config.vocab_size}')
        print(f'  d_model: {model_config.d_model}')
        print(f'  n_layers: {model_config.n_layers}')
        print(f'  n_heads: {model_config.n_heads}')
        print(f"  Total params: {param_stats['total_M']:.1f}M")
        print(f"  Within 300M budget: {('✓' if param_stats['within_budget'] else '✗')}")
        print()
    for i in range(world_size):
        if rank == i:
            model = create_lmr_foundation(model_config).to(device)
            if config['training'].get('use_gradient_checkpointing', True):
                model.enable_gradient_checkpointing()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        if world_size > 1:
            dist.barrier()
    if is_main_process:
        print('✓ All models created\n')
    if world_size > 1:
        dist.barrier()
    if world_size > 1:
        if is_main_process:
            print('Initializing DDP...')
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False, broadcast_buffers=True, gradient_as_bucket_view=True)
        dist.barrier()
        if is_main_process:
            print('✓ DDP initialized\n')
    tokenizer_path = config['data']['tokenizer_path']
    if is_main_process:
        print(f'Loading MinimalRNATokenizer from {tokenizer_path}...')
    tokenizer = MinimalRNATokenizer.load(tokenizer_path)
    if is_main_process:
        print(f'✓ Tokenizer loaded (vocab size: {len(tokenizer.vocab)})\n')
    if world_size > 1:
        dist.barrier()
    if is_main_process:
        print('Loading datasets...')
    (train_dataset, val_dataset) = get_datasets(config, rank=rank, world_size=world_size)
    if world_size > 1:
        dist.barrier()
    if is_main_process:
        print(f'✓ Datasets loaded')
        print(f'  Train: {len(train_dataset):,} sequences')
        print(f'  Val: {len(val_dataset):,} sequences\n')
    if is_main_process:
        print('Creating dataloaders...')
    collator_config = RNAMLMCollatorConfig(mask_probability=config['data']['mask_probability'], mask_replace_prob=config['data']['mask_replace_prob'], pad_token_id=config['model']['pad_token_id'], mask_token_id=config['model']['mask_token_id'], vocab_size=config['model']['vocab_size'], max_seq_len=config['data']['max_seq_len'])
    collator = RNAMLMCollator(tokenizer, collator_config)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size_per_gpu'], shuffle=False, collate_fn=collator, num_workers=4, pin_memory=True, persistent_workers=False, prefetch_factor=4, multiprocessing_context='fork')
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size_per_gpu'], shuffle=False, collate_fn=collator, num_workers=4, pin_memory=True, persistent_workers=False, prefetch_factor=4, multiprocessing_context='fork')
    if world_size > 1:
        dist.barrier()
    if is_main_process:
        print('✓ Dataloaders created\n')
    batch_size_per_gpu = config['training']['batch_size_per_gpu']
    grad_accum_steps = config['training']['gradient_accumulation_steps']
    effective_batch_size = batch_size_per_gpu * grad_accum_steps * world_size
    num_train_sequences = config['data'].get('estimated_train_sequences', 26000000)
    steps_per_epoch = num_train_sequences // effective_batch_size
    max_steps = config['training']['max_steps']
    num_epochs = config['training'].get('num_epochs', 1)
    num_training_steps = min(max_steps, steps_per_epoch * num_epochs)
    if is_main_process:
        print('Training Configuration:')
        print(f'  Batch per GPU: {batch_size_per_gpu}')
        print(f'  Grad accum: {grad_accum_steps}')
        print(f'  Effective batch: {effective_batch_size}')
        print(f'  Steps per epoch: {steps_per_epoch:,}')
        print(f'  Total steps: {num_training_steps:,}')
        print(f'  Est. time (2 it/s): {num_training_steps / 2 / 3600:.1f}h\n')
    config['training']['num_training_steps'] = num_training_steps
    config['training']['steps_per_epoch'] = steps_per_epoch
    if is_main_process:
        print('Creating optimizer...')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], betas=(config['training']['adam_beta1'], config['training']['adam_beta2']), eps=config['training']['adam_epsilon'], weight_decay=config['training']['weight_decay'])
    scheduler = get_lr_scheduler(optimizer=optimizer, config=config, num_training_steps=num_training_steps)
    if is_main_process:
        print('✓ Optimizer created\n')
    if world_size > 1:
        dist.barrier()
    if is_main_process:
        print('Creating trainer...')
    trainer = FoundationTrainer(model=model, train_dataloader=train_loader, val_dataloader=val_loader, optimizer=optimizer, scheduler=scheduler, config=config, model_config=model_config, device=device, rank=rank, world_size=world_size)
    if args.resume:
        if is_main_process:
            print(f'Resuming from: {args.resume}')
        trainer.load_checkpoint(args.resume)
        if world_size > 1:
            dist.barrier()
    os.makedirs(config['checkpointing']['output_dir'], exist_ok=True)
    if is_main_process:
        print('✓ Trainer created\n')
    if world_size > 1:
        dist.barrier()
    if is_main_process:
        print('=' * 80)
        print(' Starting Training')
        print('=' * 80)
        print()
    try:
        trainer.train()
    except KeyboardInterrupt:
        if is_main_process:
            print('\n' + '=' * 80)
            print('Training interrupted by user')
            print('=' * 80)
            emergency_path = os.path.join(config['checkpointing']['output_dir'], f'checkpoint_interrupt_step_{trainer.global_step}.pt')
            trainer._save_checkpoint(emergency_path)
            print(f'✓ Emergency checkpoint saved: {emergency_path}')
    except Exception as e:
        if is_main_process:
            print(f"\n{'=' * 80}")
            print(f'ERROR: {e}')
            print('=' * 80)
            import traceback
            traceback.print_exc()
        raise
    finally:
        if world_size > 1:
            try:
                dist.barrier(timeout=timedelta(seconds=10))
            except:
                pass
            dist.destroy_process_group()
if __name__ == '__main__':
    main()
