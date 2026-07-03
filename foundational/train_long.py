#!/usr/bin/env python3
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import yaml
import os
import argparse
import sys
import time
from datetime import timedelta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.architecture_long import LMRLong
from model.config import LMRConfig, estimate_params
from model.long_context import validate_scaling
from data.pytorch_wrapper import RNAStreamingDataset
from data.collator import RNAMLMCollator, RNAMLMCollatorConfig
from training.trainer import Trainer
from training.utils import set_seed, get_lr_scheduler
from tokenizer import MinimalRNATokenizer

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
    assert dist.is_initialized(), f'Rank {rank}: not initialized'
    assert dist.get_rank() == rank
    return (rank, world_size, local_rank)

def staggered_print(rank, world_size, message, delay=0.1):
    time.sleep(rank * delay)
    print(f'[Rank {rank}/{world_size}] {message}', flush=True)
    time.sleep((world_size - rank) * delay)

def load_v0_weights(model, ckpt_path: str, is_main: bool):
    if is_main:
        print(f'  Loading v0 weights: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    cleaned = {k.replace('module.', ''): v for (k, v) in state.items()}
    (missing, unexpected) = model.load_state_dict(cleaned, strict=False)
    if is_main:
        buffer_missing = [k for k in missing if 'freqs' in k or 'window_mask' in k]
        real_missing = [k for k in missing if k not in buffer_missing]
        n_loaded = len(cleaned) - len(unexpected)
        print(f'  ✓ {n_loaded} parameter tensors loaded')
        if real_missing:
            print(f'  ⚠ Missing (non-buffer): {real_missing}')
        if unexpected:
            print(f'  ⚠ Unexpected: {unexpected}')
        if buffer_missing:
            print(f'  ✓ {len(buffer_missing)} non-persistent buffers (RoPE/window) set by architecture')
        if isinstance(ckpt, dict):
            step = ckpt.get('step', ckpt.get('global_step', '?'))
            loss = ckpt.get('val_loss', ckpt.get('best_val_loss', '?'))
            print(f'  Source: step={step}, val_loss={loss}')

def main():
    parser = argparse.ArgumentParser(description='Train LMR-Long')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--init_from', type=str, default=None, help='v0 checkpoint for weight initialization (weights only)')
    parser.add_argument('--resume', type=str, default=None, help='LMR-Long checkpoint to resume (full state)')
    parser.add_argument('--device', type=str, default=None, help='Single-GPU device (e.g. cuda:0)')
    args = parser.parse_args()
    (rank, world_size, local_rank) = setup_distributed()
    if args.device and world_size == 1:
        device = torch.device(args.device)
        if device.type == 'cuda':
            local_rank = int(args.device.split(':')[1]) if ':' in args.device else 0
            torch.cuda.set_device(local_rank)
    elif world_size > 1:
        device = torch.device(f'cuda:{local_rank}')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    is_main = rank == 0
    config = load_config(args.config)
    set_seed(config['seed'] + rank)
    if world_size > 1:
        dist.barrier()
        staggered_print(rank, world_size, 'Process initialized')
        dist.barrier()
    rope_cfg = config.get('rope_scaling', {})
    if is_main:
        print('\n' + '=' * 80)
        print(' LMR-Long: Hybrid-Attention RNA Foundation Model')
        print('=' * 80)
        print(f"  Context       : {config['model']['max_seq_len']}")
        print(f"  Window        : {config['model'].get('attention_window', 'none')}")
        print(f"  RoPE scaling  : {rope_cfg.get('type', 'standard')} (factor={rope_cfg.get('factor', 'N/A')})")
        print(f'  GPUs          : {world_size}')
        if args.init_from:
            print(f'  Init from v0  : {args.init_from}')
        if args.resume:
            print(f'  Resume        : {args.resume}')
        print('=' * 80 + '\n')
    if is_main:
        print('Creating LMR-Long model...')
    model_config = LMRConfig.from_yaml(args.config)
    if is_main:
        param_stats = estimate_params(model_config)
        print(f'\n  d_model      : {model_config.d_model}')
        print(f'  n_layers     : {model_config.n_layers}')
        print(f'  n_heads      : {model_config.n_heads}')
        print(f'  ff_mult      : {model_config.ff_mult}')
        print(f'  max_seq_len  : {model_config.max_seq_len}')
        print(f'  attn_window  : {model_config.attention_window}')
        print(f"  ~params      : {param_stats['total_M']:.1f}M\n")
    for i in range(world_size):
        if rank == i:
            model = LMRLong(model_config, rope_scaling=rope_cfg).to(device)
            if args.init_from and (not args.resume):
                load_v0_weights(model, args.init_from, is_main=rank == 0)
            if config['training'].get('use_gradient_checkpointing', True):
                model.enable_gradient_checkpointing()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        if world_size > 1:
            dist.barrier()
    if is_main:
        if rope_cfg.get('type', 'none') != 'none':
            head_dim = model_config.d_model // model_config.n_heads
            diag = validate_scaling(dim=head_dim, original_len=rope_cfg.get('original_max_seq_len', 2048), extended_len=model_config.max_seq_len, theta=model_config.rope_theta, scaling_type=rope_cfg['type'], factor=rope_cfg.get('factor', 2.0))
            print(f'\n  RoPE validation:')
            print(f"    High-freq preservation : {diag['high_freq_ratio']:.4f}")
            print(f"    Low-freq extension     : {diag['low_freq_ratio']:.4f}")
        schedule = model.get_layer_schedule()
        n_w = schedule.count('window')
        n_f = schedule.count('full')
        print(f'\n  Layer schedule: {n_w} window + {n_f} full')
        print(f'  Schedule: {schedule}\n')
        print('✓ All models created\n')
    if world_size > 1:
        dist.barrier()
    if world_size > 1:
        if is_main:
            print('Initializing DDP...')
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False, broadcast_buffers=True, gradient_as_bucket_view=True)
        dist.barrier()
        if is_main:
            print('✓ DDP initialized\n')
    tokenizer_path = config['data']['tokenizer_path']
    if is_main:
        print(f'Loading tokenizer from {tokenizer_path}...')
    tokenizer = MinimalRNATokenizer.load(tokenizer_path)
    if is_main:
        print(f'✓ Tokenizer loaded (vocab={len(tokenizer.vocab)})\n')
    if world_size > 1:
        dist.barrier()
    if is_main:
        print('Loading datasets...')
    data_dir = config['data']['data_dir']
    max_seq_len = config['data']['max_seq_len']
    train_ratio = config['data']['train_split']
    train_dataset = RNAStreamingDataset(data_dir=data_dir, split='train', train_ratio=train_ratio, max_seq_len=max_seq_len, rank=rank, world_size=world_size, shuffle_files=True)
    val_dataset = RNAStreamingDataset(data_dir=data_dir, split='val', train_ratio=train_ratio, max_seq_len=max_seq_len, rank=rank, world_size=world_size, shuffle_files=False)
    if world_size > 1:
        dist.barrier()
    if is_main:
        print(f'✓ Datasets loaded')
        print(f'  Train  : {len(train_dataset):,} seqs')
        print(f'  Val    : {len(val_dataset):,} seqs')
        print(f'  MaxLen : {max_seq_len}\n')
    if is_main:
        print('Creating dataloaders...')
    collator_config = RNAMLMCollatorConfig(mask_probability=config['data']['mask_probability'], mask_replace_prob=config['data']['mask_replace_prob'], pad_token_id=config['model']['pad_token_id'], mask_token_id=config['model']['mask_token_id'], vocab_size=config['model']['vocab_size'], max_seq_len=config['data']['max_seq_len'])
    collator = RNAMLMCollator(tokenizer, collator_config)
    batch_size = config['training']['batch_size_per_gpu']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=4, pin_memory=True, persistent_workers=False, prefetch_factor=4, multiprocessing_context='fork')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=4, pin_memory=True, persistent_workers=False, prefetch_factor=4, multiprocessing_context='fork')
    if world_size > 1:
        dist.barrier()
    if is_main:
        print('✓ Dataloaders created\n')
    grad_accum = config['training']['gradient_accumulation_steps']
    effective_batch = batch_size * grad_accum * world_size
    num_seqs = config['data']['estimated_train_sequences']
    steps_per_epoch = num_seqs // effective_batch
    max_steps = config['training'].get('max_steps', 30000)
    num_epochs = config['training']['num_epochs']
    num_training_steps = min(max_steps, steps_per_epoch * num_epochs)
    if is_main:
        print('Training plan:')
        print(f'  Effective batch : {effective_batch}')
        print(f'  Steps / epoch   : {steps_per_epoch:,}')
        print(f'  Total steps     : {num_training_steps:,}')
        print(f"  LR              : {config['training']['learning_rate']}")
        est_hrs = num_training_steps / 1.5 / 3600
        print(f'  Est. time       : {est_hrs:.1f}h\n')
    config['training']['num_training_steps'] = num_training_steps
    config['training']['steps_per_epoch'] = steps_per_epoch
    if is_main:
        print('Creating optimizer...')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], betas=(config['training']['adam_beta1'], config['training']['adam_beta2']), eps=config['training']['adam_epsilon'], weight_decay=config['training']['weight_decay'])
    scheduler = get_lr_scheduler(optimizer=optimizer, config=config, num_training_steps=num_training_steps)
    if is_main:
        init_mode = 'fresh (v0 weights loaded)' if args.init_from else 'fresh (random)'
        print(f'✓ Optimizer created ({init_mode})\n')
    if world_size > 1:
        dist.barrier()
    if is_main:
        print('Creating trainer...')
    trainer = Trainer(model=model, train_dataloader=train_loader, val_dataloader=val_loader, optimizer=optimizer, scheduler=scheduler, config=config, device=device, rank=rank, world_size=world_size)
    if args.resume:
        if is_main:
            print(f'Resuming from: {args.resume}')
        trainer.load_checkpoint(args.resume)
        if world_size > 1:
            dist.barrier()
    os.makedirs(config['checkpointing']['output_dir'], exist_ok=True)
    if is_main:
        print('✓ Trainer ready\n')
    if world_size > 1:
        dist.barrier()
    if is_main:
        print('=' * 80)
        print(' Starting LMR-Long Training')
        print('=' * 80 + '\n')
    try:
        trainer.train()
    except KeyboardInterrupt:
        if is_main:
            print('\n' + '=' * 80)
            print('Training interrupted')
            print('=' * 80)
            emergency = os.path.join(config['checkpointing']['output_dir'], f'checkpoint_interrupt_step_{trainer.global_step}.pt')
            trainer._save_checkpoint(emergency, is_best=False)
            print(f'✓ Emergency checkpoint: {emergency}')
    except Exception as e:
        if is_main:
            print(f"\n{'=' * 80}")
            print(f'ERROR: {e}')
            print('=' * 80)
            import traceback
            traceback.print_exc()
        raise
    finally:
        if world_size > 1:
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()
    if is_main:
        print('\n' + '=' * 80)
        print(' LMR-Long Training Complete')
        print('=' * 80)
        print(f'  Best val loss : {trainer.best_val_loss:.4f}')
        print(f'  Best val acc  : {trainer.best_val_acc:.2%}')
        print(f"  Output        : {config['checkpointing']['output_dir']}")
        print('=' * 80 + '\n')
if __name__ == '__main__':
    main()
