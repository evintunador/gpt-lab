import argparse
import os
import time
import pickle
from contextlib import nullcontext

import random
import numpy as np
import torch
import tiktoken

from configuration import get_config, ConfigObject
from distributed import DistributedManager
from reproducibility import ReproducibilityManager
from logger import ExperimentLogger
from checkpointer import save_checkpoint, load_checkpoint
from data_sources.catalog.pretraining.fineweb import PrecachedFineWebDataset, FineWebSize
from data_sources.catalog_utils import Split
from nn_modules.catalog.models import ModdedNanoGPT
from models.catalog.llms import ModdedNanoGPTModel
from optimizers.catalog import Muon
from data_sources.catalog.benchmarks.multiple_choice import WikiQADataset, HellaSwagDataset
from data_sources.catalog.benchmarks.fill_in_the_blank import ASDivDataset
from benchmarks.catalog import MultipleChoiceBenchmark, FillInTheBlankBenchmark


def main(cfg: ConfigObject, dist: DistributedManager, rep: ReproducibilityManager):
    """Main experiment script for ModdedNanoGPT."""
    dist.set_seed(cfg.seed)
    device_type = "cuda" if "cuda" in dist.device.type else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    ctx = torch.amp.autocast(device_type=device_type, dtype=torch_dtype) if torch_dtype != torch.float32 else nullcontext()

    logger = ExperimentLogger(rep.output_dir, dist.rank, dist.is_main_process)
    logger.log_system_info(rep.get_git_info())
    logger.log_hyperparams(cfg.__dict__)

    with open(cfg.tokenizer_path, 'rb') as f:
        tokenizer_config = pickle.load(f)
    
    # Cap tokenizer vocab to match model config
    num_merges_needed = cfg.model.vocab_size - 257 # 256 base tokens + 1 special
    if len(tokenizer_config['mergeable_ranks']) < num_merges_needed:
        raise ValueError(
            f"Tokenizer vocab is smaller than model vocab size. "
            f"Tokenizer has {len(tokenizer_config['mergeable_ranks'])} merges, "
            f"but model needs {num_merges_needed}."
        )
    
    sorted_merges = sorted(tokenizer_config['mergeable_ranks'].items(), key=lambda item: item[1])
    capped_merges = dict(sorted_merges[:num_merges_needed])
    eot_token_id = 256 + len(capped_merges)

    enc = tiktoken.Encoding(
        name=os.path.basename(cfg.tokenizer_path),
        pat_str=tokenizer_config['pat_str'],
        mergeable_ranks=capped_merges,
        special_tokens={"<|endoftext|>": eot_token_id}
    )

    # The PrecachedFineWebDataset will build a cache if it doesn't exist,
    # which will take some time on the first run.
    common_dataset_args = {
        "save_dir": "./data/fineweb_cache",
        "tokenizer_encode_fn": enc.encode,
        "vocab_size": cfg.model.vocab_size,
        "doc_separator": enc.eot_token,
        "size": FineWebSize.v10B, # Using 10B subset for faster setup
    }
    train_dataset = PrecachedFineWebDataset(split=Split.TRAIN, seq_len=cfg.sequence.train_seq_len, **common_dataset_args)
    val_dataset = PrecachedFineWebDataset(split=Split.VAL, seq_len=cfg.sequence.val_seq_len, **common_dataset_args)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=0) # BS=1 to handle packed sequences
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=0)
    
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    model_args = {**cfg.model, 'max_seq_len': max(cfg.sequence.train_seq_len, cfg.sequence.val_seq_len)}
    model = ModdedNanoGPT(**model_args).to(dist.device)
    dist.print_on_main(f"Model parameters: {model.get_num_params():,}")

    # The Muon adapter is designed to handle all parameter groups internally
    all_params = list(model.parameters())
    optimizer = Muon(
        all_params,
        muon_lr=cfg.training.muon_lr,
        adamw_lr=cfg.training.adam_embed_lr, # Use embed_lr as the default AdamW lr
        momentum=cfg.training.muon_momentum,
    )

    # Manually set different LR for head and scalars if needed, as AdamW handles these.
    for group in optimizer.param_groups:
        if not group['use_muon']: # This is the AdamW group
            if any(p in group['params'] for p in model.lm_head.parameters()):
                group['lr'] = cfg.training.adam_head_lr

    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    start_step = 0
    if cfg.training.resume_from_checkpoint:
        dist.print_on_main(f"Resuming from checkpoint: {cfg.training.resume_from_checkpoint}")
        resume_data = load_checkpoint(
            cfg.training.resume_from_checkpoint,
            map_location=dist.device,
            model=model,
            optimizer=optimizer
        )
        
        start_step = resume_data.get('metadata', {}).get('step', -1) + 1
        dist.print_on_main(f"Resuming training from step {start_step}")
        
        if 'rng_states' in resume_data:
            rng_states = resume_data['rng_states']
            torch.set_rng_state(rng_states['torch'].cpu()) # must be on CPU
            np.random.set_state(rng_states['numpy'])
            random.setstate(rng_states['random'])
            dist.print_on_main("Restored RNG states from checkpoint.")

    if cfg.training.use_fp8:
        # Note: Requires Hopper GPU. Will not error on others but may not use FP8.
        from nn_modules.catalog.channel_mixing.fp8_linear import is_hopper_available
        if is_hopper_available():
            dist.print_on_main("Compiling model with FP8 support.")
            model = torch.compile(model)
        else:
            dist.print_on_main("Warning: FP8 requested but Hopper GPU not available. Compiling without FP8.")
            model = torch.compile(model, mode="reduce-overhead")
    else:
        dist.print_on_main("Compiling model.")
        model = torch.compile(model, mode="reduce-overhead")

    if dist.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.local_rank])

    def get_lr_multiplier(step: int):
        progress = step / cfg.training.train_steps
        if progress < 1.0 - cfg.training.cooldown_frac:
            return 1.0
        else:
            cooldown_progress = (progress - (1.0 - cfg.training.cooldown_frac)) / cfg.training.cooldown_frac
            return (1.0 - cooldown_progress) * (1.0 - 0.1) + 0.1 # Decay to 10%

    dist.print_on_main("Starting training...")
    training_time_ms = 0
    t0 = time.perf_counter()

    for step in range(start_step, cfg.training.train_steps):
        lr_mult = get_lr_multiplier(step)
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * lr_mult
        
        # Momentum warmup for Muon
        frac = min(step / 300, 1.0)
        for group in optimizer.param_groups:
            if group['use_muon']:
                group['momentum'] = (1 - frac) * 0.85 + frac * 0.95

        for micro_step in range(cfg.training.grad_acc_steps):
            batch = next(train_iter).squeeze(0) # Remove batch dim
            inputs = batch[:-1].to(dist.device, non_blocking=True)
            targets = batch[1:].to(dist.device, non_blocking=True)

            with ctx:
                loss = model(inputs, targets)
                loss = loss / cfg.training.grad_acc_steps
            loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        if step % cfg.validation.val_every_steps == 0 or step == cfg.training.train_steps - 1:
            torch.cuda.synchronize()
            step_time_ms = (time.perf_counter() - t0) * 1000
            training_time_ms += step_time_ms

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(cfg.validation.val_steps):
                    val_batch = next(val_iter).squeeze(0)
                    v_inputs = val_batch[:-1].to(dist.device)
                    v_targets = val_batch[1:].to(dist.device)
                    with ctx:
                        v_loss = model(v_inputs, v_targets)
                    val_loss += v_loss.item()
            val_loss /= cfg.validation.val_steps
            model.train()
            
            if dist.is_distributed:
                val_loss_tensor = torch.tensor(val_loss, device=dist.device)
                dist.all_reduce(val_loss_tensor)
                val_loss = val_loss_tensor.item() / dist.world_size
            
            avg_step_ms = training_time_ms / (step + 1)
            log_data = {
                "step": step,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]['lr'],
                "muon_momentum": optimizer.param_groups[0]['momentum'],
                "avg_step_ms": avg_step_ms,
            }
            logger.log(log_data, print_to_console=True)

            if cfg.save_model and rep.output_dir:
                raw_model = model.module if dist.is_distributed else model
                checkpoint_path = save_checkpoint(
                    save_dir=os.path.join(rep.output_dir, "checkpoints"),
                    filename=f"step_{step}.pt",
                    metadata={"step": step, "val_loss": val_loss, "config": cfg.__dict__},
                    model=raw_model,
                    optimizer=optimizer,
                )
                logger.info(f"Saved checkpoint to {checkpoint_path}")

            t0 = time.perf_counter()

    dist.print_on_main("\n--- Generating Samples ---")
    raw_model = model.module if dist.is_distributed else model
    
    # Wrap the trained nn.Module with the high-level model interface
    model_wrapper = ModdedNanoGPTModel(raw_model, enc)

    prompts = [
        "Once upon a time,",
        "The meaning of life is",
    ]
    for prompt in prompts:
        generation = model_wrapper.generate(prompt, max_new_tokens=32, temperature=0.8, top_k=200)
        dist.print_on_main(f"\nPROMPT: {prompt}")
        dist.print_on_main(f"GENERATION: {generation}")

    if dist.is_main_process:
        dist.print_on_main("\n--- Running Benchmarks ---")

        dist.print_on_main("\n--- HellaSwag Benchmark (Multiple Choice) ---")
        try:
            hellaswag_dataset = HellaSwagDataset(split=Split.VAL, limit=500)
            mc_benchmark = MultipleChoiceBenchmark(model_wrapper)
            hellaswag_results = mc_benchmark.run(hellaswag_dataset, batch_size=4) # Small batch for LLMs
            dist.print_on_main(f"HellaSwag Results: {hellaswag_results}")
            logger.log({"type": "benchmark_results", "name": "HellaSwag", "results": hellaswag_results})
        except Exception as e:
            dist.print_on_main(f"Failed to run HellaSwag benchmark: {e}")

        dist.print_on_main("\n--- WikiQA Benchmark (Multiple Choice) ---")
        try:
            wiki_dataset = WikiQADataset(split=Split.TEST, in_memory=True, limit=500)
            mc_benchmark = MultipleChoiceBenchmark(model_wrapper)
            wiki_results = mc_benchmark.run(wiki_dataset, batch_size=4) # Small batch for LLMs
            dist.print_on_main(f"WikiQA Results: {wiki_results}")
            logger.log({"type": "benchmark_results", "name": "WikiQA", "results": wiki_results})
        except Exception as e:
            dist.print_on_main(f"Failed to run WikiQA benchmark: {e}")

        dist.print_on_main("\n--- ASDiv Benchmark (Fill-in-the-Blank) ---")
        try:
            asdiv_dataset = ASDivDataset(target="number", limit=500)
            fitb_benchmark = FillInTheBlankBenchmark(model_wrapper)
            asdiv_results = fitb_benchmark.run(asdiv_dataset, batch_size=4) # Small batch for LLMs
            dist.print_on_main(f"ASDiv Results: {asdiv_results}")
            logger.log({"type": "benchmark_results", "name": "ASDiv", "results": asdiv_results})
        except Exception as e:
            dist.print_on_main(f"Failed to run ASDiv benchmark: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ModdedNanoGPT. Overrides config with CLI args.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
NOTE: Any parameter in the YAML configuration can also be overridden from the command line
using dot notation, which is useful for parameters not exposed below. For example:

  python experiments/modded_nano_gpt/main.py --config experiments/modded_nano_gpt/config.yaml \\
    --model.mlp_ratio 8 \\
    --training.muon_lr 0.01
"""
    )

    # Core arguments
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")

    # Key hyperparameters for convenience
    parser.add_argument("--resume-from-checkpoint", dest="training.resume_from_checkpoint", type=str, help="Path to a checkpoint to resume from. Overrides config.")
    parser.add_argument("--seed", dest="seed", type=int, help="Random seed.")
    parser.add_argument("--save-model", dest="save_model", action=argparse.BooleanOptionalAction, help="Enable/disable model saving.")
    parser.add_argument("--train-steps", dest="training.train_steps", type=int, help="Total training steps.")
    parser.add_argument("--grad-acc-steps", dest="training.grad_acc_steps", type=int, help="Gradient accumulation steps.")
    parser.add_argument("--model-dim", dest="model.model_dim", type=int, help="Model dimension.")
    parser.add_argument("--num-layers", dest="model.num_layers", type=int, help="Number of model layers.")
    parser.add_argument("--num-heads", dest="model.num_heads", type=int, help="Number of attention heads.")
    
    with DistributedManager() as dist:
        config = get_config(parser)
        with ReproducibilityManager(config.experiment_name, is_main_process=dist.is_main_process) as rep:
            main(config, dist, rep)
