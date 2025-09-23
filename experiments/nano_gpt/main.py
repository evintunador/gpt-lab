import argparse
import os
import math
from typing import Dict, Any

import torch
import tiktoken
from torch.nn import CrossEntropyLoss

from gpt_lab.configuration import get_config
from gpt_lab.distributed import DistributedManager
from gpt_lab.reproducibility import ReproducibilityManager
from gpt_lab.logger import ExperimentLogger
from gpt_lab.checkpointer import load_checkpoint
from gpt_lab.data_sources.catalog.pretraining.fineweb import FineWebDataset, FineWebSize
from gpt_lab.data_sources.catalog_utils import Split
from gpt_lab.nn_modules.catalog.models import NanoGPT
from gpt_lab.models.catalog.llms import NanoGPTModel
from gpt_lab.train_loops.smart_api import smart_train
from gpt_lab.data_sources.catalog.benchmarks.multiple_choice import WikiQADataset, HellaSwagDataset
from gpt_lab.data_sources.catalog.benchmarks.fill_in_the_blank import ASDivDataset
from gpt_lab.benchmarks.catalog import MultipleChoiceBenchmark, FillInTheBlankBenchmark


def main(cfg: Dict[str, Any], dist: DistributedManager, rep: ReproducibilityManager):
    """Main experiment script for NanoGPT."""
    dist.set_seed(cfg['seed'])
    
    logger = ExperimentLogger(rep.output_dir, dist.rank, dist.is_main_process)
    logger.log_system_info(rep.get_git_info())
    logger.log_hyperparams(cfg)

    enc = tiktoken.get_encoding("gpt2")

    common_dataset_args = {
        "edu": True,
        "streaming": True,
        "world_size": dist.world_size,
        "tokenizer_enc_func": enc.encode,
        "max_seq_len": cfg['model']['max_seq_len']
    }
    train_dataset = FineWebDataset(size=FineWebSize.v10B, seed=cfg['seed'], **common_dataset_args)
    val_dataset = FineWebDataset(size=FineWebSize.v350B, seed=cfg['seed']+1, **common_dataset_args) 
        # TODO: real train vs val split; rn i'm in a rush so using the bigger dataset is a proxy w/ only (1/35)*100% overlap
    
    bsz = cfg['data']['batch_size']
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, num_workers=min(bsz, 4))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, num_workers=min(bsz, 4))

    model = NanoGPT(**cfg['model']).to(dist.device)
    dist.print_on_main(f"Model parameters: {model.get_num_params():,}")

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': cfg['optimizer']['weight_decay']},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=cfg['training']['learning_rate'], betas=(cfg['optimizer']['beta1'], cfg['optimizer']['beta2']))

    """
    start_step = 0
    if cfg['training']['resume_from_checkpoint']:
        dist.print_on_main(f"Resuming from checkpoint: {cfg['training']['resume_from_checkpoint']}")
        resume_data = load_checkpoint(
            cfg['training']['resume_from_checkpoint'],
            map_location=dist.device,
            model=model,
            optimizer=optimizer
        )
        start_step = resume_data.get('metadata', {}).get('step', -1) + 1
        dist.print_on_main(f"Resuming training from step {start_step}")
    
    """
    loss_fn = CrossEntropyLoss()

    # Define the lambda function for the learning rate schedule
    def get_lr_lambda(step):
        # 1) linear warmup for warmup_iters steps
        if step < cfg['training']['warmup_iters']:
            return step / max(1, cfg['training']['warmup_iters'])
        # 2) if it > lr_decay_iters, return min learning rate
        if step > cfg['training']['lr_decay_iters']:
            return cfg['training']['min_lr'] / cfg['training']['learning_rate']
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (step - cfg['training']['warmup_iters']) / (cfg['training']['lr_decay_iters'] - cfg['training']['warmup_iters'])
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        final_lr = cfg['training']['min_lr'] + coeff * (cfg['training']['learning_rate'] - cfg['training']['min_lr'])
        return final_lr / cfg['training']['learning_rate']

    training_kwargs = {}
    training_kwargs['val_loader'] = val_loader
    training_kwargs['logger'] = logger
    training_kwargs['output_dir'] = rep.output_dir
    #training_kwargs['start_step'] = start_step
    training_kwargs['scheduler_kwargs'] = {'lr_lambda': get_lr_lambda}
    training_kwargs['total_steps'] = cfg['training']['total_steps']


    dist.print_on_main("Starting training with smart_train API...")
    result = smart_train(
        model=model, optimizer=optimizer, loss_fn=loss_fn,
        train_loader=train_loader, **training_kwargs
    )
    trained_model = result['model']
    
    if dist.is_main_process:
        dist.print_on_main("\n--- Training Complete ---")
        model_wrapper = NanoGPTModel(trained_model, enc)

        dist.print_on_main("\n--- Generating Samples ---")
        prompts = ["Once upon a time,", "The meaning of life is"]
        for prompt in prompts:
            generation = model_wrapper.generate(prompt, **cfg['generation'])
            dist.print_on_main(f"\nPROMPT: {prompt}\nGENERATION: {generation}")

        if cfg['benchmarks']['run_benchmarks']:
            dist.print_on_main("\n--- Running Benchmarks ---")
            benchmarks_to_run = [
                {"name": "HellaSwag", "dataset": HellaSwagDataset(split=Split.VAL, limit=cfg['benchmarks']['hellaswag_limit']), "runner": MultipleChoiceBenchmark(model_wrapper), "batch_size": 4},
                {"name": "WikiQA", "dataset": WikiQADataset(split=Split.TEST, in_memory=True, limit=cfg['benchmarks']['wikiqa_limit']), "runner": MultipleChoiceBenchmark(model_wrapper), "batch_size": 4},
                {"name": "ASDiv", "dataset": ASDivDataset(target="number", limit=cfg['benchmarks']['asdiv_limit']), "runner": FillInTheBlankBenchmark(model_wrapper), "batch_size": 4},
            ]
            for benchmark in benchmarks_to_run:
                dist.print_on_main(f"\n--- {benchmark['name']} Benchmark ---")
                try:
                    results = benchmark['runner'].run(benchmark['dataset'], batch_size=benchmark['batch_size'])
                    dist.print_on_main(f"{benchmark['name']} Results: {results}")
                    logger.log({"type": "benchmark_results", "name": benchmark['name'], "results": results})
                except Exception as e:
                    dist.print_on_main(f"Failed to run {benchmark['name']} benchmark: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train NanoGPT using the GPT-Lab harness.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
NOTE: Any parameter in the YAML configuration can also be overridden from the command line
using dot notation. For example, to run with the default config but override the number of layers:

  python experiments/nano_gpt/main.py --model.n_layer 16

To specify a different config file:

  python experiments/nano_gpt/main.py --config path/to/your_config.yaml
"""
    )
    parser.add_argument("--resume-from-checkpoint", dest="training.resume_from_checkpoint", type=str, help="Path to a checkpoint to resume from.")
    parser.add_argument("--save-best-model", dest="training.save_best_model", action=argparse.BooleanOptionalAction, help="Enable saving best model based on validation loss.")
    parser.add_argument("--max-steps", dest="training.max_steps", type=int, help="Total training steps.")
    parser.add_argument("--accum-steps", dest="training.accum_steps", type=int, help="Gradient accumulation steps.")
    parser.add_argument("--model-dim", dest="model.n_embd", type=int, help="Model dimension.")
    parser.add_argument("--num-layers", dest="model.n_layer", type=int, help="Number of model layers.")
    parser.add_argument("--num-heads", dest="model.n_head", type=int, help="Number of attention heads.")
    
    with DistributedManager() as dist:
        config = get_config(parser)
        
        runs_dir = os.path.join(os.path.dirname(__file__), "runs")
        with ReproducibilityManager(
            output_dir=runs_dir,
            is_main_process=dist.is_main_process
        ) as rep:
            main(config, dist, rep)
