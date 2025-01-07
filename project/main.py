import torch
from torch.utils.data import DataLoader
from config import get_config
from datasets import TrainDataset, collate_fn
from models import load_model_and_tokenizer
from train_utils import train_one_epoch, evaluate
from utils import setup_logger, save_model
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import deepspeed
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import os
from transformers import set_seed, get_scheduler
import random
import numpy as np
import math

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_loop(data, fold, config, device, logger):
    deepspeed_config = config.get_deepspeed_config()
    print("DeepSpeed Configuration:")
    print(deepspeed_config)
    data['response_b'] = data.apply(
        lambda row: '\n\n<response_b>: ' + (str(row['response_b']) if pd.notnull(row['response_b']) else 'N/A'),
        axis=1
    )

    data['response_a'] = data.apply(
        lambda row: '\n\n<response_a>: ' + (str(row['response_a']) if pd.notnull(row['response_a']) else 'N/A'),
        axis=1
    )
    train_data = data[data['fold'] != fold].reset_index(drop=True)
    eval_data = data[data['fold'] == fold].reset_index(drop=True)

    if config.debug:
        train_data = train_data.iloc[:4000]
        eval_data = eval_data.iloc[:1000]

    model, tokenizer = load_model_and_tokenizer(config)

    train_dataset = TrainDataset(train_data, tokenizer, config)
    eval_dataset = TrainDataset(eval_data, tokenizer, config)

    # DataLoaders creation:
    if config.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        test_sampler = RandomSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(eval_dataset)

    train_dataloader = DataLoader(train_dataset, 
                                shuffle=(train_sampler is None),
                                batch_size=config.per_device_train_batch_size, 
                                collate_fn=collate_fn,
                                sampler=train_sampler,
                                pin_memory=True
                                )
    eval_dataloader = DataLoader(eval_dataset, 
                                shuffle=(test_sampler is None),
                                batch_size=config.per_device_eval_batch_size, 
                                collate_fn=collate_fn,
                                sampler=test_sampler,
                                pin_memory=True
                                )
    
    # optimizer_grouped_parameters = get_optimizer_grouped_parameters(
    # model, args.weight_decay, args.lora_learning_rate)

    AdamOptimizer =  FusedAdam
    optimizer = AdamOptimizer(model.parameters(),
                              lr=config.learning_rate,
                              betas=(0.9, 0.95))
    
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps)
    max_steps = math.ceil(config.per_device_train_batch_size * num_update_steps_per_epoch)


    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=math.ceil(max_steps * 0.01) if config.num_warmup_steps == 0 else config.num_warmup_steps,
        num_training_steps=config.per_device_train_batch_size * num_update_steps_per_epoch,
    )
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
                                        model=model,
                                        config_params=deepspeed_config,
                                        optimizer=optimizer,
                                        lr_scheduler=lr_scheduler)

    # Training loop
    best_score = -np.inf
    for epoch in range(config.num_train_epochs):
        train_loss = train_one_epoch(model_engine, train_dataloader, optimizer, lr_scheduler, device, logger, epoch, config)
        outputs = evaluate(model_engine, eval_dataloader, device, logger, epoch, config)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Eval Loss={outputs['loss']:.4f}")
        if config.local_rank == 0:
            if outputs['score'] > best_score:
                save_model(config, model, tokenizer, f"best_model_epoch_{epoch}")
                best_score = outputs['score']
                logger.info(f"Saving....\nbest score: {outputs['score']}")

def test(data, fold, config, device, logger):
    deepspeed_config = config.get_deepspeed_config()
    print("DeepSpeed Configuration:")
    print(deepspeed_config)
    data['response_b'] = data.apply(
        lambda row: '\n\n<response_b>: ' + (str(row['response_b']) if pd.notnull(row['response_b']) else 'N/A'),
        axis=1
    )

    data['response_a'] = data.apply(
        lambda row: '\n\n<response_a>: ' + (str(row['response_a']) if pd.notnull(row['response_a']) else 'N/A'),
        axis=1
    )
    train_data = data[data['fold'] != fold].reset_index(drop=True)
    eval_data = data[data['fold'] == fold].reset_index(drop=True)

    if config.debug:
        train_data = train_data.iloc[:4000]
        eval_data = eval_data.iloc[:1000]

    model, tokenizer = load_model_and_tokenizer(config)

    train_dataset = TrainDataset(train_data, tokenizer, config)
    eval_dataset = TrainDataset(eval_data, tokenizer, config)

    # DataLoaders creation:
    if config.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        test_sampler = RandomSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(eval_dataset)

    train_dataloader = DataLoader(train_dataset, 
                                shuffle=(train_sampler is None),
                                batch_size=config.per_device_train_batch_size, 
                                collate_fn=collate_fn,
                                sampler=train_sampler,
                                pin_memory=True
                                )
    eval_dataloader = DataLoader(eval_dataset, 
                                shuffle=(test_sampler is None),
                                batch_size=config.per_device_eval_batch_size, 
                                collate_fn=collate_fn,
                                sampler=test_sampler,
                                pin_memory=True
                                )
    
    # optimizer_grouped_parameters = get_optimizer_grouped_parameters(
    # model, args.weight_decay, args.lora_learning_rate)

    AdamOptimizer =  FusedAdam
    optimizer = AdamOptimizer(model.parameters(),
                              lr=config.learning_rate,
                              betas=(0.9, 0.95))
    
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps)
    max_steps = math.ceil(config.per_device_train_batch_size * num_update_steps_per_epoch)


    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=math.ceil(max_steps * 0.01) if config.num_warmup_steps == 0 else config.num_warmup_steps,
        num_training_steps=config.per_device_train_batch_size * num_update_steps_per_epoch,
    )
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
                                        model=model,
                                        config_params=deepspeed_config,
                                        optimizer=optimizer,
                                        lr_scheduler=lr_scheduler)


    outputs = evaluate(model_engine, eval_dataloader, device, logger, 0, config)
    if logger:
        logger.info(f"best score: {outputs['score']}")

def main():
    config = get_config()
    print("Training Configuration:")
    print(config)

    if config.local_rank == -1:
        device = torch.device("cuda")
    else:
        # Set the GPU device for this process
        torch.cuda.set_device(config.local_rank)
        device = torch.device("cuda", config.local_rank)
        
        # Initialize distributed backend
        try:
            import deepspeed
            deepspeed.init_distributed()
        except ImportError:
            # Fallback to PyTorch native distributed initialization
            import torch.distributed as dist
            dist.init_process_group(backend="nccl")
    rank = config.local_rank
    # world_size = dist.get_world_size()
    if rank == 0:
        os.makedirs(config.output_dir, exist_ok=True)
        # Setup logger (only for the main process)
        logger = setup_logger(config.output_dir)

        logger.info("Training started with distributed training.")
    else:
        logger = None
    # If passed along, set the training seed now.
    set_random_seed(42)

    torch.distributed.barrier()
    # Load data
    data = pd.read_csv(config.train_dataset_path)
    # kf = KFold(n_splits=config.n_splits, shuffle=True, random_state=42)
    
    # for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
    #     data.loc[val_idx, 'fold'] = int(fold)
    # data['fold'] = data['fold'].astype(int)

    for i in range(config.n_splits):
        if i in config.train:
            _ = train_loop(data, i, config, device, logger)
            # _ = test(data, i, config, device, logger)
       

if __name__ == "__main__":
    main()
