import torch
from tqdm import tqdm
import time
from utils import timeSince, compute_metrics, AverageMeter
import torch.distributed as dist


def train_one_epoch(model, dataloader, optimizer, lr_scheduler, device, logger, epoch, config):
    model.train()
    total_loss = 0
    losses = AverageMeter()

    time_start = time.time()
    total_steps = len(dataloader)
    if config.local_rank == 0:
        logger.info(f"Starting training for epoch {epoch}...")
    for step, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}")):
        batch = {k: v.to(device) for k, v in batch.items()}
        # for k,v in model.named_parameters():
        #     print(f"{k} -> requires_grad: {v.requires_grad}")
        outputs = model(**batch)
        loss = outputs.loss

        # Backpropagation and optimization
        model.backward(loss)  # DeepSpeed backward
        model.step()  # DeepSpeed optimizer step

        total_loss += loss.item()
        

        if dist.is_initialized():
            global_loss = torch.tensor(loss.item(), device=device)
            dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
            global_loss = global_loss.item() / dist.get_world_size()
            losses.update(global_loss, n=batch["input_ids"].size(0) * dist.get_world_size())
        else:
            global_loss = loss.item()
            losses.update(global_loss, n=batch["input_ids"].size(0))
        
        # Optional: Log loss every N steps
        if step % 10 == 0 and logger:
            log_message = (
                f"Epoch {epoch + 1}, Elasped {timeSince(time_start, float(step+1)/total_steps)} Step {step}/{total_steps}, Loss: {loss.item():.4f}/({losses.avg:.4f}), "
                f"LR: {lr_scheduler.get_last_lr()[0]:.6f}"
            )
            logger.info(log_message)

    avg_loss = total_loss / len(dataloader)
    if config.local_rank == 0:
        logger.info(f"Epoch {epoch} completed. Average Training Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(model, dataloader, device, logger, epoch, config):
    model.eval()
    total_loss = 0
    if config.local_rank == 0:
        logger.info(f"Starting evaluation for epoch {epoch}...")

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            total_loss += outputs.loss.item()

            logits = outputs.logits
            all_logits.append(logits.detach())
            all_labels.append(batch["labels"].detach())

    avg_loss = total_loss / len(dataloader)

    # Concatenate logits and labels across batches
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if dist.is_initialized():  # Only if running distributed
        gathered_logits = [torch.zeros_like(all_logits) for _ in range(dist.get_world_size())]
        gathered_labels = [torch.zeros_like(all_labels) for _ in range(dist.get_world_size())]

        dist.all_gather(gathered_logits, all_logits)
        dist.all_gather(gathered_labels, all_labels)

        # Concatenate all gathered data
        all_logits = torch.cat(gathered_logits, dim=0).cpu()
        all_labels = torch.cat(gathered_labels, dim=0).cpu()
    
    if config.local_rank == 0:
        acc = compute_metrics(all_logits, all_labels, sigmoid=False)

        logger.info(f"Epoch {epoch} completed. Average Evaluation Loss: {avg_loss:.4f} Average Acc: {acc:.4f}")
        return {'loss':avg_loss, 
                'score': acc}
    return {'loss':avg_loss, 
            'score': None}
