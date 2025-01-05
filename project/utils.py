import logging
import os
import torch.distributed as dist
import math
import time
import torch

def is_main_process():
    """
    Check if the current process is the main (rank=0) process.
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_logger(output_dir, logger_name="train_logger"):
    """
    Setup a logger that writes logs only for the main process.

    Args:
        output_dir (str): Directory to save the log file.
        logger_name (str): Name of the logger.

    Returns:
        logger: Configured logger instance for the main process.
    """
    logger = logging.getLogger(logger_name)

    # Ensure we don't duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Only initialize the logger for the main process
    if is_main_process():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        log_file = os.path.join(output_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logger initialized. Logs will be saved to: {log_file}")

    else:
        # For non-main processes, set the logger to a "do-nothing" logger
        logger.setLevel(logging.CRITICAL)  # Prevent non-main processes from logging anything

    return logger


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


from sklearn.metrics import accuracy_score

def compute_metrics(logits, labels, threshold=0.5, sigmoid=True):
    """
    Compute metrics for binary classification.
    
    Args:
        logits (torch.Tensor): The raw model outputs (logits).
        labels (torch.Tensor): The ground truth labels.
        threshold (float): The decision threshold for classification.
        
    Returns:
        dict: A dictionary containing computed metrics.
    """
    # Apply sigmoid activation to logits for binary classification
    if sigmoid:
        probabilities = torch.sigmoid(logits)
        # Convert probabilities to predictions based on the threshold
        predictions = (probabilities >= threshold).long()
    else:
        all_logits = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(all_logits, dim=1)
        # print(all_logits, predictions)

    # Compute accuracy
    accuracy = accuracy_score(labels.numpy(), predictions.numpy())

    # Return all computed metrics
    return accuracy



def save_model(args, model, tokenizer, sub_fold=None):
    sub_fold  = "model" if sub_fold is None else sub_fold
    output_dir = os.path.join(args.output_dir, sub_fold)
    os.makedirs(output_dir, exist_ok=True)

    tokenizer.save_pretrained(output_dir)
    # model = convert_lora_to_linear_layer(model)
    if args.local_rank == 0:
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save = model_to_save.model
        # model_to_save.save_pretrained(output_dir)

        CONFIG_NAME = "config.json"
        WEIGHTS_NAME = "adapter.bin"

        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(output_dir, CONFIG_NAME)
        save_dict = model_to_save.state_dict()
        final_d = {}
        for k, v in save_dict.items():
            if "lora" in k  or 'score' in k:
                final_d[k] = v
        torch.save(final_d, output_model_file)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def get_optimizer_grouped_parameters(
        model,
        weight_decay,
        lora_lr=5e-4,
        no_decay_name_list=["bias", "LayerNorm.weight"],
        lora_name_list=["lora_right_weight", "lora_left_weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n
                                                    for nd in lora_name_list))
            ],
            "weight_decay":
                weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n
                                                for nd in lora_name_list))
            ],
            "weight_decay":
                weight_decay,
            "lr":
                lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
                0.0,
        },
    ]
    if not optimizer_grouped_parameters[1]["params"]:
        optimizer_grouped_parameters.pop(1)
    return optimizer_grouped_parameters