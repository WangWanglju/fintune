from dataclasses import dataclass, field
import torch
import argparse


@dataclass
class Config:
    """
    Training configuration, including both general and DeepSpeed-specific options.
    """
    # General training configuration
    debug: bool = True
    train_dataset_path: str = "/root/autodl-tmp/WSDM/input/train.csv"
    model_name_or_path: str = "/root/autodl-tmp/WSDM/working/gemma-2-9b-it-bnb-4bit"
    output_dir: str = "exp/test"
    lora_path: str = 'none'
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1
    learning_rate: float = 5e-5
    max_prompt_length: int = 512
    max_length: int = 2048
    num_warmup_steps: int = 100
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    n_splits: int = 5  # For cross-validation
    train: list = field(default_factory=lambda: [0, ])  # For cross-validation folds

    # DeepSpeed-specific options
    zero_stage: int = 2  # ZeRO optimization stage
    fp16: bool = True  # Enable FP16 mixed precision training
    bf16: bool = False
    local_rank: int = -1  # For distributed training

    
    lr_scheduler_type: str = 'cosine'
    num_warmup_steps: int = 100

    def get_deepspeed_config(self) -> dict:
        """
        Generate a minimal DeepSpeed configuration dictionary.

        Returns:
            dict: DeepSpeed configuration.
        """
        
        return {
            "train_micro_batch_size_per_gpu": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "fp16": {"enabled": self.fp16},
            'bf16': {'enabled': self.bf16},
            # 'pin_memory': True,
            # "gradient_clipping": 1.0,    
            # "prescale_gradients": False,
            # "wall_clock_breakdown": False,
            "zero_optimization": {
                "stage": self.zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if self.zero_stage == 3 else "none"
                },
                "offload_param": {
                    "device": "cpu" if self.zero_stage == 3 else "none"
                },

            },
        }

    def update_from_args(self, args: argparse.Namespace):
        """
        Update the default configuration with argparse arguments.

        Args:
            args (Namespace): Parsed arguments from argparse.
        """
        for key, value in vars(args).items():
            if value is not None:  # Only update if the argument is provided
                setattr(self, key, value)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training configuration.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Fine-tune a model with DeepSpeed and optional overrides.")
    parser.add_argument("--train_dataset_path", type=str, help="Path to the training dataset (CSV).")
    parser.add_argument("--dev_dataset_path", type=str, help="Path to the development dataset (CSV).")
    parser.add_argument("--model_name_or_path", type=str, help="Path to the pre-trained model.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the outputs.")
    parser.add_argument("--train_batch_size", type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, help="Batch size for evaluation.")
    parser.add_argument("--num_train_epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, help="Learning rate.")
    parser.add_argument("--max_prompt_length", type=int, help="Maximum prompt length.")
    parser.add_argument("--max_length", type=int, help="Maximum input sequence length.")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation steps.")
    parser.add_argument("--num_warmup_steps", type=int, help="Number of warmup steps.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--n_splits", type=int, help="Number of folds for cross-validation.")
    parser.add_argument("--device", type=str, help="Device to run training on ('cuda' or 'cpu').")
    parser.add_argument("--zero_stage", type=int, help="ZeRO optimization stage for DeepSpeed.")
    parser.add_argument("--fp16", type=bool, help="Enable FP16 mixed precision training.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    return parser.parse_args()


def get_config() -> Config:
    """
    Load and merge configuration from default settings and command-line arguments.

    Returns:
        Config: Final configuration object.
    """
    args = parse_args()
    config = Config()
    config.update_from_args(args)
    return config
