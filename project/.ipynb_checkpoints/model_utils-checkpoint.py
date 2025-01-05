from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import Gemma2ForSequenceClassification, GemmaTokenizerFast
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch


def load_model_and_tokenizer(args):
    tokenizer = GemmaTokenizerFast.from_pretrained(args.model_name_or_path)
    tokenizer.add_eos_token = True
    tokenizer.padding_side = "right"
    # if args.use_4bit:
    model = Gemma2ForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        use_cache=False,
        torch_dtype=torch.float16
    )
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
                            r=256, 
                            lora_alpha=128, 
                            target_modules=[
                                    "q_proj",
                                    "k_proj",
                                    "v_proj",
                                    "o_proj",
                                    "gate_proj",
                                    "up_proj",
                                    "down_proj",
                                ],
                            bias="none",
                            lora_dropout=0.05, 
                            task_type="CAUSAL_LM"
                            )
    model = get_peft_model(model, lora_config)
    # if args.lora_path != "none":
    #     model.load_state_dict(torch.load(args.lora_path), strict=False)
    return model, tokenizer
