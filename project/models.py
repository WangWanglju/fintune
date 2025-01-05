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
        torch_dtype=torch.bfloat16
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

    # for name, param in model.named_parameters():
        # if "score" in name:
        #     param.requires_grad = True
    if args.lora_path != "none":
        d = torch.load(args.lora_path, map_location=model.device)
        state_dic = {}
        for k,v in d.items():
            state_dic['base_model.model.' + k] = v
        model.load_state_dict(state_dic, strict=False)

    # for name, param in model.named_parameters():
    #     if "score" in name:
    #         print(name)
    #         param.requires_grad = True
    model.print_trainable_parameters()
    return model, tokenizer


if __name__ == '__main__':
    from peft import PeftModel
    
    model_path = '/root/autodl-tmp/WSDM/working/gemma-2-9b-it-bnb-4bit'
    lora_path = '/root/autodl-tmp/WSDM/project/exp/baseline-onlyuse-tailtext-fulldata/best_model_epoch_0/adapter.bin'
    tokenizer = GemmaTokenizerFast.from_pretrained(model_path)
    tokenizer.add_eos_token = True
    tokenizer.padding_side = "right"
    # if args.use_4bit:
    model = Gemma2ForSequenceClassification.from_pretrained(
        model_path,
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
    for k,v in model.named_parameters():
        print(f"{k} -> requires_grad: {v.requires_grad}")
    # d = torch.load(lora_path, map_location=model.device)
    # # print(d)
    # state_dic = {}
    # # for k,v in d.items():
    # #     state_dic['base_model.model.' + k] = v
    # # print(state_dic)
    # # for k,v in state_dic.items():
    # #     print(k)
    # model.load_state_dict(state_dic, strict=False)
    # # print(model)

