from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, \
Qwen2ForSequenceClassification
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import Gemma2ForSequenceClassification, GemmaTokenizerFast
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch
from transformers import BitsAndBytesConfig

def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_eos_token = True

    tokenizer.padding_side = "right"
    # if args.use_4bit:
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        use_cache=False,
        torch_dtype=torch.float16
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.attn_logit_softcapping = None

    if 'wen' in args.model_name_or_path:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
                            r=args.r, 
                            lora_alpha=args.lora_alpha, 
                            target_modules=[
                                    "q_proj",
                                    "k_proj",
                                    "v_proj",
                                    "o_proj",
                                    "gate_proj",
                                    "up_proj",
                                    "down_proj",
                                    "score",
                                ],
                            bias="none",
                            lora_dropout=0.05, 
                            task_type="CAUSAL_LM"
                            )
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if "score" in name:
            print(name)
            param.requires_grad = True

    if args.lora_path != "none":
        d = torch.load(args.lora_path, map_location=model.device)
        state_dic = {}
        for k,v in d.items():
            state_dic['base_model.model.' + k] = v
        model.load_state_dict(state_dic, strict=False)
        print(f"Loaded LoRA model from {args.lora_path}")
    return model, tokenizer

# def load_model_and_tokenizer(args):
    
#     qlora = {}
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit = True,
#         bnb_4bit_quant_type = "nf4", #nf4 or fp4
#         bnb_4bit_use_double_quant = False,
#         bnb_4bit_compute_dtype=torch.float16,
#         llm_int8_skip_modules = ["score"]
#     )
#     qlora['quantization_config'] = bnb_config
#     print("Using QLoRA")

#     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
#     tokenizer.add_eos_token = True
#     tokenizer.padding_side = "right"
#     # if args.use_4bit:
#     model = AutoModelForSequenceClassification.from_pretrained(
#         args.model_name_or_path,
#         num_labels=2,
#         use_cache=False,
#         torch_dtype=torch.float16,
#         **qlora
#     )
#     # else:
#     #     model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
#     model = prepare_model_for_kbit_training(model)
#     lora_config = LoraConfig(
#                             r=256, 
#                             lora_alpha=128, 
#                             target_modules=[
#                                     "q_proj",
#                                     "k_proj",
#                                     "v_proj",
#                                     "o_proj",
#                                     "gate_proj",
#                                     "up_proj",
#                                     "down_proj",
#                                 ],
#                             bias="none",
#                             lora_dropout=0.05, 
#                             task_type="CAUSAL_LM"
#                             )
#     model = get_peft_model(model, lora_config)
#     if args.lora_path != "none":
#         d = torch.load(args.lora_path, map_location=model.device)
#         state_dic = {}
#         for k,v in d.items():
#             state_dic['base_model.model.' + k] = v
#         model.load_state_dict(state_dic, strict=False)
#         print(f"Loaded LoRA model from {args.lora_path}")
#     return model, tokenizer


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
    d = torch.load(lora_path, map_location=model.device)
    # print(d)
    state_dic = {}
    for k,v in d.items():
        state_dic['base_model.model.' + k] = v
    # print(state_dic)
    # for k,v in state_dic.items():
    #     print(k)
    model.load_state_dict(state_dic, strict=False)
    # # print(model)
    for k,v in model.named_parameters():
        print(v)

