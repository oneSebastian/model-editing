from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    GPTJForCausalLM,
    LlamaForCausalLM,
    GPTNeoXForCausalLM,
)

def load_model(model_name, device="cuda"):
    if model_name in ["gpt2-xl", "gpt2-large", "gpt2-medium"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
    elif model_name == "gpt-j":
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B', clean_up_tokenization_spaces=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPTJForCausalLM.from_pretrained('EleutherAI/gpt-j-6B', pad_token_id=tokenizer.eos_token_id)
    elif model_name == "gpt-neo":
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
        tokenizer.pad_token = tokenizer.eos_token
        model = GPTNeoXForCausalLM.from_pretrained('EleutherAI/gpt-neox-20b', device_map="auto", offload_folder="offload", offload_state_dict=True, pad_token_id=tokenizer.eos_token_id)
    elif model_name == "llama":
        tokenizer = AutoTokenizer.from_pretrained(f'huggyllama/{model_name}', use_fast=False, add_bos_token=False)
        tokenizer.pad_token = tokenizer.eos_token
        model = model = LlamaForCausalLM.from_pretrained(f'huggyllama/{model_name}', device_map="auto", offload_folder="offload", offload_state_dict=True)
    else:
        raise ValueError(f"{model_name} is not a supported model type.")
    return model.to(device), tokenizer
