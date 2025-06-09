import torch
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    GPTJForCausalLM,
    LlamaForCausalLM,
    GPTNeoXForCausalLM,
    AutoModelForCausalLM,
)

def load_model(model_name, device="cuda"):
    if model_name in ["gpt2-xl", "gpt2-large", "gpt2-medium"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).to(device)
    elif model_name == "gpt-j":
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B', clean_up_tokenization_spaces=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPTJForCausalLM.from_pretrained('EleutherAI/gpt-j-6B', pad_token_id=tokenizer.eos_token_id).to(device)
    elif model_name == "gpt-neo":
        tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
        tokenizer.pad_token = tokenizer.eos_token
        model = GPTNeoXForCausalLM.from_pretrained('EleutherAI/gpt-neox-20b', device_map="auto", offload_folder="offload", offload_state_dict=True, pad_token_id=tokenizer.eos_token_id)
    elif model_name == "llama":
        tokenizer = AutoTokenizer.from_pretrained(f'huggyllama/{model_name}', use_fast=False, add_bos_token=False)
        tokenizer.pad_token = tokenizer.eos_token
        model = model = LlamaForCausalLM.from_pretrained(f'huggyllama/{model_name}', device_map="auto", offload_folder="offload", offload_state_dict=True)
    elif model_name == "qwen_32B":
        model_name = "Qwen/Qwen2.5-32B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", attn_implementation="flash_attention_2", device_map="auto")
        #model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    elif model_name == "mistral_7B":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3", device_map=None, pad_token_id=tokenizer.eos_token_id).to(device)
        #model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3", device_map="auto", pad_token_id=tokenizer.eos_token_id)
    elif model_name == "mistral_7B_instruct":
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", device_map=None, pad_token_id=tokenizer.eos_token_id).to(device)
    else:
        raise ValueError(f"{model_name} is not a supported model type.")
    return model, tokenizer
