
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import tqdm
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, BitsAndBytesConfig
import datasets
from datasets import load_dataset
from functools import partial
import gc
import copy




model_path = "/raid/lgh/aaai25/models/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

q_bits = 2           ##########################
q_group_size = -1    ##########################
# per-channel quantization: groupsize=-1

quantization_config = GPTQConfig(bits=q_bits, dataset = "c4", group_size=q_group_size, tokenizer=tokenizer)
q_model_gptq = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", quantization_config=quantization_config)

# for para in q_model_gptq.parameters():
#     para.requires_grad = False
# q_model_gptq.config.use_cache = False
# q_model_gptq.eval()
# #print(q_model_gptq)
# gc.collect()
# torch.cuda.empty_cache()

save_dir = f"/raid/lgh/aids24/Mem/llama2_7b_w{q_bits}_g{q_group_size}"

q_model_gptq.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
