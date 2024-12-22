import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Import Libraries
import tqdm
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, BitsAndBytesConfig
import datasets
from datasets import load_dataset
from functools import partial
import gc
import copy


q_bits = 4           ##########################
q_group_size = 64    ##########################
# per-channel quantization: groupsize=-1

save_dir = f"/raid/lgh/aids24/Mem/llama2_7b_w{q_bits}_g{q_group_size}"

q_model_gptq = AutoModelForCausalLM.from_pretrained(save_dir, device_map="cuda")
gc.collect()
torch.cuda.empty_cache()

import pdb; pdb.set_trace() ## pdb exit -> enter 'q'
