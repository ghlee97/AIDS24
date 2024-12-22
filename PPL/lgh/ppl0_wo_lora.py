import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns
import copy
import json
from tqdm import tqdm as tqdm


from ppl_utils import eval_ppl


from peft import LoraConfig, get_peft_model, PeftModel


base_path = "/raid/lgh/aids24/EX1/ex1_llama2_7b_awq_w2_fake_manual_w3scale"
output_dir = "ppl_results/EX1/llama2-7b-awq_w2_fake_manual_w3scale"

print("PPL Evaluation - Without LoRA")
print(base_path)
print(output_dir)

model = AutoModelForCausalLM.from_pretrained(
    base_path,
    torch_dtype = torch.float32, #torch.float16
    device_map = "cuda",
    trust_remote_code=True
)

#model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
for para in model.parameters():
    para.requires_grad = False
model.config.use_cache = False
model.eval()


dumped = json.dumps(
    results, indent=2, ensure_ascii=False
)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        f.write(dumped)
        f.close()