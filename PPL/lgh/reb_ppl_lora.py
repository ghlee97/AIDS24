import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



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

base_model_family = "llama2-7b" ###########

base_dir = "/raid/lgh/aids"

base_model = "llama2-7b-loftq-w2a16g64-r32"  #############################################

base_path = f"{base_dir}/{base_model}"

adapter_dir = "/raid/lgh/output_models"
#/raid/lgh/output_models/llama2-7b-rtn-w4a16g64_lqat_ag_hid_r16_1e-4_seq128/approx_init
adapter_model = "llama2-7b-loftq-w2a16g64-r32-rilq_qlayer" #######################################

adapter_path = f"{adapter_dir}/{adapter_model}/approx_init"

output_dir = f"rebuttal_ppl_results/{base_model_family}/{adapter_model}"





print(f"base_path: {base_path}")
print(f"adapter_path: {adapter_path}")
print(f"output_dir: {output_dir}")


if os.path.exists(output_dir):
    print("Directory is already taken")
else:
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype = torch.float32,
        device_map = "cuda",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)

    for para in model.parameters():
        para.requires_grad = False
    model.config.use_cache = False
    model.eval()

    results = eval_ppl(model, True, "llama2", "cuda", base_path) #####################

    dumped = json.dumps(
        results, indent=2, ensure_ascii=False
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

        with open(os.path.join(output_dir, "results.json"), "w") as f:
            f.write(dumped)
            f.close()


