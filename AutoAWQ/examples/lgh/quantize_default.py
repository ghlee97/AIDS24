import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "/raid/lgh/aaai25/models/Llama-2-7b-hf"
quant_path = '/raid/lgh/aids_24/EX1/ex1_llama2_7b_awq_w2_real'
quant_config = { "zero_point": True, "q_group_size": 64, "w_bit": 2, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(
    tokenizer, 
    quant_config=quant_config, 
    calib_data="pileval", 
    max_calib_samples=256, 
    max_calib_seq_len=512
)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')