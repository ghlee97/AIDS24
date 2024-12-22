import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "/raid/lgh/aaai25/models/Llama-2-7b-hf"
quant_path = '/raid/lgh/aids24/EX2/ex2_llama2_7b_awq_w3_c4_scale'
quant_config = { "zero_point": True, "q_group_size": 64, "w_bit": 3, "version": "scale" }
# "version": fake -> fake AWQ quantize
# "version": scale -> only scaling, no quantize
# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)



# Quantize
model.quantize(
    tokenizer, 
    quant_config=quant_config, 
    calib_data="c4", 
    max_calib_samples=256, 
    max_calib_seq_len=512
)

# Save quantized model

#import pdb; pdb.set_trace()

if quant_config["version"] == "fake" or "scale":
    if quant_config["version"] == "fake":
        print("Extract State Dicts of Fake Quantized Model")
    elif quant_config["version"] == "scale":
        print("Extract State Dicts of Scaled Model")
    model.to("cpu")
    sd = {k:v.cpu() for k,v in model.state_dict().items()}
    del model
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    

    print("Load Dummy Model")
    from transformers import AutoModelForCausalLM
    dummy_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    for para in dummy_model.parameters():
        para.requires_grad=False
    dummy_model.config.use_cache=False
    dummy_model.eval()
    
    #import pdb; pdb.set_trace()

    if quant_config["version"] == "fake":
        print("Replace into fake quantized parameters")
    elif quant_config["version"] == "scale":
        print("Replace into scaled parameters")
    import torch.nn as nn
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    for n, m in dummy_model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.modules.sparse.Embedding) or isinstance(m, LlamaRMSNorm):
            print(n)
            m.weight.data = sd["model."+n+".weight"]
    
    if quant_config["version"] == "fake":
        print("Save Fake Quantized Model")
    elif quant_config["version"] == "scale":
        print("Save Scaled Model")
    dummy_model.save_pretrained(quant_path)
    tokenizer.save_pretrained(quant_path)


else:
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')