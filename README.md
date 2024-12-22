# AI System Design 2024 Project

Geonho Lee, Sumin Lee, Sungwan Ryu

- AutoAWQ
  - Scale activations and weights to make model show better quantized performance
  - How to Run
    - Example file: AutoAWQ/examples/quantize_c4_ex1_w2.py
    - model_path: baseline model path
    - quant_path: save directory for quantized model
    - quant_config
      - zero_point: True=asymmetric quantization, False=symmetric quantization
      - q_gtoup_size: groupsize
      - w_bit: weight quantization bit precision
      - version
        - "GEMM": default (real quantization with packing)
        - "fake": (added) fake_quantization
        - "scale": (added) only scaled model (no quantization)
    - calib_data: calibration dataset (default="pileval", added "c4" and "wikitext2")
    - max_calib_samples: number of sample sentences
    - max_calib_seq_len: number of tokens per sentence

- OmniQuant
  - Learnable Weight Clipping (LWC) for better quantization factors (scale, zero)
  - How to Run
    - Example file: OmniQuant/scripts/ex1_omni_w3.sh
    - model: baseline model path
    - output_dir: save directory for log
    - save_dir: save_directory for quantized model
    - real_quant: (default=False) True=real quantization with packing, False=fake quantization
    - wbits: weight quantization bit precision
    - abits: activation quantization bit precision (abits=16: weight-only quantization)
    - group_size: groupsize
    - lwc: True=do learnable weight clipping
    - calib_dataset: calibration dataset (examples: wikitext2 or c4)
    - nsmaples: number of sample senteces
    - seqlen: number of tokens per sentence
    - epochs: training epoch (20 epochs for 3,4 wbits & 40 epochs for 2bits)

- AutoGPTQ_memory
  - Memory requirement of real quantized model
  - *gen.py: run AutoGPTQ to generate quantized model
    - model_path: baseline model path
    - q_bits: weight quantization bit precision
    - q_group_size: groupsize
  - mem_check: memory check for quantized model
    - q_bits: weight quantization bit precision
    - q_group_size: groupsize
    - PDB exit -> enter 'q'

- Evaluation
  - PPL
    - Example: PPL/ppl0.ipynb
      - base_path: fp or quantized (fake quantized) model directory
      - eval_ppl: False=no LoRA, True=with LoRA
      - output_dir: save directory for PPL results
  - lm_eval (CSQA benchmark)
    - Example: lm_eval/csqa0_zero_shot_wo_lora._example.sh
      - TARGET_MODEl: fp or quantized (fake quantized) model directory
      - dataet: target task
      - OUTPUT_BASE: save directory for results
      - num_fewshot: number of shot (ex: zero_shot=0, five_shot=5)  






 
