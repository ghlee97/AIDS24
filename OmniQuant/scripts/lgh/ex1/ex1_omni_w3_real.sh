CUDA_VISIBLE_DEVICES=3 python3 main.py \
 --model "/raid/lgh/aaai25/models/Llama-2-7b-hf" \
 --epochs 20 --output_dir /raid/lgh/aids_24/EX1/ex1_llama2_7b_omni_w3_real \
 --calib_dataset wikitext2 --nsamples 256 --seqlen 512 --seed 42 \
 --wbits 3 --abits 16 --group_size 64 --lwc --aug_loss --real_quant \
 --save_dir /raid/lgh/aids_24/EX1/ex1_llama2_7b_omni_w3_real