CUDA_VISIBLE_DEVICES=2 python3 main.py \
 --model "/raid/lgh/aids24/EX2/ex2_llama2_7b_awq_w3_scale" \
 --epochs 20 --output_dir /raid/lgh/aids24/EX3/ex3_llama2_7b_awq_omni_w3_g128 \
 --calib_dataset wikitext2 --nsamples 256 --seqlen 512 --seed 42 \
 --wbits 3 --abits 16 --group_size 128 --lwc --aug_loss \
 --save_dir /raid/lgh/aids24/EX3/ex3_llama2_7b_awq_omni_w3_g128