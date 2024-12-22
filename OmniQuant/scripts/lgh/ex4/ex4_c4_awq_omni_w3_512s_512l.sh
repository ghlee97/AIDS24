CUDA_VISIBLE_DEVICES=0 python3 main.py \
 --model "/raid/lgh/aaai25/models/Llama-2-7b-hf" \
 --epochs 20 --output_dir /raid/lgh/aids24/EX4/ex4_llama2_7b_c4_omni_w3_512s_512l \
 --calib_dataset c4 --nsamples 512 --seqlen 512 --seed 42 \
 --wbits 3 --abits 16 --group_size 64 --lwc --aug_loss \
 --save_dir /raid/lgh/aids24/EX4/ex4_llama2_7b_c4_omni_w3_512s_512l