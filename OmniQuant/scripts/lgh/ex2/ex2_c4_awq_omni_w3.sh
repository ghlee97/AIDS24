CUDA_VISIBLE_DEVICES=3 python3 main.py \
 --model "/raid/lgh/aids24/EX2/ex2_llama2_7b_awq_w3_c4_scale" \
 --epochs 20 --output_dir /raid/lgh/aids24/EX2/ex2_llama2_7b_c4_awq_omni_w3 \
 --calib_dataset c4 --nsamples 256 --seqlen 512 --seed 42 \
 --wbits 3 --abits 16 --group_size 64 --lwc --aug_loss \
 --save_dir /raid/lgh/aids24/EX2/ex2_llama2_7b_c4_awq_omni_w3