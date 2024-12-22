CUDA_VISIBLE_DEVICES=2 python3 main.py \
 --model "/raid/lgh/models/Llama-2-7b-hf" \
 --epochs 20 --output_dir /raid/lgh/omni_output/EX1/ex1_llama2_7b_omni_w4 \
 --calib_dataset wikitext2 --nsamples 256 --seqlen 512 --seed 42 \
 --wbits 4 --abits 16 --group_size 64 --lwc --aug_loss \
 --save_dir /raid/lgh/omni_output/EX1/ex1_llama2_7b_omni_w4