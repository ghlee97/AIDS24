CUDA_VISIBLE_DEVICES=0 python3 main.py \
 --model "/raid/lgh/aaai25/models/Llama-2-7b-hf" \
 --epochs 40 --output_dir /raid/lgh/test \
 --calib_dataset wikitext2 --nsamples 256 --seqlen 512 --seed 42 \
 --wbits 2 --abits 16 --group_size 64 --lwc --aug_loss \
 --save_dir /raid/lgh/test