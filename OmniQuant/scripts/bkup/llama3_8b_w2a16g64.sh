CUDA_VISIBLE_DEVICES=3 python main.py \
 --model "meta-llama/Meta-Llama-3-8B" --eval_ppl \
 --epochs 40 --output_dir /home/leegh/lgh_n24/OmniQuant/output/Llama-3-8b-c4-w2a16g64 \
 --calib_dataset c4 --nsamples 256 --seqlen 512 \
 --wbits 2 --abits 16 --group_size 64 --lwc --aug_loss \
 --tasks winogrande,piqa,hellaswag,arc_easy,arc_challenge \
 --save_dir /home/leegh/lgh_n24/OmniQuant/output/Llama-3-8b-c4-w2a16g64/model