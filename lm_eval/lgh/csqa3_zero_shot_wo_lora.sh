
export CUDA_VISIBLE_DEVICES=3


TARGET_MODEL="/raid/lgh/aids24/EX4/ex4_llama2_7b_c4_omni_w3_256s_1024l"

dataset=arc_easy,winogrande,hellaswag,piqa,arc_challenge


echo $TARGET_MODEL

OUTPUT_BASE="csqa_results/EX4/ex4_llama2_7b_c4_omni_w3_256s_1024l"

lm_eval --model hf \
    --model_args pretrained=${TARGET_MODEL},dtype='bfloat16'\
    --tasks ${dataset}\
    --device cuda \
    --output_path ${OUTPUT_BASE} \
    --batch_size 4 \
    --num_fewshot 0


