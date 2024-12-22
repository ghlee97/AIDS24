
export CUDA_VISIBLE_DEVICES=0


TARGET_MODEL="/raid/lgh/aids24/EX2/ex2_llama2_7b_awq_omni_w3" ###########################

dataset=arc_easy,winogrande,hellaswag,piqa,arc_challenge

# transformers==4.40.2
echo $TARGET_MODEL

OUTPUT_BASE="csqa_results/EX2/ex2_llama2_7b_awq_omni_w3" ##################################

lm_eval --model hf \
    --model_args pretrained=${TARGET_MODEL},dtype='bfloat16'\
    --tasks ${dataset}\
    --device cuda \
    --output_path ${OUTPUT_BASE} \
    --batch_size 4 \
    --num_fewshot 0


