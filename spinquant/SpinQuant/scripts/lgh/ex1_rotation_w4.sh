# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.

w_bits=4

output_dir=/raid/lgh/aids24/EX1/ex1_llama2_7b_spin_w4_wclip



CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nnodes=1 --nproc_per_node=4 optimize_rotation.py \
    --input_model "/raid/lgh/aaai25/models/Llama-2-7b-hf" \
    --output_rotation_path ${output_dir} \
    --output_dir ${output_dir} \
    --logging_dir ${output_dir} \
    --model_max_length 2048 \
    --fp16 False \
    --bf16 True \
    --log_on_each_node False \
    --per_device_train_batch_size 1 \
    --logging_steps 1 \
    --learning_rate 1.5 \
    --weight_decay 0. \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --save_safetensors False \
    --max_steps 100 \
    --w_bits ${w_bits} \
    --w_groupsize 64 \
    --w_clip 