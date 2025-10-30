#!/bin/bash

# export PADDLE_XCCL_BACKEND=iluvatar_gpu


# JOB_NAME=pt-debug
# rm -r /data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/${JOB_NAME}
# python3 -u  -m paddle.distributed.launch \
#     --devices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
#     run_pretrain.py /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/config/pt-yulan_base_debug.json \
# &> /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/logs/${JOB_NAME}.log


# JOB_NAME=cpt-YuLan_Base_12B-OpenWebText_10k-lr_2e5-bs_device_1-acc_8-gpu_16-warmup_10-len_1024
# rm -r /data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/${JOB_NAME}
# python3 -u  -m paddle.distributed.launch \
#     --devices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
#     run_pretrain.py /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/config/pt-yulan_base-openwebtext_10k.json \
# &> /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/logs/${JOB_NAME}.log


# JOB_NAME=cpt-YuLan_Base_12B-pile-lr_1e6-bs_device_1-acc_64-gpu_16-warmup_10-len_1024
# rm -r /data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/${JOB_NAME}
# python3 -u  -m paddle.distributed.launch \
#     --devices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
#     run_pretrain.py /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/config/cpt-yulan_base-pile.json \
# &> /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/logs/${JOB_NAME}.log


# JOB_NAME=cpt-YuLan_Base_12B-pile-lr_1e6-bs_device_1-acc_1-gpu_16-warmup_10-len_1024
# rm -r /data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/${JOB_NAME}
# python3 -u  -m paddle.distributed.launch \
#     --devices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
#     run_pretrain.py /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/config/cpt-yulan_base-pile.json \
# &> /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/logs/${JOB_NAME}.log


# JOB_NAME=yulan_3_xpu_pt
# rm -r /home/chenzhipeng/ckpts/${JOB_NAME}
# python3 -u  -m paddle.distributed.launch \
#     --devices=0,1,2,3,4,5,6,7 \
#     run_pretrain_new.py /home/chenzhipeng/yulan_paddle/finetune/PaddleNLP/PaddleNLP-develop/llm/config/cpt-yulan_base-pile-xpu.json \
# &> /home/chenzhipeng/yulan_paddle/finetune/PaddleNLP/PaddleNLP-develop/llm/logs/xpu-${JOB_NAME}.log


export XPU_PADDLE_FC_INT32_WITH_LL=1

JOB_NAME=yulan_3_xpu_pt_bf16_2k
rm -r ~/ckpts/${JOB_NAME}
python3 -u  -m paddle.distributed.launch \
    --devices=0,1,2,3,4,5,6,7 \
    run_pretrain_new.py /home/chenzhipeng/yulan_paddle/finetune/PaddleNLP/PaddleNLP-develop/llm/config/cpt-yulan_base-pile-xpu-bf16-2k.json \
&> /home/chenzhipeng/yulan_paddle/finetune/PaddleNLP/PaddleNLP-develop/llm/logs/xpu-${JOB_NAME}.log


# JOB_NAME=cpt-yulan_base-pile-jjj
# rm -r /data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/${JOB_NAME}
# python3 -u  -m paddle.distributed.launch \
#     --devices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
#     run_pretrain.py /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/config/cpt-yulan_base-pile-jjj.json \
# &> /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/logs/${JOB_NAME}.log
