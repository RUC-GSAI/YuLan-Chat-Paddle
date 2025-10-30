#!/bin/bash

export PADDLE_XCCL_BACKEND=iluvatar_gpu
export PADDLE_TRAINER_ENDPOINTS="11.1.114.11:43576,11.1.114.20:43576"
export PADDLE_TRAINER_ID=1

JOB_NAME=cpt-YuLan_Base_12B-pile-lr_1e6-bs_device_1-acc_1-gpu_16-node_2-warmup_0-len_1024
rm -r /data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/${JOB_NAME}
python3 -u  -m paddle.distributed.launch \
    --devices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    --master=11.1.114.20:43576 \
    --servers=[11.1.114.11:43576] \
    --nnodes=2 \
    --job_id=0 \
    run_pretrain_new.py /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/config/cpt-yulan_base-pile-multi_node.json \
&> /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/logs/new-${JOB_NAME}.log
