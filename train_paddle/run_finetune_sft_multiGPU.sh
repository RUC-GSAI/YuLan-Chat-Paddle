#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3,4,5

python3 -u  -m paddle.distributed.launch  run_finetune.py /data/czp/YuLan-Chat-Paddle-main/finetune/config/yulan_base.json