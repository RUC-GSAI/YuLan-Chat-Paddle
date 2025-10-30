import os
import json


job_name = "YuLan_Base_12B-Alpaca_4096-lr_2e5-bs_device_1-acc_1-gpu_16-warmup_10-src_1024-tgt_1024"
src_path = f"/data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/logs/{job_name}.log"
tgt_path = f"/data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/train_loss/{job_name}.txt"

print(tgt_path)

with open(src_path, 'r') as fin:
    dataset = fin.readlines()

with open(tgt_path, 'w') as fout:
    for info in dataset:
        if ('loss' not in info or 'learning_rate' not in info):
            continue
        
        loss = info.split("loss:")[-1].strip().split(', learning_rate')[0].strip()
        loss = float(loss)
        print(loss, file=fout)