#!/bin/bash



# python3 -u  -m paddle.distributed.launch \
#     --devices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
#     run_finetune.py /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/config/yulan_base_debug.json \
# &> /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/logs/${JOB_NAME}.log

# JOB_NAME=YuLan_Base_12B-Alpaca_4096-lr_2e5-bs_device_1-acc_1-gpu_16-warmup_10-src_1024-tgt_1024
# rm -r /data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/${JOB_NAME}
# python3 -u  -m paddle.distributed.launch \
#     --devices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
#     run_finetune.py /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/config/yulan_base-alpaca_4096.json \
# &> /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/logs/${JOB_NAME}.log


# JOB_NAME=YuLan_PT_V0-Alpaca_4096-lr_2e5-bs_device_1-acc_8-gpu_16-warmup_10-src_1024-tgt_1024
# rm -r /data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/${JOB_NAME}
# python3 -u  -m paddle.distributed.launch \
#     --devices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
#     run_finetune.py /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/config/yulan_pt_v0-alpaca_4096.json \
# &> /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/logs/${JOB_NAME}.log


# JOB_NAME=YuLan_CPT_V0-Alpaca_4096-lr_2e5-bs_device_1-acc_8-gpu_16-warmup_10-src_1024-tgt_1024
# rm -r /data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/${JOB_NAME}
# python3 -u  -m paddle.distributed.launch \
#     --devices=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
#     run_finetune.py /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/config/yulan_cpt_v0-alpaca_4096.json \
# &> /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/logs/${JOB_NAME}.log


export XPU_PADDLE_FC_INT32_WITH_LL=1

# JOB_NAME=yulan3_base_xpu_sft_bf16
# rm -r /home/chenzhipeng/ckpts/${JOB_NAME}
# python3 -u  -m paddle.distributed.launch \
#     --devices=0,1,2,3,4,5,6,7 \
#     run_finetune.py /home/chenzhipeng/yulan_paddle/finetune/PaddleNLP/PaddleNLP-develop/llm/config/yulan_cpt_v1-alpaca_ori-bf16.json \
#&> /home/chenzhipeng/yulan_paddle/finetune/PaddleNLP/PaddleNLP-develop/llm/logs/${JOB_NAME}.log


JOB_NAME=yulan3_base_xpu_sft
rm -r ~/ckpts/${JOB_NAME}
python3 -u  -m paddle.distributed.launch \
    --devices=0,1,2,3,4,5,6,7 \
    run_finetune.py /home/chenzhipeng/yulan_paddle/finetune/PaddleNLP/PaddleNLP-develop/llm/config/yulan_cpt_v1-alpaca_ori.json \
&> /home/chenzhipeng/yulan_paddle/finetune/PaddleNLP/PaddleNLP-develop/llm/logs/${JOB_NAME}.log
