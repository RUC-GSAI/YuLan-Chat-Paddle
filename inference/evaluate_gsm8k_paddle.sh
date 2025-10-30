export HF_ENDPOINT=https://hf-mirror.com
export PADDLE_XCCL_BACKEND=iluvatar_gpu

MODEL_PATH=/data/czp/model/YuLan-Base-12b-paddle
# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/YuLan_Base_12B-Alpaca_4096-lr_2e5-bs_device_1-acc_1-gpu_16-warmup_10-src_1024-tgt_1024/checkpoint-96/


# MODEL_PATH=/data/czp/model/YuLan-Mini-paddle
# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/YuLan_Mini-Alpaca_256-lr_2e5-bs_device_1-acc_1-gpu_16-parallel_2-warmup_10-src_1024-tgt_1024
# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/YuLan_Base_12B-Alpaca_1024-lr_2e5-bs_device_1-acc_1-gpu_16-warmup_10-src_1024-tgt_1024/checkpoint-192
# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/YuLan_Base_12B-Alpaca_4096-lr_2e5-bs_device_1-acc_1-gpu_16-warmup_10-src_1024-tgt_1024/checkpoint-96-st/
mkdir ${MODEL_PATH}/results/
mkdir ${MODEL_PATH}/results/gsm8k-multigpu

python3 -u  -m paddle.distributed.launch \
    --devices=2,3 \
    eval/run_eval_paddle.py \
    --model ${MODEL_PATH} \
    --max_len 256 \
    --flash_attn \
    --output_dir ${MODEL_PATH}/results/gsm8k-multigpu 

# python3 eval/run_eval_paddle.py \
#     --model ${MODEL_PATH} \
#     --max_len 512 \
#     --flash_attn \
#     --output_dir ${MODEL_PATH}/results/gsm8k 

# python3 -u  -m paddle.distributed.launch \
#     --devices=1,2 \
#     eval/run_eval_paddle.py \
#     --model /data/czp/model/YuLan-Chat-3-12b-paddle

