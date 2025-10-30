export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1

# MODEL_PATH=/data/czp/model/YuLan-Base-12b-paddle
# MODEL_PATH=/data/czp/model/YuLan-Mini-paddle
# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/YuLan_Mini-Alpaca_256-lr_2e5-bs_device_1-acc_1-gpu_16-parallel_2-warmup_10-src_1024-tgt_1024
# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/YuLan_Base_12B-Alpaca_1024-lr_2e5-bs_device_1-acc_1-gpu_16-warmup_10-src_1024-tgt_1024/checkpoint-192
# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/YuLan_Base_12B-Alpaca_4096-lr_2e5-bs_device_1-acc_1-gpu_16-warmup_10-src_1024-tgt_1024/checkpoint-32/
# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/YuLan_Base_12B-Alpaca_4096-lr_2e5-bs_device_1-acc_1-gpu_16-warmup_10-src_1024-tgt_1024/checkpoint-96-st/

# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/pt-YuLan_Base_12B-OpenWebText_10k-lr_2e5-bs_device_1-acc_8-gpu_16-warmup_10-len_1024/
# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/YuLan_PT_V0-Alpaca_4096-lr_2e5-bs_device_1-acc_8-gpu_16-warmup_10-src_1024-tgt_1024/checkpoint-32/
# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/YuLan_PT_V0-Alpaca_4096-lr_2e5-bs_device_1-acc_8-gpu_16-warmup_10-src_1024-tgt_1024/checkpoint-96/


# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/cpt-YuLan_Base_12B-OpenWebText_10k-lr_2e5-bs_device_1-acc_8-gpu_16-warmup_10-len_1024
# mkdir ${MODEL_PATH}/results/
# mkdir ${MODEL_PATH}/results/gsm8k
# python3 eval/run_eval_paddle_single.py \
#     --model ${MODEL_PATH} \
#     --max_len 512 \
#     --flash_attn \
#     --output_dir ${MODEL_PATH}/results/gsm8k 


# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/YuLan_CPT_V0-Alpaca_4096-lr_2e5-bs_device_1-acc_8-gpu_16-warmup_10-src_1024-tgt_1024/checkpoint-96
# mkdir ${MODEL_PATH}/results/
# mkdir ${MODEL_PATH}/results/gsm8k
# python3 eval/run_eval_paddle_single.py \
#     --model ${MODEL_PATH} \
#     --max_len 512 \
#     --flash_attn \
#     --output_dir ${MODEL_PATH}/results/gsm8k 


# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/YuLan_CPT_V0-Alpaca_4096-lr_2e5-bs_device_1-acc_8-gpu_16-warmup_10-src_1024-tgt_1024/checkpoint-32
# mkdir ${MODEL_PATH}/results/
# mkdir ${MODEL_PATH}/results/gsm8k
# python3 eval/run_eval_paddle_single.py \
#     --model ${MODEL_PATH} \
#     --max_len 512 \
#     --flash_attn \
#     --output_dir ${MODEL_PATH}/results/gsm8k 


MODEL_PATH=~/ckpts/yulan3_base_xpu_sft
mkdir ${MODEL_PATH}/results/
mkdir ${MODEL_PATH}/results/gsm8k
python3 eval/run_eval_paddle_single.py \
    --model ${MODEL_PATH} \
    --max_len 512 \
    --flash_attn \
    --output_dir ${MODEL_PATH}/results/gsm8k \
&> /home/chenzhipeng/yulan_paddle/inference/logs/eval/gsm8k_paddle.log 


# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/YuLan_CPT_V1-Alpaca_ori-lr_1e6-bs_device_1-acc_8-gpu_16-warmup_10-src_1024-tgt_1024/checkpoint-32
# mkdir ${MODEL_PATH}/results/
# mkdir ${MODEL_PATH}/results/gsm8k
# python3 eval/run_eval_paddle_single.py \
#     --model ${MODEL_PATH} \
#     --max_len 512 \
#     --flash_attn \
#     --output_dir ${MODEL_PATH}/results/gsm8k 


