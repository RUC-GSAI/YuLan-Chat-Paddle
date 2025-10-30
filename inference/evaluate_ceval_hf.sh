export HF_ENDPOINT=https://hf-mirror.com
# export PADDLE_XCCL_BACKEND=iluvatar_gpu
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH=/home/ubuntu/jiajie/tianshu/finetune_torch/ckpt/nvidia1028/checkpoint-32
# MODEL_PATH=/data/czp/model/YuLan-Mini-paddle
# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/YuLan_Mini-Alpaca_256-lr_2e5-bs_device_1-acc_1-gpu_16-parallel_2-warmup_10-src_1024-tgt_1024
# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/YuLan_Base_12B-Alpaca_1024-lr_2e5-bs_device_1-acc_1-gpu_16-warmup_10-src_1024-tgt_1024/checkpoint-64
# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/YuLan_Base_12B-Alpaca_4096-lr_2e5-bs_device_1-acc_1-gpu_16-warmup_10-src_1024-tgt_1024/checkpoint-32/
# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/YuLan_Base_12B-Alpaca_4096-lr_2e5-bs_device_1-acc_1-gpu_16-warmup_10-src_1024-tgt_1024/checkpoint-96/

# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/pt-YuLan_Base_12B-OpenWebText_10k-lr_2e5-bs_device_1-acc_8-gpu_16-warmup_10-len_1024/
# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/YuLan_PT_V0-Alpaca_4096-lr_2e5-bs_device_1-acc_8-gpu_16-warmup_10-src_1024-tgt_1024/checkpoint-32/
# MODEL_PATH=/data/czp/YuLan-Chat-Paddle-main/finetune/ckpts/YuLan_PT_V0-Alpaca_4096-lr_2e5-bs_device_1-acc_8-gpu_16-warmup_10-src_1024-tgt_1024/checkpoint-96/
mkdir ${MODEL_PATH}/torch_results-test

python3 eval/ceval_hf.py \
    --model ${MODEL_PATH} \
    --output_dir ${MODEL_PATH}/torch_results-test/ceval \
&> logs/eval/ceval_torch-test.log