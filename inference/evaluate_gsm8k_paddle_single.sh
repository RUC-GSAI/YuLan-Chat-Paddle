
export OMP_NUM_THREADS=1
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=5



# MODEL_PATH=/data/model/yulan_3_base
# mkdir ${MODEL_PATH}/results/
# mkdir ${MODEL_PATH}/results/gsm8k
# python3 eval/run_eval_paddle_single.py \
#     --model ${MODEL_PATH} \
#     --max_len 512 \
#     --flash_attn \
#     --output_dir ${MODEL_PATH}/results/gsm8k \
# &> /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/logs/eval/yulan_3_base-gsm8k.log


MODEL_PATH=/home/ubuntu/
mkdir ${MODEL_PATH}/results-test/
mkdir ${MODEL_PATH}/results-test/gsm8k
python3 eval/run_eval_paddle_single.py \
    --model ${MODEL_PATH} \
    --max_len 512 \
    --output_dir ${MODEL_PATH}/results-test/gsm8k \
&> ./logs/eval/yulan_3_base-gsm8k-test.log