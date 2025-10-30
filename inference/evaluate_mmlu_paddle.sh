# CUDA_VISIBLE_DEVICES=1 python eval/ceval_paddle.py --model /home/huyiwen/YuLan-Mini-Instruct-paddle | tee log/ceval-paddle-hyw.log

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=4



MODEL_PATH=/home/ubuntu/
mkdir ${MODEL_PATH}/results-test
python3 eval/mmlu_paddle.py \
    --model ${MODEL_PATH} \
    --save_dir ${MODEL_PATH}/results-test/mmlu \
&> ./logs/eval/yulan_3_base-mmlu-test.log