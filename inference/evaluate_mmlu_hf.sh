export CUDA_VISIBLE_DEVICES=1

MODEL_PATH=/home/ubuntu/jiajie/tianshu/finetune_torch/ckpt/nvidia1028/checkpoint-32

mkdir ${MODEL_PATH}/torch_results-test

python3  eval/mmlu_hf.py \
    --model ${MODEL_PATH} \
    --save_dir ${MODEL_PATH}/torch_results-test/mmlu \
&> logs/eval/mmlu_torch-test.log