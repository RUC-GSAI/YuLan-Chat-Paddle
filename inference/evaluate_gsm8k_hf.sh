export CUDA_VISIBLE_DEVICES=2

MODEL_PATH=/home/ubuntu/jiajie/tianshu/finetune_torch/ckpt/nvidia1028/checkpoint-32

mkdir ${MODEL_PATH}/torch_results-test
mkdir ${MODEL_PATH}/torch_results-test/gsm8k

python3 eval/run_eval_hf_new.py \
    --model ${MODEL_PATH}  \
    --output_dir ${MODEL_PATH}/torch_results-test/gsm8k \
&> logs/eval/gsm8k_torch-test.log