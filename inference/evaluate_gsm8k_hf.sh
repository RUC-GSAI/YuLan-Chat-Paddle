export CUDA_VISIBLE_DEVICES=8
export OMP_NUM_THREADS=1

python eval/run_eval_hf.py --model /home/huyiwen/YuLan-Mini-Instruct | tee log/hf-gsm8k-0330.log

