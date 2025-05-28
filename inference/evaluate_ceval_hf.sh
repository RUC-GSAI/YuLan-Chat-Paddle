export CUDA_VISIBLE_DEVICES=3

# python eval/ceval_hf.py --model /home/huyiwen/YuLan-Mini-Instruct | tee log/hf-ceval-0323.log
python eval/mmlu_hf.py --model /home/huyiwen/YuLan-Mini-Instruct | tee log/hf-mmlu-0323.log


