# CUDA_VISIBLE_DEVICES=1 python eval/ceval_paddle.py --model /home/huyiwen/YuLan-Mini-Instruct-paddle | tee log/ceval-paddle-hyw.log
CUDA_VISIBLE_DEVICES=1 python eval/mmlu_paddle.py --model /home/huyiwen/YuLan-Mini-Instruct-paddle | tee log/mmlu-paddle-hyw.log
