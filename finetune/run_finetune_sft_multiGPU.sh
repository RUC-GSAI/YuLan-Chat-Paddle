#!/bin/bash
python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" run_finetune.py ./config/qwen/sft_argument.json