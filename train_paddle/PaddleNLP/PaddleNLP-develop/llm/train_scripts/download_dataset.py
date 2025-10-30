import os
import json
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset

ds = load_dataset("Ankursingh/openwebtext_10K")

tgt_path = "/data/czp/YuLan-Chat-Paddle-main/finetune/data/openwebtext_10k/raw_data.jsonl"

with open(tgt_path, 'w') as fout:
    for data in ds['train']:
        new_data = {
            'text': data['text']
        }
        fout.write(json.dumps(new_data) + '\n')