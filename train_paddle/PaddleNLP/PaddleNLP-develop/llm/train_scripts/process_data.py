import os
import json

from tqdm import tqdm
from paddlenlp.transformers import AutoTokenizer

model_path = "/data/czp/model/YuLan-Mini"

data_folder = "/data/czp/YuLan-Chat-Paddle-main/finetune/data/alpaca_4096"
os.makedirs(data_folder, exist_ok=True)
src_path = f"/data/czp/YuLan-Chat-Paddle-main/finetune/data/alpaca/raw_train.json"
tgt_path = f"{data_folder}/train.json"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

with open(src_path, 'r') as fin:
    raw_dataset = json.load(fin)
print('Total Data: {}'.format(len(raw_dataset)))

total_data = 0
with open(tgt_path, 'w') as fout:
    for data in tqdm(raw_dataset):
        if (data['input'] != ''):
            continue
        if (total_data >= 4096):
            break
        
        prompt = [
            {"role": "user", "content": data['instruction']},
            {"role": "assistant", "content": data['output']}
        ]
        chat = tokenizer.apply_chat_template(prompt, tokenize=False)
        src = tokenizer(data['instruction'])
        tgt = tokenizer(data['output'])
        full_chat = tokenizer(chat)

        if (
            len(src['input_ids']) > 1024 or 
            len(tgt['input_ids']) > 1024 or 
            len(full_chat['input_ids']) > 2048 or
            len(src['input_ids']) < 8 or 
            len(tgt['input_ids']) < 8 or 
            len(full_chat['input_ids']) < 16
        ):
            continue
        
        new_data = {
            'src': 'Q: ' + data['instruction'].strip() + '\nA:',
            'tgt': data['output'].strip()
        }
        fout.write(json.dumps(new_data, ensure_ascii=False) + '\n')
        total_data = total_data + 1

fout.close()
print('Total Valid Data: {}'.format(total_data))