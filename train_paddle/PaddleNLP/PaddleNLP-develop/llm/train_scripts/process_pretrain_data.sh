cd /data/czp/YuLan-Chat-Paddle-main/finetune/PaddleNLP/PaddleNLP-develop/llm/tools/preprocess

# python3 -u create_pretraining_data.py \
#     --model_name_or_path /data/czp/model/YuLan-Base-12b-paddle \
#     --input_path /data/czp/YuLan-Chat-Paddle-main/finetune/data/openwebtext_10k/raw_data.jsonl \
#     --output_prefix openwebtext  \
#     --data_format "JSON" \
#     --json_key "text" \
#     --data_impl "mmap" \
#     --append_eos \
#     --log_interval 5 \
#     --workers 8
# mv openwebtext* /data/czp/YuLan-Chat-Paddle-main/finetune/data/openwebtext_10k



# python3 -u create_pretraining_data.py \
#     --model_name_or_path /data/czp/model/YuLan-Mini-paddle \
#     --input_path /data/czp/YuLan-Chat-Paddle-main/finetune/data/openwebtext_10k/raw_data.jsonl \
#     --output_prefix openwebtext  \
#     --data_format "JSON" \
#     --json_key "text" \
#     --data_impl "mmap" \
#     --append_eos \
#     --log_interval 5 \
#     --workers 8
# mv openwebtext* /data/czp/YuLan-Chat-Paddle-main/finetune/data/openwebtext_10k-yulan_mini


# python3 -u create_pretraining_data.py \
#     --model_name_or_path /data/czp/model/YuLan-Base-12b-paddle \
#     --input_path /data/czp/YuLan-Chat-Paddle-main/finetune/data/pretrain_pile/small_pile_new.jsonl \
#     --output_prefix pile  \
#     --data_format "JSON" \
#     --json_key "text" \
#     --data_impl "mmap" \
#     --append_eos \
#     --log_interval 5 \
#     --workers 1
mkdir /data/czp/YuLan-Chat-Paddle-main/finetune/data/pile_new
mv pile* /data/czp/YuLan-Chat-Paddle-main/finetune/data/pile_new