export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
python -u  preprocess_data.py \
    --model_name "./model/yulan_3_base" \
    --tokenizer_name "LlamaTokenizer" \
    --data_format "JSON" \
    --input_path "./data/small_pile_new.jsonl" \
    --output_prefix "./data"  \
    --workers 8 \
    --log_interval 5 \
    --data_impl "mmap"