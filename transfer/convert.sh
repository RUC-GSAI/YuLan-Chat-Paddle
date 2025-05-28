#!/bin/bash

# 定义变量
MODEL_PATH="/media/wangyuhao/yulan_3_base"  # 替换为你的模型路径
PROMPT_TEXT="你好！请自我介绍一下。"       # 替换为你想输入的提示文本
MAX_LENGTH=128                              # 生成的最大长度
DTYPE="bfloat16"                            # 数据类型 (float32 或 bfloat16)
CONVERT_FROM_TORCH=true                     # 是否从 PyTorch 转换模型

# 运行 Python 脚本
python yulan2paddle.py \
    --model_path "$MODEL_PATH" \
    --prompt_text "$PROMPT_TEXT" \
    --max_length "$MAX_LENGTH" \
    --dtype "$DTYPE" \
    --convert_from_torch "$CONVERT_FROM_TORCH"