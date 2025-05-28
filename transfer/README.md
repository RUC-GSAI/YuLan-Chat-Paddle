# 模型转换说明

本仓库提供了一个完整的工具链，用于将 PyTorch 模型转换为 PaddlePaddle 格式。以下是具体操作步骤和脚本说明。

---

## 目录结构

```
.
├── convert.sh                # 将 PyTorch 模型转换为 PaddlePaddle 格式的脚本
├── yulan2paddle.py           # 将 PyTorch 模型转换为 PaddlePaddle 格式的代码
├── log/                      # 存储日志文件的目录
└── README.md                 # 当前文档
```

---

## 操作流程

#### 功能
- **`convert.sh`** 脚本用于将 PyTorch 格式的模型转换为 PaddlePaddle 格式。
- 转换完成后，生成的 PaddlePaddle 模型可以用于后续评测。

#### 使用方法
1. 确保已安装 `paddlenlp` 和其他依赖项。
2. 修改 `convert.sh` 中的模型路径：
   ```
   TORCH_MODEL_PATH=/path/to/pytorch_model
   PADDLE_MODEL_PATH=/path/to/save/paddle_model
   ```
3. 运行脚本：
   ```
   bash convert.sh
   ```

#### 输出
- 转换后的 PaddlePaddle 模型存储在指定路径（`PADDLE_MODEL_PATH`）。
