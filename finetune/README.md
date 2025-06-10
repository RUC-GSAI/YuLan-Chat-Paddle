# Paddle微调
本部分提供单脚本使用paddle框架进行微调的脚本。

# 环境配置
以使用conda为例。
Step1: 创建、激活虚拟环境
```
conda create -n "paddle-finetune" python=3.12
conda activate paddle-finetune
```

Step2: 安装paddle和paddlenlp环境
本finetune环境基于paddlepaddle3.0，假设您的设备有GPU.
---
paddle
```bash
# if CUDA version higher or equal 11.8 and lower than 12.6 
python3 -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ 
# if CUDA version equal or higher than 12.6 
python3 -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```
---
paddlenlp
```bash
pip install --upgrade --pre paddlenlp
```

# Paddle微调流程

- 数据准备：在`data/`下指定微调数据集。要求每个数据含有`"src"`和`tgt`字段
- 配置文件：在`config/`下指定参数配置文件。具体可参考https://paddlenlp.readthedocs.io/zh/latest/llm/docs/finetune.html#id8
- 配置脚本：按照需求修改并运行当前目录下的`*.sh`脚本。