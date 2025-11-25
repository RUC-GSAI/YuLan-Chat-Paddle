<div align="center">

# 🌸 YuLan-Chat-Paddle

**A Comprehensive Toolkit for YuLan-Chat Large Language Model Based on PaddlePaddle**

[![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-3.0-blue)](https://www.paddlepaddle.org.cn/)
[![Python](https://img.shields.io/badge/Python-3.12+-brightgreen)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-orange)](./LICENSE)


</div>

---

## 📖 Introduction

**YuLan-Chat-Paddle** is a comprehensive toolkit for large language models based on the PaddlePaddle deep learning framework. It provides a complete solution covering model conversion, pre-training, fine-tuning, and evaluation. This project focuses on adapting the YuLan-Chat model to the PaddlePaddle ecosystem, supporting various training strategies and mainstream evaluation benchmarks.

### ✨ Key Features

- 🔄 **Model Conversion**: Seamless conversion from PyTorch models to PaddlePaddle format
- 🚀 **Pre-training**: Complete pre-training pipeline with comprehensive configurations
- 🎯 **Flexible Fine-tuning**: Support for both full parameter fine-tuning (SFT) and parameter-efficient fine-tuning (LoRA)
- 📊 **Comprehensive Evaluation**: Integration with mainstream benchmarks including C-Eval, MMLU, and GSM8K
- ⚡ **Multi-GPU Support**: Single-GPU and multi-GPU training capabilities
- 🛠️ **Easy to Use**: One-click scripts with detailed documentation

---

## 📁 Project Structure

```
YuLan-Chat-Paddle/
├── transfer/              # Model conversion module
│   ├── yulan2paddle.py   # PyTorch → PaddlePaddle conversion script
│   ├── convert.sh        # Conversion execution script
│   └── README.md
│
├── pretrain/             # Pre-training module
│   ├── run_pretrain.py   # Pre-training main program
│   ├── run_pretrain.sh   # Pre-training execution script
│   ├── preprocess_data.py # Data preprocessing
│   ├── config.json       # Training configuration
│   ├── data/             # Pre-training data
│   └── README.md
│
├── finetune/             # Fine-tuning module
│   ├── run_finetune.py   # Fine-tuning main program
│   ├── run_finetune_sft_singleGPU.sh  # Single-GPU SFT fine-tuning
│   ├── run_finetune_sft_multiGPU.sh   # Multi-GPU SFT fine-tuning
│   ├── run_finetune_lora.sh           # LoRA fine-tuning
│   ├── data/             # Fine-tuning data
│   └── README.md
│
├── inference/            # Inference and evaluation module
│   ├── eval/             # Evaluation scripts library
│   │   ├── tasks/        # Various evaluation tasks
│   │   └── *.py          # Evaluation tools
│   ├── data/             # Evaluation datasets
│   │   ├── ceval/        # C-Eval Chinese evaluation dataset
│   │   ├── mmlu/         # MMLU multi-task evaluation dataset
│   │   └── gsm8k/        # GSM8K math reasoning dataset
│   ├── evaluate_*_hf.sh     # HuggingFace framework evaluation scripts
│   ├── evaluate_*_paddle.sh # PaddlePaddle framework evaluation scripts
│   └── README.md
│
├── LICENSE               # Project license
└── README.md            # Project documentation
```

---

## 🚀 Quick Start

### Environment Setup

#### 1. Create Virtual Environment

```bash
conda create -n paddle-yulan python=3.12
conda activate paddle-yulan
```

#### 2. Install PaddlePaddle

Choose the appropriate installation command based on your CUDA version:

```bash
# CUDA 11.8 - 12.5
python3 -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# CUDA 12.6+
python3 -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

#### 3. Install PaddleNLP

```bash
pip install --upgrade --pre paddlenlp
```

---

## 📚 Usage Guide

### 🔄 Model Conversion

Convert YuLan-Chat models from PyTorch format to PaddlePaddle format:

```bash
cd transfer
# Modify model paths in convert.sh
bash convert.sh
```

For detailed instructions, please refer to: [transfer/README.md](./transfer/README.md)

### 🚀 Pre-training

Train language models from scratch:

```bash
cd pretrain
# Prepare data in the data/ directory
# Modify config.json configuration file
bash run_pretrain.sh
```

For detailed instructions, please refer to: [pretrain/README.md](./pretrain/README.md)

### 🎯 Fine-tuning

#### Full Parameter Fine-tuning (SFT)

```bash
cd finetune
# Prepare training data in data/ directory (must contain src and tgt fields)

# Single-GPU training
bash run_finetune_sft_singleGPU.sh

# Multi-GPU training
bash run_finetune_sft_multiGPU.sh
```

#### LoRA Fine-tuning

```bash
cd finetune
bash run_finetune_lora.sh
```

For detailed instructions, please refer to: [finetune/README.md](./finetune/README.md)

### 📊 Model Evaluation

Evaluate model performance on multiple mainstream benchmarks:

```bash
cd inference
mkdir log

# C-Eval Chinese evaluation
bash evaluate_ceval_paddle.sh

# MMLU multi-task evaluation
bash evaluate_mmlu_paddle.sh

# GSM8K math reasoning evaluation
bash evaluate_gsm8k_paddle.sh
```

For detailed instructions, please refer to: [inference/README.md](./inference/README.md)

---

## 🎯 Supported Evaluation Benchmarks

| Benchmark | Description | Number of Tasks |
|-----------|-------------|-----------------|
| **C-Eval** | Comprehensive Chinese evaluation covering humanities, social sciences, STEM, and more | 52 subtasks |
| **MMLU** | Massive Multitask Language Understanding evaluation including math, history, law, etc. | 57 subtasks |
| **GSM8K** | Grade School Math dataset for testing mathematical reasoning capabilities | 8.5K problems |

---

## 📝 License

This project is licensed under the [Apache-2.0 License](./LICENSE).
