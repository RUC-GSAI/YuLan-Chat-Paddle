# 评测代码

This folder contains evaluation scripts for testing language models using both HuggingFace and PaddlePaddle frameworks.

## Supported Benchmarks

- CEVAL
- GSM8K 
- MMLU

## Directory Structure

```
.
├── data/                      # Evaluation datasets
├── eval/                      # Evaluation library
├── evaluate_ceval_hf.sh       # CEVAL evaluation script (HuggingFace)
├── evaluate_ceval_paddle.sh   # CEVAL evaluation script (PaddlePaddle) 
├── evaluate_gsm8k_hf.sh       # GSM8K evaluation script (HuggingFace)
├── evaluate_gsm8k_paddle.sh   # GSM8K evaluation script (PaddlePaddle)
├── evaluate_mmlu_hf.sh        # MMLU evaluation script (HuggingFace)
└── evaluate_mmlu_paddle.sh    # MMLU evaluation script (PaddlePaddle)
```

## Usage

```bash
mkdir log
```

### HuggingFace Evaluation
```bash
# Evaluate on CEVAL benchmark
bash evaluate_ceval_hf.sh

# Evaluate on GSM8K benchmark
bash evaluate_gsm8k_hf.sh

# Evaluate on MMLU benchmark
bash evaluate_mmlu_hf.sh
```

### PaddlePaddle Evaluation
```bash
# Evaluate on CEVAL benchmark
bash evaluate_ceval_paddle.sh

# Evaluate on GSM8K benchmark
bash evaluate_gsm8k_paddle.sh

# Evaluate on MMLU benchmark
bash evaluate_mmlu_paddle.sh
```

## Benchmarks Description

### CEVAL
CEVAL is a comprehensive Chinese evaluation suite for foundation models.

### GSM8K
GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically diverse grade school math word problems.

### MMLU
MMLU (Massive Multitask Language Understanding) is a benchmark that evaluates language models on 57 tasks including elementary mathematics, US history, computer science, law, and more.

## Requirements

- Python 3.7+
- HuggingFace Transformers
- PaddlePaddle
- Additional dependencies (specified in requirements.txt)


