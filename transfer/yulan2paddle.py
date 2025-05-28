import argparse
from paddlenlp.transformers import LlamaTokenizer, LlamaForCausalLM

def main(model_path, prompt_text, max_length, dtype, convert_from_torch):
    # 加载 tokenizer 和模型
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        convert_from_torch=convert_from_torch
    )

    # 格式化提示文本
    prompt = (
        "The following is a conversation between a human and an AI assistant namely YuLan, developed by GSAI, Renmin University of China. "
        "The AI assistant gives helpful, detailed, and polite answers to the user's questions.\n"
        "[|Human|]:{}\n[|AI|]:"
    )
    formatted_prompt = prompt.format(prompt_text)

    # Tokenize 输入
    input_features = tokenizer(formatted_prompt, return_tensors="pd")

    # 生成输出
    outputs = model.generate(**input_features, max_length=max_length)

    # 解码并打印结果
    res = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
    print("Generated Response:")
    print(res)

if __name__ == "__main__":
    # 设置 argparse
    parser = argparse.ArgumentParser(description="Run Llama model inference with PaddleNLP.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--prompt_text", type=str, default="你好！请自我介绍一下。", help="Input prompt text for the model.")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum length of the generated sequence.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"], help="Data type for model weights.")
    parser.add_argument("--convert_from_torch", action="store_true", help="Whether to convert the model from PyTorch.")

    args = parser.parse_args()

    # 调用主函数
    main(
        model_path=args.model_path,
        prompt_text=args.prompt_text,
        max_length=args.max_length,
        dtype=args.dtype,
        convert_from_torch=args.convert_from_torch
    )