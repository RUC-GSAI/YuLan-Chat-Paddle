import argparse
import os
import torch
import paddle
import numpy as np
import pandas as pd
from paddle.nn import CrossEntropyLoss
from categories import subcategories, categories
from transformers import AutoTokenizer, AutoModelForCausalLM
from paddlenlp.transformers import AutoTokenizer as PaddleTokenizer
from paddlenlp.transformers import AutoModelForCausalLM as PaddleModelForCausalLM
import time
from tqdm import tqdm
from torch.nn.functional import cross_entropy
from paddle.nn.functional import cross_entropy as pcross_entropy
choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(args, subject, models, tokenizers, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]
    model, pmodel = models
    tokenizer, ptokenizer = tokenizers
    time_cost = []
    ppls = []

    for i in tqdm(range(test_df.shape[0])):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        inputs = tokenizer(prompt, return_tensors="pt")

        while inputs.input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            inputs = tokenizer(prompt, return_tensors="pt")
        pinputs = ptokenizer(prompt, return_tensors="pd")
        
        pinputs = {k: v for k, v in pinputs.items()}
        pinputs["labels"] = pinputs["input_ids"].clone()
        pinputs["return_dict"] = True
        inputs = {k: v.to(1) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        label = test_df.iloc[i, test_df.shape[1] - 1]
        
        start_time = time.time()
        outputs = model(**inputs)
        poutputs = pmodel(**pinputs)
        end_time = time.time()
        elapsed_time = end_time - start_time  # 计算耗时
        time_cost.append(elapsed_time)
        
        logits = outputs.logits[:,-1].flatten()
        logits = logits.to(torch.float32)
        
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        loss = cross_entropy(shift_logits.to(torch.float32), shift_labels)
        
        pshift_logits = poutputs.logits[..., :-1, :].contiguous()
        pshift_labels = pinputs["labels"][..., 1:].contiguous()
        pshift_logits = pshift_logits.reshape([-1, pmodel.config.vocab_size])
        pshift_labels = pshift_labels.flatten()
        ploss = pcross_entropy(pshift_logits.astype("float32"), pshift_labels).to(paddle.float32)
        
        cor = torch.tensor(ploss.item()).to(1) - loss
        # cor = cor / loss
        cor = torch.abs(cor).item()
        
        plogits = poutputs.logits[:,-1].flatten()
        plogits = torch.tensor(plogits.to(paddle.float32).numpy()).to(1)      
        
        ppl = (logits - plogits)
        # ppl = ppl / logits
        ppl = torch.abs(ppl).mean().item()
        
        cors.append(cor)
        ppls.append(ppl)
        all_probs.append([0, 0, 0, 0])

    acc = np.mean(cors)
    cors = np.array(cors)
    ppls = np.array(ppls)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs, np.array(time_cost), ppls


def main(args):
    
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              use_fast=True,
                                              model_max_length=4096,
                                              padding_side="right",
                                              trust_remote_code=True)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.model, 
                                                 torch_dtype=dtype,
                                                 low_cpu_mem_usage=True,
                                                 attn_implementation='flash_attention_2', 
                                                 trust_remote_code=True).to(1)
    model = model.eval()
    ptokenizer = PaddleTokenizer.from_pretrained(args.model, 
                                                 use_fast=True, 
                                                 model_max_length=4096, 
                                                 padding_side="right", 
                                                 trust_remote_code=True)
    pmodel = PaddleModelForCausalLM.from_pretrained(args.model, 
                                                    dtype=args.dtype,
                                                    use_flash_attention=True,
                                                    # low_cpu_mem_usage=True,
                                                    )
    pmodel.eval()
    args.model = f'compare_{args.dtype}'
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.model))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.model)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}
    
    all_times = []
    subcat_times = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_times = {cat: [] for cat in categories}
    
    all_ppls = []
    subcat_ppls = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_ppls = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs, times, ppls = eval(args, subject, (model, pmodel), (tokenizer, ptokenizer), dev_df, test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)
        
        for subcat in subcats:
            subcat_times[subcat].append(times)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_times[key].append(times)
        all_times.append(times)
        
        for subcat in subcats:
            subcat_ppls[subcat].append(ppls)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_ppls[key].append(ppls)
        all_ppls.append(ppls)

        test_df["{}_correct".format(args.model)] = cors
        test_df["{}_time".format(args.model)] = times
        test_df["{}_eps".format(args.model)] = ppls
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "results_{}".format(args.model), "{}.csv".format(subject)
            ),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        subcat_time = np.mean(np.concatenate(subcat_times[subcat]))
        subcat_ppl = np.mean(np.concatenate(subcat_ppls[subcat]))
        print("Average accuracy {:.2e} - {}".format(subcat_acc, subcat))
        print("Average time {:.2f} - {}".format(subcat_time, subcat))
        print("Average eps {:.2e} - {}".format(subcat_ppl, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        cat_time = np.mean(np.concatenate(cat_times[cat]))
        cat_ppl = np.mean(np.concatenate(cat_ppls[cat]))
        print("Average accuracy {:.2e} - {}".format(cat_acc, cat))
        print("Average time {:.2f} - {}".format(cat_time, cat))
        print("Average eps {:.2e} - {}".format(cat_ppl, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    weighted_time = np.mean(np.concatenate(all_times))
    weighted_ppl = np.mean(np.concatenate(all_ppls))
    print("Average accuracy: {:.2e}".format(weighted_acc))
    print("Average time {:.2f}".format(weighted_time))
    print("Average eps {:.2e}".format(weighted_ppl))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--ngpu", "-g", type=int, default=1)
    parser.add_argument("--data_dir", "-d", type=str, default="/home/wangyuhao/yulan3/data/mmlu")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="/mnt/wangyuhao/yulan3-ckpt",
    )
    args = parser.parse_args()
    main(args)