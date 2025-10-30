import argparse
import os
import random
import torch
import numpy as np
import pandas as pd
from categories import subcategories, categories
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from tqdm import tqdm
from torch.nn.functional import cross_entropy
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
def eval(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]
    
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
        
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        label = test_df.iloc[i, test_df.shape[1] - 1]
        
        start_time = time.time()
        outputs = model(**inputs)
        end_time = time.time()
        elapsed_time = end_time - start_time  # 计算耗时
        time_cost.append(elapsed_time)
        
        logits = outputs.logits[:,-1].flatten()
        logits = logits.to(torch.float32)        

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[-1]],
                        logits[tokenizer("B").input_ids[-1]],
                        logits[tokenizer("C").input_ids[-1]],
                        logits[tokenizer("D").input_ids[-1]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
        nll = cross_entropy(logits.view(-1, model.config.vocab_size), tokenizer(label, return_tensors="pt").input_ids[-1].cuda())
        ppl = torch.exp(nll).item()
        
        cor = pred == label
        cors.append(cor)
        ppls.append(ppl)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)
    ppls = np.array(ppls)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs, np.array(time_cost), ppls


def main(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,add_bos_token=False, model_max_length=4096,padding_side="right",trust_remote_code=True)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.model, 
                                                 torch_dtype=dtype, 
                                                 device_map="auto",
                                                 # attn_implementation='flash_attention_2',
                                                 # low_cpu_mem_usage=True,
                                                 trust_remote_code=True)
    model = model.eval()
    args.model = f'hf_model_{args.dtype}'
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

        cors, acc, probs, times, ppls = eval(args, subject, model, tokenizer, dev_df, test_df)
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
        test_df["{}_ppl".format(args.model)] = ppls
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
        print("Average accuracy {:.1f} - {}".format(subcat_acc * 100, subcat))
        # print("Average time {:.2f} - {}".format(subcat_time, subcat))
        # print("Average ppl {:.2f} - {}".format(subcat_ppl, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        cat_time = np.mean(np.concatenate(cat_times[cat]))
        cat_ppl = np.mean(np.concatenate(cat_ppls[cat]))
        print("Average accuracy {:.1f} - {}".format(cat_acc * 100, cat))
        # print("Average time {:.2f} - {}".format(cat_time, cat))
        # print("Average ppl {:.2f} - {}".format(cat_ppl, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    weighted_time = np.mean(np.concatenate(all_times))
    weighted_ppl = np.mean(np.concatenate(all_ppls))
    print("Average accuracy: {:.1f}".format(weighted_acc * 100))
    # print("Average time {:.2f}".format(weighted_time))
    # print("Average ppl {:.2f}".format(weighted_ppl))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--ngpu", "-g", type=int, default=1)
    parser.add_argument("--data_dir", "-d", type=str, default="./data/mmlu")
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
