import argparse
import random
import time
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    LlamaTokenizer,
    LlamaForCausalLM
)
import torch.nn.functional as F
import tasks
from tqdm import tqdm
import json
import os
import time
import datetime

"""
llm2 : Baichuan llama2
llm : Mistral YI ChatGLM
llm-qwen: Qwen

"""

def _load_causal_model(model_path, device, flash_attn_2=False):
    if flash_attn_2:
        attn_implementation = 'flash_attention_2'
    else:
        attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().bfloat16().to(device)

    return model

def _load_tokenizer(tok_path, name):
    if "llama" in name.lower() or "intern" in name.lower():
        tokenizer = LlamaTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    print("testing tokenizer...")
    print("hello, world: ", tokenizer.encode("hello\n\nhello"))
    print("你好，世界: ", tokenizer.encode("你好，世界"))
    print("bos_token_id: {}, eos_token_id: {}, pad_token_id: {}".format(
        tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id))
    if "chatglm" in name.lower():
        pass
    elif "qwen" in name.lower():
        pass
    else:
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = 0
            print("setting pad_token_id to {}, now pad_token is {}".format(0, tokenizer.decode(0)))
        if not tokenizer.bos_token_id:
            tokenizer.bos_token_id = 1
            print("setting bos_token_id to {}, now bos_token is {}".format(1, tokenizer.decode(1)))
        if not tokenizer.eos_token_id:
            tokenizer.eos_token_id = 2
            print("setting eos_token_id to {}, now eos_token is {}".format(2, tokenizer.decode(2)))
    tokenizer.padding_side = 'left'
    print("setting padding_side to left")
    print("vocab length: %d"%len(tokenizer))
    return tokenizer

def test_model_logits(model, tokenizer, prompt="长亭外，古道边，芳草碧连天。"):
    prompt = "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.\nQ: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\nQ: Leah had 32 chocolates and her sister had "
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    outputs = model(input_ids)
    
    seq_len, top_k = outputs.logits.size(1), 3
    top_k_scores, top_k_idxs = torch.topk(outputs.logits[0], k=top_k, dim=-1, largest=True, sorted=True) 
    # output_ids = torch.argmax(outputs.logits, dim=-1)
    for i in range(seq_len):
        preds = []
        for j in range(top_k):
            preds.append("({}, {})".format(tokenizer.decode(top_k_idxs[i][j]), top_k_scores[i][j]))
        print("{} --> {}".format(tokenizer.decode(input_ids[0, :i+1]), " ".join(preds)))

def test_model_generate(model, tokenizer, prompt="长亭外，古道边，芳草碧连天。"):
    # inputs = tokenizer(prompt,return_tensors='pt',padding="longest",max_length=2048,truncation=True,return_attention_mask=True,add_special_tokens=True)
    inputs = tokenizer(prompt,return_tensors='pt',max_length=2048,truncation=True,return_attention_mask=True,add_special_tokens=True)
    # kwargs = {'temperature': 0.8,'top_p': 0.95,"top_k": 50,"repetition_penalty": 1.1,"no_repeat_ngram_size": 64,"max_length": 2048,"pad_token_id": tokenizer.pad_token_id,"eos_token_id": tokenizer.eos_token_id}
    kwargs = {'temperature': 0.8,'top_p': 0.95,"top_k": 50,"max_length": 2048,"pad_token_id": tokenizer.pad_token_id,"eos_token_id": tokenizer.eos_token_id}
    outputs = model.generate(inputs['input_ids'].to(model.device), do_sample=False,**kwargs)
    out_txt = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response = out_txt[len(prompt):]
    print(response)
    
def load_model_and_tokenizer(model_config):
    device = torch.device("cuda:0")
    if model_config["path"] == "default":
        model_path = "/fs/archive/share/u2022000139/hf_models/%s" % model_config["name"]
    else:
        model_path = model_config["path"]
    print("loading model and tokenizer from <%s>"%model_path)
    
    tokenizer = _load_tokenizer(model_path, model_config["name"])
    model = _load_causal_model(model_path, device, flash_attn_2=model_config["flash_attn_2"])
    # import pdb;pdb.set_trace()
    # test_model_logits(model, tokenizer)
    # test_model_generate(model, tokenizer)
    return model, tokenizer

@torch.no_grad()
def loglikelihood(requests, model, tokenizer, max_len, config, disable_tqdm=False):
    """
    requests -> new_reqs -> chunk, cache_res -> results

    step 1: sort, from long to short
    step 2: split, chunksmax_seq_length
    step 3: padding
    step 4: call model
    """
    add_special_tokens = config['add_special_tokens']
    batch_size = config['batch_size']

    new_reqs = []
    for request in requests:
        for sub_req in request:
            context, continuation = sub_req['context'], sub_req['continuation']

            context_enc = tokenizer.encode(context, add_special_tokens=add_special_tokens)
            continuation_enc = tokenizer.encode(context + continuation, add_special_tokens=add_special_tokens)[len(context_enc):]

            key = (context, continuation)
            value = (context_enc, continuation_enc)
            length = - (len(context_enc) + len(continuation_enc))
            
            new_reqs.append((key, value, length))

    # step 1: sort
    new_reqs.sort(key=lambda x: x[2])
    print("get %d sub_request, max len: %d, min len %d"%(len(new_reqs), -new_reqs[0][2], -new_reqs[-1][2]))

    # step 2: batch
    cache_res = {}
    for i in tqdm(range(0, len(new_reqs), batch_size), disable=disable_tqdm):
        chunk = new_reqs[i:i + batch_size]
        input_texts = [x[0][0] + x[0][1] for x in chunk]

        # step 3: padding
        inputs = tokenizer(
            input_texts,
            return_tensors='pt',
            padding="longest", # if tokenizer.pad_token_id else False,
            max_length=max_len,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=add_special_tokens,
        )
        # step 4: call model
        # logits = model(inputs['input_ids'].to(model.device))
        logits = model(inputs['input_ids'].to(model.device), 
                       attention_mask=inputs['attention_mask'].to(model.device))

        # step 5: get answer
        logits = F.log_softmax(logits[0], dim=-1).cpu()  # [batch, padding_length, vocab]
        
        for one_req, one_logits in zip(chunk, logits):
            key, value, _ = one_req
            _, cont_toks = value

            # Slice to original seq length
            contlen = len(cont_toks)
            one_logits = one_logits[-contlen-1 : -1].unsqueeze(0)  # [1, seq, vocab]

            # Check if per-token argmax is exactly equal to continuation
            greedy_tokens = one_logits.argmax(dim=-1)
            cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)  # [1, seq]
            if len(greedy_tokens[0]) != len(cont_toks[0]):
                # 超出最大长度限制
                answer = (-float('inf'), False, contlen)
            else:
                max_equal = (greedy_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                one_logits = torch.gather(one_logits,  dim=2, index=cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = (float(one_logits.sum()), bool(max_equal), contlen)

            cache_res[key] = answer

    results = []
    for request in requests:
        result = []
        for sub_req in request:
            context, continuation = sub_req['context'], sub_req['continuation']
            key = (context, continuation)
            answer = cache_res[key]
            result.append(answer)
        results.append(result)
    print("get %d sub_result, and will back %d result"%(len(cache_res), len(results)))
    return results

@torch.no_grad()
def fast_loglikelihood(requests, model, tokenizer, max_len, config, disable_tqdm=False):
    """
    requests -> new_reqs -> chunk, cache_res -> results
    step 1: sort, from long to short
    step 2: split, chunks
    step 3: padding
    step 4: call model
    """
    add_special_tokens = config['add_special_tokens']
    batch_size = config['batch_size']
    batch_size = 4

    new_reqs = []
    for request in requests:
        context, continuations = request['context'], request['continuations']

        context_enc = tokenizer.encode(context, add_special_tokens=add_special_tokens)
        
        continuation_encs = []
        for continuation in continuations:
            continuation_enc = tokenizer.encode(context + continuation, add_special_tokens=add_special_tokens)[len(context_enc):]
            assert len(continuation_enc) == 1
            continuation_encs.append(continuation_enc)

        key = context
        value = (context_enc, continuation_encs)
        length = -len(context_enc)
        new_reqs.append((key, value, length))

    # step 1: sort
    new_reqs.sort(key=lambda x: x[2])
    print("get %d sub_request, max len: %d, min len %d"%(len(new_reqs), -new_reqs[0][2], -new_reqs[-1][2]))
    # step 2: batch
    cache_res = {}
    for i in tqdm(range(0, len(new_reqs), batch_size), disable=disable_tqdm):
        chunk = new_reqs[i:i + batch_size]
        input_texts = [x[0] for x in chunk]

        # step 3: padding
        inputs = tokenizer(
            input_texts,
            return_tensors='pt',
            padding="longest", #if tokenizer.pad_token_id else False,
            max_length=max_len,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=add_special_tokens,
        )
        # step 4: call model
        logits = model(inputs['input_ids'].to(model.device))
        # logits = model(inputs['input_ids'].to(model.device), 
        #                attention_mask=inputs['attention_mask'].to(model.device))

        # step 5: get answer
        logits = F.log_softmax(logits[0], dim=-1).cpu()  # [batch, padding_length, vocab]

        
        for one_req, one_logits in zip(chunk, logits):
            key, value, _ = one_req
            _, multi_cont_toks = value 
            one_logits = one_logits[-1]  # [vocab]

            answers = []
            greedy_token = one_logits.argmax(dim=-1)
            for cont_tok in multi_cont_toks:
                # cont_tok = torch.tensor(cont_tok, dtype=torch.long)  # [1]
                one_logit = one_logits[cont_tok]
                if greedy_token == torch.tensor(cont_tok):
                    answer = (float(one_logit), True, 1)
                else:
                    answer = (float(one_logit), False, 1)
                answers.append(answer)

            cache_res[key] = answers

    results = []
    for request in requests:
        key = request['context']
        answers = cache_res[key]
        results.append(answers)

    print("get %d result, and will back %d result"%(len(cache_res), len(results)))
    return results

@torch.no_grad()
def greedy_until(requests, model, tokenizer, max_len, config, disable_tqdm=False):
    """
    requests -> new_reqs -> chunk, cache_res -> results
    step 0: unpack config
    step 1: sort, from long to short
    step 2: split, chunks
    step 3: padding
    step 4: call model
    """
    # step 0
    batch_size = config['batch_size']
    max_gen_len = config['max_gen_len']
    end_token = config['end_token']
    is_sample = config['is_sample']

    new_reqs = []
    for request in requests:
        context = request['context']
        context_enc = tokenizer.encode(context, add_special_tokens=False)

        key = context
        clean_key = tokenizer.decode(tokenizer.encode(context, add_special_tokens=False))
        length = -len(context_enc)
        new_reqs.append((key, clean_key, length))

    # step 1: sort
    new_reqs.sort(key=lambda x: x[-1])
    print("get %d request, max len: %d, min len %d"%(len(new_reqs), -new_reqs[0][-1], -new_reqs[-1][-1]))

    # step 2: batch
    cache_res = {}
    for i in tqdm(range(0, len(new_reqs), batch_size), disable=disable_tqdm):
        chunk = new_reqs[i:i + batch_size]
        input_texts = [x[0] for x in chunk]
        clean_keys = [x[1] for x in chunk]

        # step 3: padding
        inputs = tokenizer(
            input_texts,
            return_tensors='pt',
            padding="longest", # if tokenizer.pad_token_id else False,
            max_length=max_len,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True
        )
        # step 4: call model
        kwargs = {
            # 'temperature': 0.8,
            # 'top_p': 0.95,
            # "top_k": 50,
            # "repetition_penalty": 1.1,
            # "no_repeat_ngram_size": 64,
            # "max_length": min(inputs['input_ids'].size(1) + max_gen_len, max_len),
            "max_new_tokens": min(max_gen_len, max_len - inputs['input_ids'].size(1)),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.encode(end_token) if end_token else tokenizer.eos_token_id,
            "do_sample": is_sample,
        }
        outputs = model.generate(
            inputs['input_ids'].to(model.device),
            # attention_mask=inputs['attention_mask'].to(model.device),
            **kwargs
            )
        
        
        # step 5: get answer
        output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for in_txt, key_txt, out_txt in zip(input_texts, clean_keys, output_texts):
            # response = out_txt[len(key_txt):]
            response = out_txt.split(key_txt)[-1]
            cache_res[in_txt] = response

    results = []
    for request in requests:
        context = request['context']
        response = cache_res[context]
        results.append(response)
    print("get %d result, and will back %d result"%(len(cache_res), len(results)))

    return results

def run_model_call(requests, model, tokenizer, max_len, config):
    disable_tqdm = False
    # process loglikelihood requests
    if "loglikelihood" in requests.keys():
        print("Running loglikelihood requests")
        results = loglikelihood(requests['loglikelihood'], model, tokenizer, max_len, config, disable_tqdm)

    # process fast loglikelihood requests (the length of continuation is 1)
    elif "fast_loglikelihood" in requests.keys():
        print("Running fast loglikelihood requests")
        results = fast_loglikelihood(requests['fast_loglikelihood'], model, tokenizer, max_len, config, disable_tqdm)
    
    # process greedy_until requests
    elif 'greedy_until' in requests.keys():
        print("Running greedy_until requests")
        results = greedy_until(requests['greedy_until'], model, tokenizer, max_len, config, disable_tqdm)
    
    else:
        results = None

    return results

def run_one_model(tasks_list, model_config):
    model_name = model_config["name"]
    print("testing model <%s>"%model_name)
    # load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    # load dataset
    task_dict = {}
    for task_name in tasks_list:
        if "c_eval" in task_name or "mmlu" in task_name or "agi" in task_name:
           task_dict[task_name]  = tasks.TASK_REGISTRY[task_name](tokenizer, model_config["max_seq_length"])
        elif task_name in ["cmrc2018", "coqa", "race_middle", "race_high"]:
            task_dict[task_name]  = tasks.TASK_REGISTRY[task_name](tokenizer, model_config["max_seq_length"])
        else:
            task_dict[task_name]  = tasks.TASK_REGISTRY[task_name]()

    # config for tasks
    # batch, max_gen_len, end_token, etc.
    task_config = {}
    for task_name, task in task_dict.items():
        task_config[task_name] = task.get_config()
        # dynamic batch 
        task_config[task_name]['batch_size'] //= model_config["size_level"]

    # get requests
    task_requests = {}
    for task_name, task in task_dict.items():
        task_requests[task_name], random_score = task.construct_requests()
        print("build <%s> requests success, random score is %f"%(task_name, random_score))
    # test
    all_metrics = {}
    for task_name, task in task_dict.items():
        print("test: %s"%task_name)
        start = time.time()
        results = run_model_call(
            task_requests[task_name], 
            model, 
            tokenizer, 
            model_config["max_seq_length"],
            task_config[task_name])
        metrics, logs = task.process_results(results)
        print(metrics)
        print("time use (s): %.0f"%(time.time()-start))   

        # save metrics
        for k in metrics.keys():
            all_metrics[task_name+"_"+k] = metrics[k]

        # write to file for logs
        output_dir = "./logs-torch/%s"%task_name
        output_path = "./logs-torch/%s/%s.json"%(task_name, model_name)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding="utf8") as f:
            f.write(json.dumps(logs, indent=4, ensure_ascii=False) + '\n')

        # write to file for eval
        if task_name == "humaneval":
            eval_path = "./logs/%s/%s-eval.json"%(task_name, model_name)
            with open(eval_path, 'w', encoding="utf8") as f:
                for log in logs:
                    line = {
                        "task_id": log['task_id'],
                        "completion": log['pred']
                    }
                    f.write(json.dumps(line) + '\n')

        if "c_eval" in task_name:
            eval_path = "./logs/%s/%s-eval.json"%(task_name, model_name)
            results = {}
            for log in logs:
                if log["data_type"] not in results:
                    results[log["data_type"]] = {}
                results[log["data_type"]][str(log["data_idx"])] = log["answer"]
            with open(eval_path, 'w', encoding="utf8") as f:
                f.write(json.dumps(results, indent=4, ensure_ascii=False))
    
    # 释放显存
    del model
    torch.cuda.empty_cache()
    time.sleep(5)

    return all_metrics


def get_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--model', type=str)
    parser.add_argument('--flash_attn', default=False, action='store_true')
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tasks_list = [
        # "boolq", "piqa", "hellaswag", "winogrande", "wsc273", "arc_easy", "arc_challenge", "openbookqa", 
        # "commonsenseqa", 
        "gsm8k", 
        # "aqua",
        # "trivia_qa",
        # "race_middle", "race_high", "coqa", "cmrc2018", "c3_dialog", "c3_mixed",
        # "mmlu",
        # "c_eval",
        # "agi_gaokao_multichoice", "agi_gaokao_cloze_direct",
        ]
    
    # tasks_list = ["knowledge_prompt"]
    args = get_args()
    model_list = [
        {
            "name": "new_model",
            "path": args.model,
            "load_type": "causal",
            "flash_attn_2": args.flash_attn,
            "size_level": 1, #(<7B)->1, 13B->2, 30B->4, 65B->8
            "max_seq_length": 2048, # 2K, 8K, 32K
        },
        # {
        #     "name": "yulan_chat",
        #     "path": "/mnt/wangyuhao/yulan3-ckpt",
        #     "load_type": "causal",
        #     "flash_attn_2": True,
        #     "size_level": 1, #(<7B)->1, 13B->2, 30B->4, 65B->8
        #     "max_seq_length": 2048, # 2K, 8K, 32K
        # },
        # {
        #     "name": "yulan_base",
        #     "path": "/media/wangyuhao/yulan_3_base/",
        #     "load_type": "causal",
        #     "flash_attn_2": True,
        #     "size_level": 1, #(<7B)->1, 13B->2, 30B->4, 65B->8
        #     "max_seq_length": 2048, # 2K, 8K, 32K
        # },
    ]


    all_metrics = []
    for model_config in model_list:
        metrics = run_one_model(tasks_list, model_config)
        all_metrics.append(metrics)

    # write to file
    time_tag = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    with open("x%s.csv"%(time_tag), "w", encoding='utf8') as f:
        for model in model_list:
            f.write(",%s"%model["name"])
        f.write(",\n")
        keys = list(all_metrics[0].keys())
        for k in keys:
            if "c_eval" in k or "humaneval" in k:
                continue
            f.write(k + ",")
            for metrics in all_metrics:
                f.write("%.2f,"%(metrics[k] * 100))
            f.write("\n")