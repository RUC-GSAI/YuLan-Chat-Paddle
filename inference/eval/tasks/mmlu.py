import re
import glob
import csv
import numpy as np

DATA_MAP = {
    'abstract_algebra': "STEM", 
    'anatomy': "STEM", 
    'astronomy': "STEM", 
    'business_ethics': "Other", 
    'clinical_knowledge': "Other", 
    'college_biology': "STEM", 
    'college_chemistry': "STEM", 
    'college_computer_science': "STEM", 
    'college_mathematics': "STEM", 
    'college_medicine': "Other", 
    'college_physics': "STEM", 
    'computer_security': "STEM", 
    'conceptual_physics': "STEM", 
    'econometrics': "Social Sciences", 
    'electrical_engineering': "STEM", 
    'elementary_mathematics': "STEM", 
    'formal_logic': "Humanities", 
    'global_facts': "Other", 
    'high_school_biology': "STEM", 
    'high_school_chemistry': "STEM", 
    'high_school_computer_science': "STEM", 
    'high_school_european_history': "Humanities", 
    'high_school_geography': "Social Sciences", 
    'high_school_government_and_politics': "Social Sciences", 
    'high_school_macroeconomics': "Social Sciences", 
    'high_school_mathematics': "STEM", 
    'high_school_microeconomics': "Social Sciences", 
    'high_school_physics': "STEM", 
    'high_school_psychology': "Social Sciences", 
    'high_school_statistics': "STEM", 
    'high_school_us_history': "Humanities", 
    'high_school_world_history': "Humanities", 
    'human_aging': "Other", 
    'human_sexuality': "Social Sciences", 
    'international_law': "Humanities", 
    'jurisprudence': "Humanities", 
    'logical_fallacies': "Humanities", 
    'machine_learning': "STEM", 
    'management': "Other", 
    'marketing': "Other", 
    'medical_genetics': "Other", 
    'miscellaneous': "Other", 
    'moral_disputes': "Humanities", 
    'moral_scenarios': "Humanities", 
    'nutrition': "Other", 
    'philosophy': "Humanities", 
    'prehistory': "Humanities", 
    'professional_accounting': "Other", 
    'professional_law': "Humanities", 
    'professional_medicine': "Other", 
    'professional_psychology': "Social Sciences", 
    'public_relations': "Social Sciences", 
    'security_studies': "Social Sciences", 
    'sociology': "Social Sciences", 
    'us_foreign_policy': "Social Sciences", 
    'virology': "Other", 
    'world_religions': "Humanities"
}

class MMLU:
    def __init__(self, tokenizer, max_len):
        self.tmp = self.load_data("/data/czp/YuLan-Chat-Paddle-main/inference/data/mmlu/test")
        self.data_num = {"STEM":0, "Social Sciences":0, "Humanities":0, "Other":0}
    
        # select data
        self.data = {}
        for k, v in self.tmp.items():
            self.data_num[DATA_MAP[k]] += len(v)
            self.data[k] = self.tmp[k]

        self.examples = self.load_data("/data/czp/YuLan-Chat-Paddle-main/inference/data/mmlu/dev")
        self.logs = []
        self.n_shot = 5

        self.tokenizer = tokenizer
        self.max_len = max_len

    def load_data(self, path):
        data = {}
        for path in sorted(glob.glob(path + "/*.csv")):
            sub_data_name = path.split('/')[-1]
            sub_data_name = "_".join(sub_data_name.split("_")[:-1])
            sub_data = []
            with open(path, 'r', encoding='utf8') as f:
                for line in csv.reader(f, skipinitialspace=True):
                    sub_data.append({
                        "question": line[0],
                        "options": line[1:5],
                        "label": line[5]
                    })
            data[sub_data_name] = sub_data
        return data
    
    def get_config(self):
        return {
            "batch_size": 8,
            "add_special_tokens": True,
        }
    
    def softmax(self, inp):   
        inp = np.array(inp) - np.max(inp)   
        return np.exp(inp) / np.sum(np.exp(inp))
    
    def format_prompt(self, data_sample, include_answer=True):
        prompt = "Question: " 
        prompt += data_sample["question"]
        choice = '\nChoices: '
        for x, c in zip(["A", "B", "C", "D"], data_sample['options']):
            choice += "\n{}. {}".format(x, c)
        prompt += choice
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(data_sample['label'])
        return prompt
    
    def gen_prompt(self, sub_data_name, k):
        format_data_name = sub_data_name.replace("_", " ")
        mmlu_prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_data_name)
        for i in range(k):
            mmlu_prompt += self.format_prompt(self.examples[sub_data_name][i])
        return mmlu_prompt

    def construct_requests(self):
        req_log = []
        random_score = 0.0

        for sub_data_name, sub_data in self.data.items(): 
            for idx, data_sample in enumerate(sub_data):
                k = self.n_shot
                prompt = self.gen_prompt(sub_data_name, k) + self.format_prompt(data_sample, include_answer=False)
                while len(self.tokenizer.encode(prompt)) >= self.max_len:
                    k -= 1
                    prompt = self.gen_prompt(sub_data_name, k) + self.format_prompt(data_sample, include_answer=False)

                request = []
                for c in ["A", "B", "C", "D"]:
                    request.append({
                        "context": prompt,
                        "continuation": " " + c
                    })
                req_log.append(request)
                random_score += 1.0 / len(data_sample['options'])

                # logs 
                self.logs.append({
                    "data_type": sub_data_name,
                    "data_idx": idx,
                    "prompt": prompt,
                    "continuations": [x['continuation'] for x in request],
                })
        
        random_score = random_score / len(req_log)
        return {
            "loglikelihood": req_log,
        }, random_score
    
    def process_results(self, results):
        assert len(results) == len(self.logs)

        labels = []
        for _, sub_data in self.data.items():
            for data_sample in sub_data:
                labels.append(data_sample['label'])
        assert len(results) == len(labels)

        label_dict = {"A":0, "B":1, "C":2, "D":3}
        
        acc_stem, acc_social, acc_human, acc_other = 0.0, 0.0, 0.0, 0.0
        acc_dict = {x:0 for x in self.data.keys()}
        for pred, label, log in zip(results, labels, self.logs):
            log['response'] = ",".join([str(x) for x in pred])
            is_right = False

            pred = [x[0] for x in pred]
            pred = self.softmax(pred)
            label = label_dict[label]

            if np.argmax(pred) == label:
                if DATA_MAP[log['data_type']] == "STEM":
                    acc_stem += 1.0
                elif DATA_MAP[log['data_type']] == "Social Sciences":
                    acc_social += 1.0
                elif DATA_MAP[log['data_type']] == "Humanities":
                    acc_human += 1.0
                else:
                    acc_other += 1.0
                is_right = True
                acc_dict[log['data_type']] += 1.0

            # logs 
            log['label'] = label
            log['pred'] = ",".join(["%.2f"%x for x in pred]) 
            log["is_right"] = is_right

        if self.data_num["STEM"] > 0:
            acc_stem = acc_stem / self.data_num["STEM"]
        if self.data_num["Social Sciences"] > 0:
            acc_social = acc_social / self.data_num["Social Sciences"]
        if self.data_num["Humanities"] > 0:
            acc_human = acc_human / self.data_num["Humanities"]
        if self.data_num["Other"] > 0:
            acc_other = acc_other / self.data_num["Other"]
        for k in acc_dict.keys():
            acc_dict[k] = acc_dict[k] / len(self.data[k])
        acc = (acc_stem + acc_social + acc_human + acc_other) / 4
        
        acc_dict["acc_stem"] = acc_stem
        acc_dict["acc_social"] = acc_social
        acc_dict["acc_human"] = acc_human
        acc_dict["acc_other"] = acc_other
        acc_dict["acc"] = acc
        
        return acc_dict, self.logs

class MMLU_fast(MMLU):
    def __init__(self, tokenizer, max_len):
        super().__init__(tokenizer, max_len)

    def construct_requests(self):
        req_log = []
        random_score = 0.0

        for sub_data_name, sub_data in self.data.items(): 
            for idx, data_sample in enumerate(sub_data):
                k = self.n_shot
                prompt = self.gen_prompt(sub_data_name, k) + self.format_prompt(data_sample, include_answer=False)
                while len(self.tokenizer.encode(prompt)) >= self.max_len:
                    k -= 1
                    if k == -1:
                        break
                    prompt = self.gen_prompt(sub_data_name, k) + self.format_prompt(data_sample, include_answer=False)

                request = {
                    "context": prompt,
                    "continuations": [" A", " B", " C", " D"],
                }

                req_log.append(request)
                random_score += 1.0 / len(data_sample['options'])

                # logs 
                self.logs.append({
                    "data_type": sub_data_name,
                    "data_idx": idx,
                    "prompt": prompt,
                    "continuations": request['continuations']
                })
        
        random_score = random_score / len(req_log)
        return {
            "fast_loglikelihood": req_log,
        }, random_score

class MMLU_other(MMLU_fast):
    def __init__(self, tokenizer, max_len):
        super().__init__(tokenizer, max_len)
        self.data_num = {"STEM":0, "Social Sciences":0, "Humanities":0, "Other":0}
        
        # select data
        select = "Other"
        self.data = {}
        for k, v in self.tmp.items():
            if DATA_MAP[k] != select:
                pass
            else:
                self.data_num[DATA_MAP[k]] += len(v)
                self.data[k] = self.tmp[k]

class MMLU_human(MMLU_fast):
    def __init__(self, tokenizer, max_len):
        super().__init__(tokenizer, max_len)
        self.data_num = {"STEM":0, "Social Sciences":0, "Humanities":0, "Other":0}
        
        # select data
        select = "Humanities"
        self.data = {}
        for k, v in self.tmp.items():
            if DATA_MAP[k] != select:
                pass
            else:
                self.data_num[DATA_MAP[k]] += len(v)
                self.data[k] = self.tmp[k]

class MMLU_social(MMLU_fast):
    def __init__(self, tokenizer, max_len):
        super().__init__(tokenizer, max_len)
        self.data_num = {"STEM":0, "Social Sciences":0, "Humanities":0, "Other":0}

        
        # select data
        select = "Social Sciences"
        self.data = {}
        for k, v in self.tmp.items():
            if DATA_MAP[k] != select:
                pass
            else:
                self.data_num[DATA_MAP[k]] += len(v)
                self.data[k] = self.tmp[k]

class MMLU_stem(MMLU_fast):
    def __init__(self, tokenizer, max_len):
        super().__init__(tokenizer, max_len)
        self.data_num = {"STEM":0, "Social Sciences":0, "Humanities":0, "Other":0}
        
        # select data
        select = "STEM"
        self.data = {}
        for k, v in self.tmp.items():
            if DATA_MAP[k] != select:
                pass
            else:
                self.data_num[DATA_MAP[k]] += len(v)
                self.data[k] = self.tmp[k]

class MMLU_ZeroShot_CoT_ChatGen(MMLU):
    def __init__(self, tokenizer, max_len):
        super().__init__(tokenizer, max_len)
        self.max_gen_len = 384
        self.max_len = max_len - self.max_gen_len
        
    def get_config(self):
        return {
            "batch_size": 8,
            "add_special_tokens": True,
            "max_gen_len": self.max_gen_len,
            "end_token": None,
            "is_sample": False,
        }
    
    def build_chat_prompt(self, data_sample):
        prompt = "Question: " 
        prompt += data_sample["question"]
        choice = '\nChoices: '
        for x, c in zip(["A", "B", "C", "D"], data_sample['options']):
            choice += "\n{}. {}".format(x, c)
        prompt += choice
        prompt += "\nLet's think step by step."
        return "The following is a conversation between a human and an AI assistant namely YuLan, developed by GSAI, Renmin University of China. The AI assistant gives helpful, detailed, and polite answers to the user's questions.\n[|Human|]:{}\n[|AI|]:".format(prompt)
    
    def construct_requests(self):
        requests = []

        for sub_data_name, sub_data in self.data.items(): 
            for idx, data_sample in enumerate(sub_data):
                prompt = self.build_chat_prompt(data_sample)
                    
                requests.append({
                    "context": prompt,
                })

                # logs 
                self.logs.append({
                    "data_type": sub_data_name,
                    "data_idx": idx,
                    "prompt": prompt,
                })
        return {
            "greedy_until": requests
        }, 0.0
    
    def process_results(self, results):
        assert len(results) == len(self.logs)

        labels = []
        for _, sub_data in self.data.items():
            for data_sample in sub_data:
                labels.append(data_sample['label'])
        assert len(results) == len(labels)
        
        acc_stem, acc_social, acc_human, acc_other = 0.0, 0.0, 0.0, 0.0
        acc_dict = {x:0 for x in self.data.keys()}
        for pred, label, log in zip(results, labels, self.logs):
            log['response'] = pred
            is_right = False
            pred = pred.strip().split('[|Human|]:')[0]

            if "the answer is" in pred:
                pred = pred.split("the answer is")[-1]
            elif "answer is" in pred:
                pred = pred.split("answer is")[-1]
            else:
                pred = pred

            ans = re.findall(r'A|B|C|D.', pred)
            if len(ans) == 0:
                ans = ""
            else:
                ans = ans[-1]
            
            if ans == label:
                is_right = True

            if is_right:
                if DATA_MAP[log['data_type']] == "STEM":
                    acc_stem += 1.0
                elif DATA_MAP[log['data_type']] == "Social Sciences":
                    acc_social += 1.0
                elif DATA_MAP[log['data_type']] == "Humanities":
                    acc_human += 1.0
                else:
                    acc_other += 1.0
                acc_dict[log['data_type']] += 1.0

            # logs 
            log['label'] = label
            log['pred'] = ans
            log["is_right"] = is_right

        if self.data_num["STEM"] > 0:
            acc_stem = acc_stem / self.data_num["STEM"]
        if self.data_num["Social Sciences"] > 0:
            acc_social = acc_social / self.data_num["Social Sciences"]
        if self.data_num["Humanities"] > 0:
            acc_human = acc_human / self.data_num["Humanities"]
        if self.data_num["Other"] > 0:
            acc_other = acc_other / self.data_num["Other"]
        for k in acc_dict.keys():
            acc_dict[k] = acc_dict[k] / len(self.data[k])
        acc = (acc_stem + acc_social + acc_human + acc_other) / 4
        
        acc_dict["acc_stem"] = acc_stem
        acc_dict["acc_social"] = acc_social
        acc_dict["acc_human"] = acc_human
        acc_dict["acc_other"] = acc_other
        acc_dict["acc"] = acc
        
        return acc_dict, self.logs