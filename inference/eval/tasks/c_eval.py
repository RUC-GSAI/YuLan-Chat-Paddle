import re
import glob
import csv
import numpy as np

FORMAT_MAP = {
    'accountant': "注册会计师", 
    'advanced_mathematics': "高等数学", 
    'art_studies': "艺术学", 
    'basic_medicine': "基础医学", 
    'business_administration': "工商管理", 
    'chinese_language_and_literature': "中国语言文学", 
    'civil_servant': "公务员", 
    'clinical_medicine': "临床医学", 
    'college_chemistry': "大学化学", 
    'college_economics': "大学经济学", 
    'college_physics': "大学物理", 
    'college_programming': "大学编程", 
    'computer_architecture': "计算机组成", 
    'computer_network': "计算机网络", 
    'discrete_mathematics': "离散数学", 
    'education_science': "教育学", 
    'electrical_engineer': "注册电气工程师", 
    'environmental_impact_assessment_engineer': "环境影响评价工程师", 
    'fire_engineer': "注册消防工程师", 
    'high_school_biology': "高中生物", 
    'high_school_chemistry': "高中化学", 
    'high_school_chinese': "高中语文", 
    'high_school_geography': "高中地理", 
    'high_school_history': "高中历史", 
    'high_school_mathematics': "高中数学", 
    'high_school_physics': "高中物理", 
    'high_school_politics': "高中政治", 
    'ideological_and_moral_cultivation': "思想道德修养与法律基础", 
    'law': "法学", 
    'legal_professional': "法律职业资格", 
    'logic': "逻辑学", 
    'mao_zedong_thought': "毛泽东思想和中国特色社会主义理论体系概论", 
    'marxism': "马克思基本原理", 
    'metrology_engineer': "注册计量师", 
    'middle_school_biology': "初中生物", 
    'middle_school_chemistry': "初中化学", 
    'middle_school_geography': "初中地理", 
    'middle_school_history': "初中历史", 
    'middle_school_mathematics': "初中数学", 
    'middle_school_physics': "初中物理", 
    'middle_school_politics': "初中政治", 
    'modern_chinese_history': "中国近代史纲要", 
    'operating_system': "操作系统", 
    'physician': "医师资格", 
    'plant_protection': "植物保护", 
    'probability_and_statistics': "概率统计", 
    'professional_tour_guide': "导游资格", 
    'sports_science': "体育学", 
    'tax_accountant': "税务师", 
    'teacher_qualification': "教师资格", 
    'urban_and_rural_planner': "注册城乡规划师", 
    'veterinary_medicine': "兽医学"
}

DATA_MAP = {
    'accountant': "Other", 
    'advanced_mathematics': "STEM", 
    'art_studies': "Humanity", 
    'basic_medicine': "Other", 
    'business_administration': "Social Science", 
    'chinese_language_and_literature': "Humanity", 
    'civil_servant': "Other", 
    'clinical_medicine': "Other", 
    'college_chemistry': "STEM", 
    'college_economics': "Social Science", 
    'college_physics': "STEM", 
    'college_programming': "STEM", 
    'computer_architecture': "STEM", 
    'computer_network': "STEM", 
    'discrete_mathematics': "STEM", 
    'education_science': "Social Science", 
    'electrical_engineer': "STEM", 
    'environmental_impact_assessment_engineer': "Other", 
    'fire_engineer': "Other", 
    'high_school_biology': "STEM", 
    'high_school_chemistry': "STEM", 
    'high_school_chinese': "Humanity", 
    'high_school_geography': "Social Science", 
    'high_school_history': "Humanity", 
    'high_school_mathematics': "STEM", 
    'high_school_physics': "STEM", 
    'high_school_politics': "Social Science", 
    'ideological_and_moral_cultivation': "Humanity", 
    'law': "Humanity", 
    'legal_professional': "Humanity", 
    'logic': "Humanity", 
    'mao_zedong_thought': "Social Science", 
    'marxism': "Social Science", 
    'metrology_engineer': "STEM", 
    'middle_school_biology': "STEM", 
    'middle_school_chemistry': "STEM", 
    'middle_school_geography': "Social Science", 
    'middle_school_history': "Humanity", 
    'middle_school_mathematics': "STEM", 
    'middle_school_physics': "STEM", 
    'middle_school_politics': "Social Science", 
    'modern_chinese_history': "Humanity", 
    'operating_system': "STEM", 
    'physician': "Other", 
    'plant_protection': "Other", 
    'probability_and_statistics': "STEM", 
    'professional_tour_guide': "Humanity", 
    'sports_science': "Other", 
    'tax_accountant': "Other", 
    'teacher_qualification': "Social Science", 
    'urban_and_rural_planner': "Other", 
    'veterinary_medicine': "STEM"
}

class CEval:
    def __init__(self, tokenizer, max_len):
        self.data_path = "/fs/archive/share/u2022000139/zh_datasets/C-Eval/"
        # self.tmp = self.load_data("val")
        self.tmp = self.load_data("test")
    
        self.data_num = {"STEM":0, "Social Science":0, "Humanity":0, "Other":0}
        
        # select data
        self.data = {}
        for k, v in self.tmp.items():
            self.data_num[DATA_MAP[k]] += len(v)
            self.data[k] = self.tmp[k]

        self.examples = self.load_data("dev")
        self.logs = []
        self.n_shot = 5

        self.tokenizer = tokenizer
        self.max_len = max_len

    def load_data(self, split):
        data = {}
        for path in sorted(glob.glob(self.data_path + split + "/*.csv")):
            sub_data_name = path.split('/')[-1]
            sub_data_name = "_".join(sub_data_name.split("_")[:-1])
            sub_data = []
            is_first_line = True
            with open(path, 'r', encoding='utf8') as f:
                for line in csv.reader(f, skipinitialspace=True):
                    if is_first_line:
                        is_first_line = False
                        continue
                    sub_data.append({
                        "data_id": line[0],
                        "question": line[1],
                        "options": line[2:6],
                        "label": "E" if split == "test" else line[6],
                        "explain": line[7] if split == "dev" else None
                    })
            data[sub_data_name] = sub_data
        return data
    
    def get_config(self):
        return {
            "batch_size": 8,
            "add_special_tokens": True
        }
    
    def softmax(self, inp):   
        inp = np.array(inp) - np.max(inp)   
        return np.exp(inp) / np.sum(np.exp(inp))
    
    def format_prompt(self, data_sample, include_answer=True, is_CoT=False):
        prompt = "问题：" 
        prompt += data_sample["question"]
        choice = '\n选项：'
        for x, c in zip(["A", "B", "C", "D"], data_sample['options']):
            choice += "\n{}. {}".format(x, c)
        prompt += choice
        prompt += "\n答案："
        if include_answer:
            if is_CoT:
                prompt += " {}\n因此最终答案选{}。\n\n".format(data_sample["explain"], data_sample['label'])
            else:
                prompt += " {}\n\n".format(data_sample['label'])
        return prompt
    
    def build_prompt(self, sub_data_name, k, is_CoT=False):
        format_data_name = FORMAT_MAP[sub_data_name]
        mmlu_prompt = "下面是关于{}方向的选择测试题.\n\n".format(format_data_name)
        for i in range(k):
            mmlu_prompt += self.format_prompt(self.examples[sub_data_name][i], is_CoT=is_CoT)
        return mmlu_prompt

    def construct_requests(self):
        req_log = []
        random_score = 0.0

        for sub_data_name, sub_data in self.data.items(): 
            for idx, data_sample in enumerate(sub_data):
                k = self.n_shot
                prompt = self.build_prompt(sub_data_name, k) + self.format_prompt(data_sample, include_answer=False)
                while len(self.tokenizer.encode(prompt)) >= self.max_len:
                    k -= 1
                    prompt = self.build_prompt(sub_data_name, k) + self.format_prompt(data_sample, include_answer=False)
                
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
    
    def process_one_result(self, res, label):
        is_right = False
        response = ",".join([str(x) for x in res])
        pred = [x[0] for x in res]
        pred = self.softmax(pred)

        answer = ['A', 'B', 'C', 'D'][int(np.argmax(pred))]
        pred = ",".join(["%.4f"%x for x in pred]) 

        if answer == label:
            is_right = True
        return response, pred, answer, is_right
    
    def process_results(self, results):
        assert len(results) == len(self.logs)

        labels = []
        for _, sub_data in self.data.items():
            for data_sample in sub_data:
                labels.append(data_sample['label'])
        assert len(results) == len(labels)

        acc_stem, acc_social, acc_human, acc_other = 0.0, 0.0, 0.0, 0.0
        acc_dict = {x:0 for x in self.data.keys()}
        for res, label, log in zip(results, labels, self.logs):

            response, pred, answer, is_right = self.process_one_result(res, label)    

            if is_right:
                if DATA_MAP[log['data_type']] == "STEM":
                    acc_stem += 1.0
                elif DATA_MAP[log['data_type']] == "Social Science":
                    acc_social += 1.0
                elif DATA_MAP[log['data_type']] == "Humanity":
                    acc_human += 1.0
                else:
                    acc_other += 1.0
                acc_dict[log['data_type']] += 1.0

            # logs 
            log['response'] = response
            log['label'] = label
            log['pred'] = pred
            log['answer'] = answer
            log["is_right"] = is_right

        if self.data_num["STEM"] > 0:
            acc_stem = acc_stem / self.data_num["STEM"]
        if self.data_num["Social Science"] > 0:
            acc_social = acc_social / self.data_num["Social Science"]
        if self.data_num["Humanity"] > 0:
            acc_human = acc_human / self.data_num["Humanity"]
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
    
class CEval_fast(CEval):
    def __init__(self, tokenizer, max_len):
        super().__init__(tokenizer, max_len)

    def construct_requests(self):
        req_log = []

        for sub_data_name, sub_data in self.data.items(): 
            for idx, data_sample in enumerate(sub_data):
                k = self.n_shot
                prompt = self.build_prompt(sub_data_name, k) + self.format_prompt(data_sample, include_answer=False)
                while len(self.tokenizer.encode(prompt)) >= self.max_len:
                    k -= 1
                    if k == -1:
                        break
                    prompt = self.build_prompt(sub_data_name, k) + self.format_prompt(data_sample, include_answer=False)

                request = {
                    "context": prompt,
                    "continuations": [" A", " B", " C", " D"],
                }

                req_log.append(request)

                # logs 
                self.logs.append({
                    "data_type": sub_data_name,
                    "data_idx": idx,
                    "prompt": prompt,
                    "continuations": request['continuations']
                })
        
        return {
            "fast_loglikelihood": req_log,
        }, 0.25
    
class CEval_other(CEval_fast):
    def __init__(self, tokenizer, max_len):
        super().__init__(tokenizer, max_len)
        
        # select data
        select = "Other"
        self.data = {}
        for k, v in self.tmp.items():
            if DATA_MAP[k] != select:
                pass
            else:
                self.data_num[DATA_MAP[k]] += len(v)
                self.data[k] = self.tmp[k]

class CEval_human(CEval_fast):
    def __init__(self, tokenizer, max_len):
        super().__init__(tokenizer, max_len)
        
        # select data
        select = "Humanity"
        self.data = {}
        for k, v in self.tmp.items():
            if DATA_MAP[k] != select:
                pass
            else:
                self.data_num[DATA_MAP[k]] += len(v)
                self.data[k] = self.tmp[k]

class CEval_social(CEval_fast):
    def __init__(self, tokenizer, max_len):
        super().__init__(tokenizer, max_len)
        
        # select data
        select = "Social Science"
        self.data = {}
        for k, v in self.tmp.items():
            if DATA_MAP[k] != select:
                pass
            else:
                self.data_num[DATA_MAP[k]] += len(v)
                self.data[k] = self.tmp[k]

class CEval_stem(CEval_fast):
    def __init__(self, tokenizer, max_len):
        super().__init__(tokenizer, max_len)
        
        # select data
        select = "STEM"
        self.data = {}
        for k, v in self.tmp.items():
            if DATA_MAP[k] != select:
                pass
            else:
                self.data_num[DATA_MAP[k]] += len(v)
                self.data[k] = self.tmp[k]


class CEval_zeroshot(CEval_fast):
    def __init__(self, tokenizer, max_len):
        super().__init__(tokenizer, max_len)
        self.n_shot = 0


class CEval_CoT_Gen(CEval):
    def __init__(self, tokenizer, max_len):
        super().__init__(tokenizer, max_len)
        self.max_gen_len = 512
        self.max_len = max_len - self.max_gen_len
        self.n_shot = 0

    def get_config(self):
        return {
            "batch_size": 8,
            "add_special_tokens": True,
            "max_gen_len": self.max_gen_len,
            "end_token": None,
            "is_sample": False,
        }

    def build_prompt(self, sub_data_name, k, is_CoT=False):
        format_data_name = FORMAT_MAP[sub_data_name]
        mmlu_prompt = "下面是关于{}方向的选择测试题，请一步步思考并给出答案。\n\n".format(format_data_name)
        for i in range(k):
            mmlu_prompt += self.format_prompt(self.examples[sub_data_name][i], is_CoT=is_CoT)
        return mmlu_prompt
    
    def construct_requests(self):
        requests = []

        for sub_data_name, sub_data in self.data.items(): 
            for idx, data_sample in enumerate(sub_data):
                k = self.n_shot
                prompt = self.build_prompt(sub_data_name, k, is_CoT=True) + \
                    self.format_prompt(data_sample, include_answer=False)
                while len(self.tokenizer.encode(prompt)) >= self.max_len:
                    k -= 1
                    prompt = self.build_prompt(sub_data_name, k, is_CoT=True) + \
                        self.format_prompt(data_sample, include_answer=False)

                requests.append({
                    "context": prompt + "<思考过程>：",
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
    
    def process_one_result(self, res, label):
        is_right = False
        pred = res.strip().split('\n问题：')[0].strip()

        if "因此最终答案选" in pred:
            pred = pred.split("因此最终答案选")[-1]
        elif "最终答案" in pred:
            pred = pred.split("最终答案")[-1]
        else:
            pred = pred

        ans = re.findall(r'A|B|C|D', pred)
        if len(ans) == 0:
            ans = ""
        else:
            ans = ans[-1]
        
        if ans == label:
            is_right = True

        return res, pred, ans, is_right
