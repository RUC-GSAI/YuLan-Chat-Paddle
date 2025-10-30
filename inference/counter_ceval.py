import argparse
import json
import os
import random

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from tqdm import tqdm
import numpy as np
from datasets import load_dataset


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default='/media/wangyuhao/yulan_3_base/', help="model name or path"
    )
    parser.add_argument(
        "--shot", type=int, default=5, help="number of shot for few-shot learning"
    )
    parser.add_argument(
        "--split", type=str, default="val", help="split of dataset to evaluate"
    )
    parser.add_argument(
        "--output_dir", type=str, default="ceval_output", help="output directory"
    )
    return parser.parse_args()


class CEval:
    DATA_PATH = "ceval/ceval-exam"
    TASK2DESC = {
        "high_school_physics": "高中物理",
        "fire_engineer": "注册消防工程师",
        "computer_network": "计算机网络",
        "advanced_mathematics": "高等数学",
        "logic": "逻辑学",
        "middle_school_physics": "初中物理",
        "clinical_medicine": "临床医学",
        "probability_and_statistics": "概率统计",
        "ideological_and_moral_cultivation": "思想道德修养与法律基础",
        "operating_system": "操作系统",
        "middle_school_mathematics": "初中数学",
        "chinese_language_and_literature": "中国语言文学",
        "electrical_engineer": "注册电气工程师",
        "business_administration": "工商管理",
        "high_school_geography": "高中地理",
        "modern_chinese_history": "近代史纲要",
        "legal_professional": "法律职业资格",
        "middle_school_geography": "初中地理",
        "middle_school_chemistry": "初中化学",
        "high_school_biology": "高中生物",
        "high_school_chemistry": "高中化学",
        "physician": "医师资格",
        "high_school_chinese": "高中语文",
        "tax_accountant": "税务师",
        "high_school_history": "高中历史",
        "mao_zedong_thought": "毛泽东思想和中国特色社会主义理论概论",
        "high_school_mathematics": "高中数学",
        "professional_tour_guide": "导游资格",
        "veterinary_medicine": "兽医学",
        "environmental_impact_assessment_engineer": "环境影响评价工程师",
        "basic_medicine": "基础医学",
        "education_science": "教育学",
        "urban_and_rural_planner": "注册城乡规划师",
        "middle_school_biology": "初中生物",
        "plant_protection": "植物保护",
        "middle_school_history": "初中历史",
        "high_school_politics": "高中政治",
        "metrology_engineer": "注册计量师",
        "art_studies": "艺术学",
        "college_economics": "大学经济学",
        "college_chemistry": "大学化学",
        "law": "法学",
        "sports_science": "体育学",
        "civil_servant": "公务员",
        "college_programming": "大学编程",
        "middle_school_politics": "初中政治",
        "teacher_qualification": "教师资格",
        "computer_architecture": "计算机组成",
        "college_physics": "大学物理",
        "discrete_mathematics": "离散数学",
        "marxism": "马克思主义基本原理",
        "accountant": "注册会计师",
    }
    DATA2CAT = {
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


    def __init__(self):
        pass
'''
    def run(self, shot: int, split: str):
        results, accs = {}, {}
        cat_acc = {}
        cat_cnt = {}

        # run all task
        for task_name in self.TASK2DESC:
            print("=" * 100)
            print(f"run task: {task_name}")
            result, acc, sum_acc, data_cnt = self.run_single_task(task_name, shot, split)
            results[task_name] = result
            accs[task_name] = acc
            cat_name = self.DATA2CAT[task_name]
            if cat_name in cat_acc.keys():
                cat_acc[cat_name] += sum_acc
                cat_cnt[cat_name] += data_cnt
            else:
                cat_acc[cat_name] = sum_acc
                cat_cnt[cat_name] = data_cnt
            result_path = os.path.join(self.output_dir, f"{task_name}.json")
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"save result {acc} to {result_path}")
        for k in cat_acc.keys():
            print("{}: {:.1f}".format(k, cat_acc[k] / cat_cnt[k]  * 100))

        # results
        acc_path = os.path.join(self.output_dir, "acc.json")
        with open(acc_path, "w") as f:
            json.dump(accs, f, indent=2)
        average_acc = sum(accs.values()) / len(accs)
        print("Average Acc: {:.1f}".format(average_acc * 100))

    @torch.no_grad()
    def run_single_task(self, task_name: str, shot: int, split: str):
        # dataset = load_dataset(f"/home/huyiwen/yulan/data/hf-ceval/ceval___ceval-exam/{task_name}")
        dataset = load_dataset("ceval/ceval-exam", task_name)
        # dataset = load_dataset("/home/huyiwen/.cache/huggingface/hub/datasets--ceval--ceval-exam/snapshots/3923b519fd180e689d0961bf3a032ece929742f3/ceval-exam.py", task_name)
        results = []
        acc = 0
        for data in tqdm(dataset[split]):
            prompt = f"以下是中国关于{self.TASK2DESC[task_name]}考试的单项选择题，请选出其中的正确答案。\n"
            if shot != 0:
                shuffled = dataset["dev"].shuffle()
                for i in range(min(shot, len(shuffled))):
                    prompt += "\n" + self.build_example(shuffled[i], with_answer=True)
            prompt += "\n" + self.build_example(data, with_answer=False)
            # input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to('cuda')

            logits = self.model(
                    input_ids=input_ids,
                ).logits[:,-1].flatten()

            candidate_logits = [logits[self.tokenizer(label).input_ids[-1]] for label in ["A", "B", "C", "D"]]
            candidate_logits = torch.tensor(candidate_logits).to(torch.float32)
            probs = (
                torch.nn.functional.softmax(
                    candidate_logits,
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            answer = {i: k for i, k in enumerate(["A", "B", "C", "D"])}[np.argmax(probs)]

            results.append(
                {
                    "prompt": prompt,
                    "correct": answer == data["answer"].strip().upper(),
                    "answer": answer,
                }
            )
            acc += answer == data["answer"].strip().upper()
        ave_acc = acc / len(dataset[split])
        return results, ave_acc, acc, len(dataset[split])

    def build_example(self, data, with_answer: bool = True):
        question = data["question"]
        choice = "\n".join(
            [
                "A. " + data["A"],
                "B. " + data["B"],
                "C. " + data["C"],
                "D. " + data["D"],
            ]
        )
        answer = data["answer"].strip().upper() if with_answer else ""
        return f"{question}\n{choice}\n答案：{answer}"
'''

def main():
    args = parse_argument()

    val_counter, test_counter = 0, 0
    for task_name in tqdm(CEval.TASK2DESC):
        dataset = load_dataset("ceval/ceval-exam", task_name)
        val_counter = val_counter + len(dataset['val'])
        test_counter = test_counter + len(dataset['test'])
    print('Num. C-Eval Data:')
    print('\tVal (used): {}'.format(val_counter))
    print('\tTest: {}'.format(test_counter))

if __name__ == "__main__":
    main()
