import json
import re
import math
import datasets
from datasets import load_dataset


gsm8k_examples = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.
"""

gsm8k_3shot_cot = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
"""

gsm8k_2shot_cot = """Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.
"""

gsm8k_1shot_cot = """Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
"""

gsm8k_simple_examples = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: The answer is 6.
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: The answer is 5.
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: The answer is 39.
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: The answer is 8.
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: The answer is 9.
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: The answer is 29.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: The answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: The answer is 8.
"""

gsm8k_3shot = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: The answer is 6.
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: The answer is 33.
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: The answer is 8.
"""

# only support few shot
class GradeSchoolMath8K:
    def __init__(self):
        self.data_path = "./data/gsm8k/gsm8k.jsonl"
        self.data = self.load_data()
        self.logs = []
        # self.examples = gsm8k_3shot_cot
        self.examples = gsm8k_2shot_cot
        # self.examples = gsm8k_1shot_cot

    def load_data(self):
        # return datasets.load_dataset('json', data_files=self.data_path)
        # return datasets.load_from_disk(self.data_path)
        return [json.loads(line) for line in open(self.data_path).readlines()]
        # return [json.loads(line) for line in open(self.data_path).readlines()][:20]

    def get_config(self):
        return {
            "batch_size": 1,
            "add_special_tokens": True,
            "max_gen_len": 72,
            "end_token": None,
            "is_sample": False,
        }
    
    def construct_requests(self):
        req_gen = []
        for idx, data_sample in enumerate(self.data):
            prompt = self.examples
            prompt += "Q: " 
            prompt += data_sample["question"]
            prompt += "\nA: "

            req_gen.append({
                "context": prompt,
            })

            # logs 
            self.logs.append({
                "data_idx": idx,
                "prompt": prompt,
            })
        
        return {
            "greedy_until": req_gen
        }, 0.0
    

    def extract_ans(self, text):
        nums = re.findall(r'\d+\.?\d*', text)
        return float(nums[-1]) if nums else -10000000000000


    def process_results(self, results):
        assert len(results) == len(self.data)
        assert len(results) == len(self.logs)
        
        accuarcy = 0.0
        for pred, data_sample, log in zip(results, self.data, self.logs):
            log['response'] = pred
            is_right = False

            label = data_sample['answer'].split("### ")[-1].replace(',', '')

            pred = pred.strip().split("\nQ")[0]
            pred = pred.strip().split("\nA")[0]
            
            ans = self.extract_ans(pred)

            # if "The answer is" in pred:
            #     pred = pred.split("The answer is")[-1]
            # elif "answer is" in pred:
            #     pred = pred.split("answer is")[-1]
            # else:
            #     pred = pred

            # pred = pred.replace(',', '')
            # ans = re.findall(r'-?\d+\.?\d*', pred)
            # if len(ans) == 0:
            #     ans = ""
            # else:
            #     ans = ans[-1]
            #     ans = ans[:-1] if ans[-1] == '.' else ans

            # if ans == label:
            #     accuarcy += 1.0
            #     is_right = True
            # elif ans != "" and float(ans) == float(label):
            #     accuarcy += 1.0
            #     is_right = True

            label = float(label)
            if (math.fabs(label - ans) <= 1e-4):
                accuarcy = accuarcy + 1.0
                is_right = True
            else:
                accuarcy = accuarcy + 0.0
                is_right = False
            
            # print('label = {}; pred = {}'.format(label, ans))

            # logs 
            log['label'] = label
            log['pred'] = pred
            log['extracted_ans'] = ans
            log["is_right"] = is_right

        accuarcy = accuarcy / len(self.data)
        return {
            "acc": accuarcy 
        }, self.logs

class GSM8K_direct(GradeSchoolMath8K):
    def __init__(self):
        super().__init__()
        self.examples = gsm8k_simple_examples
