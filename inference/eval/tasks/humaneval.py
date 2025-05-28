import json

class HumanEval:
    def __init__(self):
        self.data_path = "/fs/archive/share/u2022000139/zh_datasets/code/human-eval-v2-20210705.jsonl"
        self.data = self.load_data()
        self.logs = self.data

    def load_data(self):
        data = []
        with open(self.data_path, 'r', encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                data.append(line)
        return data
    
    def get_config(self):
        return {
            "batch_size": 16,
            "add_special_tokens": True,
            "max_gen_len": 128,
            "end_token": None,
            "is_sample": False,
        }
    
    def construct_requests(self):
        req_gen = []
        for data_sample in self.data:
            prompt = data_sample["prompt"]

            req_gen.append({
                "context": prompt,
            })

        return {
            "greedy_until": req_gen
        }, 0.0
    
    def process_results(self, results):
        assert len(results) == len(self.logs)
        
        accuarcy = 0.0
        for pred, log in zip(results, self.logs):
            log['response'] = pred

            pred = pred.replace("\\n", "\n")
            pred = pred.split('\", \"source\"')[0]

            pred = pred.split("\ndef")[0]
            pred = pred.split("\n#")[0]
            pred = pred.split("\nif")[0]
            pred = pred.split("\nprint")[0]
            pred = pred.split("\nclass")[0]

            if pred[:4] != "    ":
                pred = "    " + pred

            # logs 
            log['pred'] = pred

        return {
            "acc": accuarcy 
        }, self.logs