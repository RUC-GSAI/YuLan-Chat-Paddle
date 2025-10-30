import argparse
import os
import random
import numpy as np
import pandas as pd
from categories import subcategories, categories
import time
from tqdm import tqdm
choices = ["A", "B", "C", "D"]




def main(args):
    
    args.model = f'hf_model_{args.dtype}'
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

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

    counter = 0
    for subject in tqdm(subjects):
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )
        
        counter = counter + len(test_df)

    print('Num MMLU Data: {}'.format(counter))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--ngpu", "-g", type=int, default=1)
    parser.add_argument("--data_dir", "-d", type=str, default="data/mmlu")
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

