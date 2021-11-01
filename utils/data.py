import os
import csv
import json
import string
import numpy as np
import torch

def load_data(task, split, k, seed=0, config_split=None, datasets=None,
              is_null=False):
    if config_split is None:
        config_split = split

    if datasets is None:
        with open(os.path.join("config", task+".json"), "r") as f:
            config = json.load(f)
        datasets = config[config_split]

    data = []
    for dataset in datasets:
        data_path = os.path.join("data", dataset,
                                 "{}_{}_{}_{}.jsonl".format(dataset, k, seed if split=="train" else 100,
                                                          "test" if split is None else split))
        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                if is_null:
                    dp["input"] = ""
                data.append(dp)
    return data

