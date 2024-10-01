# encoding="utf-8"
import torch
from torch.utils.data import Dataset
import json

class AsciiDataset(Dataset):
    def __init__(self, data_path):
        samples = []
        with open(data_path, "r") as f:
            for line in f:
                line = json.loads(line.strip())
                line["ori_choices"] = str(line["choices"])
                line["choices"] = "\nA: " + line["choices"][0] \
                                    + "\nB: " + line["choices"][1] \
                                    + "\nC: " + line["choices"][2] \
                                    + "\nD: " + line["choices"][3]
                line["labels"] = ["A", "B", "C", "D"][line["labels"].index(1)]
                samples.append(line)
        self.data = samples

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def map(self, func):
        self.data = [func(x) for x in self.data]

# if __name__=="__main__":
#     data = AsciiDataset("./test_set_all/test.jsonl")
#     print(data[0])
#     def func(example):
#         example["other"]=1
#         return example
#     data.map(func)
#     print(data[0])
