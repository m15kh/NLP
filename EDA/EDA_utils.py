import random
import os
import psutil
from datasets import load_dataset

def load_hf_dataset(path_huggingface):
    # RAM before
    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss  # in bytes

    # Load dataset
    dataset = load_dataset(path_huggingface)

    # RAM after
    ram_after = process.memory_info().rss

    # Result
    ram_used = ram_after - ram_before
    print(f"🔹 RAM used for loading dataset: {ram_used / (1024**2):.2f} MB")
    return dataset



def get_sample(dataset):
    idx = random.randint(0, len(dataset['train'])) 
    return dataset['train'][idx]['text']


