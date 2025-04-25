import random
import os
import psutil
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pprint import pprint
from collections import defaultdict
from collections import Counter


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


def token_distribution(tokenized_samples, dataset_type="train"):
    # Counting tokens per sample
    token_counts = []
    
    for tokens in tqdm(tokenized_samples, desc="Counting tokens per sample"):
        token_counts.append(len(tokens))
        

    os.makedirs("EDA_results", exist_ok=True)
    # Save the plot with a flexible name
    plt.figure(figsize=(12, 5))
    sns.histplot(token_counts, kde=True, bins=50, color='skyblue')
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.savefig(f"EDA_results/{dataset_type}_token_count_histogram.png")
    
    
    unique_length = np.unique(token_counts)
    print("text, with  minimum tokens")
    pprint(unique_length[:10])



def samples_by_token_length(tokenized_samples, dataset, target_lengths_count=10, max_samples_per_length=5):
    # Get unique lengths
    unique_lengths = np.unique([len(tokens) for tokens in tokenized_samples])
    target_lengths = unique_lengths[:target_lengths_count]

    # Track how many samples we've printed per length
    samples_per_length = defaultdict(int)

    for idx, tokens in enumerate(tokenized_samples):
        token_len = len(tokens)

        if token_len in target_lengths and samples_per_length[token_len] < max_samples_per_length:
            print(f"\n📘 Length: {token_len}")
            pprint(dataset["train"][idx]["text"])
            samples_per_length[token_len] += 1

        # Stop if we've collected all target_lengths_count * max_samples_per_length samples
        if sum(samples_per_length.values()) >= target_lengths_count * max_samples_per_length:
            break
    
def save_top_tokens(tokenized_train_samples, top_k=10_000, output_file="EDA_results/top_10k_tokens.txt"):
    # Counter to keep track of token frequencies
    token_counter = Counter()

    # Iterate through all token IDs and update the counter
    for tokens in tqdm(tokenized_train_samples, desc="Counting token frequencies"):
        token_counter.update(tokens)

    # Get the top_k most frequent tokens
    top_tokens = token_counter.most_common(top_k)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save to file for future use
    with open(output_file, "w") as f:
        for token_id, freq in top_tokens:
            f.write(f"{token_id}\t{freq}\n")

    print(f"\n🔹 {top_k} top tokens saved to {output_file}")
