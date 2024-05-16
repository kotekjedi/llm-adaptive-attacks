import time
from collections import Counter

import dask.dataframe as dd
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


def process_parquet_to_dict(parquet_path, column_name):
    # Load the DataFrame using Dask
    start_time = time.time()
    df = dd.read_parquet(parquet_path, engine="pyarrow")

    # Convert Dask DataFrame to Pandas DataFrame
    df = df.compute()

    result_dict = dict(zip(df[column_name], df["normalized_count"]))

    # Timing the entire process
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken} seconds")

    return result_dict


class Filter:
    def __init__(
        self,
        unigram_parquet_path="df_gutenberg_unigrams_dict_normalized.parquet",
        bigram_parquet_path="df_gutenberg_bigrams_dict_normalized.parquet",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            use_fast=True,
            trust_remote_code=False,
            legacy=False,
            padding_side="left",
        )

        self.all_unigrams_count = 951049362432
        self.all_bigrams_count = 955312057204

        self.unigrams_dict = process_parquet_to_dict(unigram_parquet_path, "unigram")
        self.bigrams_dict = process_parquet_to_dict(bigram_parquet_path, "bigram")

    def is_jailbroken(self, metric, threshold):
        return any(value < threshold for value in metric)

    def apply_filter(self, input_text, window_size, metric_name, threshold):
        token_array = self.tokenizer(input_text)["input_ids"]
        if type(token_array[0]) == list:
            token_array = token_array[0] 

        metrics_dict = calculate_metrics(
            token_array,
            window_size,
            bigram_probs=self.bigrams_dict,
            unigram_probs=self.unigrams_dict,
            cond_probs=None,
            all_unigrams_count=self.all_unigrams_count,
            all_bigrams_count=self.all_bigrams_count,
            num_unigrams=len(self.unigrams_dict),
            num_bigrams=len(self.bigrams_dict),
        )

        return self.is_jailbroken(metrics_dict[metric_name], threshold)


# Function to smooth n-gram probabilities
def smooth_ngram_probability(
    dict_used, ngram, counts_all, num_ngrams, smoothing="laplace", smallest_count=5
):
    ngram = str(ngram)
    if ngram in dict_used:
        probability = dict_used[ngram]
    else:
        probability = 0

    if smoothing == "laplace":
        return ((probability * counts_all) + 1) / (counts_all + num_ngrams)


# Function to calculate entropy for a given window of tokens
def calculate_entropy(window_tokens):
    token_counts = Counter(window_tokens)
    total_tokens = sum(token_counts.values())
    probabilities = [count / total_tokens for count in token_counts.values()]
    return -np.sum(p * np.log2(p) for p in probabilities if p > 0)


def calculate_metrics(
    token_array,
    window_size,
    bigram_probs,
    unigram_probs,
    cond_probs,
    all_unigrams_count,
    all_bigrams_count,
    num_unigrams,
    num_bigrams,
):
    # Initialize lists to store the original metrics
    (
        perplexities,
        conditional_probs_2_1,
        unconditional_probs_1,
        unconditional_probs_2,
        entropies,
    ) = ([], [], [], [], [])

    for i in range(0, len(token_array) - window_size + 1):
        window_tokens = token_array[i : i + window_size]
        window_tokens = token_array[i : i + window_size]
        window_bigrams = [
            (window_tokens[j - 1], window_tokens[j])
            for j in range(1, len(window_tokens))
        ]
        window_unigrams = [
            (window_tokens[j - 1],) for j in range(1, len(window_tokens))
        ]

        # Calculate probabilities
        window_uncond_probs_2 = [
            smooth_ngram_probability(
                bigram_probs, bigram, all_bigrams_count, num_bigrams
            )
            for bigram in window_bigrams
        ]
        window_uncond_probs_1 = [
            smooth_ngram_probability(
                unigram_probs, unigram, all_unigrams_count, num_unigrams
            )
            for unigram in window_unigrams
        ]
        window_cond_probs = [
            bigram_prob / unigram_prob
            for bigram_prob, unigram_prob in zip(
                window_uncond_probs_2, window_uncond_probs_1
            )
        ]

        # Calculate Perplexity and Entropy
        perplexity = np.exp(-np.sum(np.log(window_cond_probs)) / len(window_cond_probs))
        entropy_value = calculate_entropy(window_tokens)

        perplexities.append(perplexity)
        entropies.append(entropy_value)
        conditional_probs_2_1.append(np.median(window_cond_probs))
        unconditional_probs_1.append(np.median(window_uncond_probs_1))
        unconditional_probs_2.append(np.median(window_uncond_probs_2))

    adjusted_perplexities = np.array([p / (e if e > 0 else 1) for p, e in zip(perplexities, entropies)])  # Avoid division by zero
    weighted_cond_probs = np.array([cp * e for cp, e in zip(conditional_probs_2_1, entropies)])
    weighted_uncond_probs_1 = np.array([up1 * e for up1, e in zip(unconditional_probs_1, entropies)])
    weighted_uncond_probs_2 = np.array([up2 * e for up2, e in zip(unconditional_probs_2, entropies)])


    metrics_dict = {
        "perplexities": perplexities,
        "conditional_probs_2_1": conditional_probs_2_1,
        "unconditional_probs_1": unconditional_probs_1,
        "unconditional_probs_2": unconditional_probs_2,
        "adjusted_perplexities": adjusted_perplexities * -1,
        "weighted_cond_probs": weighted_cond_probs,
        "weighted_uncond_probs_1": weighted_uncond_probs_1,
        "weighted_uncond_probs_2": weighted_uncond_probs_2,
    }
    return metrics_dict
