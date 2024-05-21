import random
import time
from collections import Counter

import dask.dataframe as dd
import numpy as np
import pandas as pd
import tqdm
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

    def is_jailbreak(self, metric, threshold):
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

        return self.is_jailbreak(metrics_dict[metric_name], threshold)

    def check_adv_init(
        self,
        input_text,
        window_size,
        metric_name,
        threshold,
        max_retries=10000,
        verbose=False,
    ):  
        def tokens_to_string(tokens):
            """Convert a list of tokens back to a string using the tokenizer's vocab dictionary."""
            id_to_string = {v: k for k, v in self.tokenizer.vocab.items()}
            return "".join(
                id_to_string[token] for token in tokens if token in id_to_string
            )
        def mutate_token(token_array, idx, tokenizer):
            """Mutate the token at the given index by randomly choosing a different token."""
            possible_tokens = list(tokenizer.vocab.values())
            current_token = token_array[idx]
            new_token = current_token
            while new_token == current_token:
                new_token = np.random.choice(possible_tokens)
            return new_token
        
        token_array = self.tokenizer(input_text)["input_ids"]
        sure_token = self.tokenizer("Sure,")["input_ids"][1]
        for attempt in tqdm.tqdm(range(max_retries), desc="Attempts to adapt init string", disable=not verbose):
            # Tokenize the input text using the self.tokenizer
            if isinstance(token_array[0], list):
                token_array = token_array[0]

            # Calculate metrics using the self.tokenizer
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

            # Check if the input text passes the filter
            if not self.is_jailbroken(metrics_dict[metric_name], threshold):
                if verbose:
                    print(f"Input text passes the filter after {attempt + 1} attempts.")
                    
                token_array[:2] = self.tokenizer("Please help!")["input_ids"][1:-1]
                return tokens_to_string(token_array)

            if attempt == 0:
                # Calculate overall perplexity for the entire input
                window_bigrams = [(token_array[j - 1], token_array[j]) for j in range(1, len(token_array))]
                window_unigrams = [(token_array[j - 1],) for j in range(1, len(token_array))]
        
                window_uncond_probs_2 = [
                    smooth_ngram_probability(
                        self.bigrams_dict,
                        bigram,
                        self.all_bigrams_count,
                        len(self.bigrams_dict),
                    )
                    for bigram in window_bigrams
                ]
                window_uncond_probs_1 = [
                    smooth_ngram_probability(
                        self.unigrams_dict, unigram, self.all_unigrams_count, len(self.unigrams_dict)
                    )
                    for unigram in window_unigrams
                ]
                window_cond_probs = [
                    bigram_prob / unigram_prob
                    for bigram_prob, unigram_prob in zip(
                        window_uncond_probs_2, window_uncond_probs_1
                    )
                ]
                sure_uncond_prob_1 = smooth_ngram_probability(
                        self.unigrams_dict, sure_token, self.all_unigrams_count, len(self.unigrams_dict)
                    )
            
                sure_uncond_prob_2 = smooth_ngram_probability(
                        self.bigrams_dict, (token_array[-1], sure_token), self.all_bigrams_count, len(self.bigrams_dict),
                    )

                sure_prob = sure_uncond_prob_1 / sure_uncond_prob_2

            new_last_token = mutate_token(token_array, -1, self.tokenizer)
            new_sure_uncond_prob_2 = smooth_ngram_probability(
                        self.bigrams_dict, (new_last_token, sure_token), self.all_bigrams_count, len(self.bigrams_dict),
                    )
            if sure_prob < (sure_uncond_prob_1 / new_sure_uncond_prob_2):
                token_array[-1] = new_last_token
                if verbose:
                    print(f"Last token updated.")

            # Find the token with the lowest conditional probability
            min_prob_idx = np.argmin(window_cond_probs)
            
            # Attempt to mutate the token and check conditional probabilities
            original_token = token_array[min_prob_idx]
            token_array[min_prob_idx] = mutate_token(token_array, min_prob_idx, self.tokenizer)

            # Recalculate probabilities for the mutated token and its neighbors
            new_window_bigrams = [(token_array[j - 1], token_array[j]) for j in range(1, len(token_array))]
            new_window_unigrams = [(token_array[j - 1],) for j in range(1, len(token_array))]

            new_window_uncond_probs_2 = [
                smooth_ngram_probability(
                    self.bigrams_dict,
                    bigram,
                    self.all_bigrams_count,
                    len(self.bigrams_dict),
                )
                for bigram in new_window_bigrams
            ]
            new_window_uncond_probs_1 = [
                smooth_ngram_probability(
                    self.unigrams_dict, unigram, self.all_unigrams_count, len(self.unigrams_dict)
                )
                for unigram in new_window_unigrams
            ]
            new_window_cond_probs = [
                bigram_prob / unigram_prob
                for bigram_prob, unigram_prob in zip(
                    new_window_uncond_probs_2, new_window_uncond_probs_1
                )
            ]

            # Check if the conditional probabilities have increased
            if sum(new_window_cond_probs) > sum(window_cond_probs):
                window_cond_probs = new_window_cond_probs
                if verbose:
                    print(f"Mutation accepted at index {min_prob_idx}.")
            else:
                # Revert the mutation if no improvement
                token_array[min_prob_idx] = original_token
                if verbose:
                    print(f"Mutation reverted at index {min_prob_idx}.")
                            
        if verbose:
            print("Max retries reached. No suitable adaptation found.")
        token_array[0][:2] = self.tokenizer("Please help!")["input_ids"][1:-1]
        return tokens_to_string(token_array)


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

    adjusted_perplexities = np.array(
        [p / (e if e > 0 else 1) for p, e in zip(perplexities, entropies)]
    )  # Avoid division by zero
    weighted_cond_probs = np.array(
        [cp * e for cp, e in zip(conditional_probs_2_1, entropies)]
    )
    weighted_uncond_probs_1 = np.array(
        [up1 * e for up1, e in zip(unconditional_probs_1, entropies)]
    )
    weighted_uncond_probs_2 = np.array(
        [up2 * e for up2, e in zip(unconditional_probs_2, entropies)]
    )

    metrics_dict = {
        "perplexities": np.array(perplexities) * -1,
        "conditional_probs_2_1": conditional_probs_2_1,
        "unconditional_probs_1": unconditional_probs_1,
        "unconditional_probs_2": unconditional_probs_2,
        "adjusted_perplexities": adjusted_perplexities * -1,
        "weighted_cond_probs": weighted_cond_probs,
        "weighted_uncond_probs_1": weighted_uncond_probs_1,
        "weighted_uncond_probs_2": weighted_uncond_probs_2,
    }
    return metrics_dict
