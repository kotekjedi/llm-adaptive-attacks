import random
import time
from collections import Counter

import dask.dataframe as dd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def process_parquet_to_dict(parquet_path, column_name):
    start_time = time.time()
    df = dd.read_parquet(parquet_path, engine="pyarrow")#.head(100000)
    df = df.compute()

    result_dict = dict(zip(df[column_name], df["normalized_count"]))
    print(f"Time taken: {time.time() - start_time} seconds")
    return result_dict


class Filter:
    def __init__(self, unigram_path="unigrams.parquet", bigram_path="bigrams.parquet", mutation_count=1,
                 guard_model_id=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            use_fast=True,
            trust_remote_code=False,
            legacy=False,
            padding_side="left"
        )
        self.vocab = list(self.tokenizer.get_vocab().values())
        self.special_ids = set(self.tokenizer.all_special_ids)
        self.non_ascii_ids = get_nonascii_toks(self.tokenizer)
        self.vocab_no_special_ascii = [t for t in self.vocab if t not in self.special_ids.union(self.non_ascii_ids)]
        self.unigrams = process_parquet_to_dict(unigram_path, "unigram")
        self.bigrams = process_parquet_to_dict(bigram_path, "bigram")
        self.total_unigrams = 951049362432
        self.total_bigrams = 955312057204
        self.mutation_count = mutation_count
        self.num_unique_bigrams = len(self.bigrams)
        self.num_unique_unigrams = len(self.unigrams)
        print('num unigrams and bigrams', self.num_unique_bigrams, self.num_unique_unigrams)

        if guard_model_id is not None:
            dtype = torch.bfloat16
            self.guard_model_device = "cuda"
            self.guard_tokenizer = AutoTokenizer.from_pretrained(guard_model_id)
            self.guard_model = AutoModelForCausalLM.from_pretrained(guard_model_id,
                                                                    torch_dtype=dtype,
                                                                    device_map=self.guard_model_device
                                                                    )
        #random.seed(42)  # Fixing the random seed for reproducibility

    def guard_model_output(self, message):
        chat = [{"role": "user", "content": message}]
        input_ids = self.guard_tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.guard_model_device)
        output = self.guard_model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        return self.guard_tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

    def is_jailbreak(self, metrics, threshold):
        failing_windows = [i for i, value in enumerate(metrics) if value < threshold]
        return any(value < threshold for value in metrics), failing_windows

    def check_token_consistency(self, tokens, print_inconsistent=False):
        decoded = self.tokenizer.decode(tokens, add_special_tokens=False)
        reencoded = self.tokenizer(decoded, add_special_tokens=False)["input_ids"]

        # Determine if the tokens are consistent
        is_consistent = tokens == reencoded

        if print_inconsistent and not is_consistent:
            # Initialize an empty list to store indices of inconsistent tokens
            inconsistent_indices = []

            # Check consistency of each token up to the minimum length of both token lists
            min_length = min(len(tokens), len(reencoded))
            for i in range(min_length):
                if tokens[i] != reencoded[i]:
                    inconsistent_indices.append(i)

            # Handle cases where the lengths are different
            if len(tokens) != len(reencoded):
                # Add remaining indices as inconsistent from the longer list
                longer_length = max(len(tokens), len(reencoded))
                inconsistent_indices.extend(range(min_length, longer_length))

            inconsistent_tokens = []
            # Collect pairs of inconsistent tokens using the stored indices
            for idx in inconsistent_indices:
                # Check bounds as the lists can be of different lengths
                original_token = tokens[idx] if idx < len(tokens) else 'None'
                reencoded_token = reencoded[idx] if idx < len(reencoded) else 'None'
                inconsistent_tokens.append((original_token, reencoded_token))

            print('Inconsistent pairs:', inconsistent_tokens, tokens, reencoded)

        return is_consistent

    def calculate_window_metrics(self, token_window):
        if len(token_window) == 0:
            return float('inf'), 0  # Return infinite perplexity for an empty window

        # Create bigrams from the token window
        bigrams = [(token_window[i-1], token_window[i]) for i in range(1, len(token_window))]

        # Create unigrams for all tokens, including the first token for its own probability
        unigrams = [(token_window[i-1],) for i in range(1, len(token_window))]
        #print(bigrams, unigrams)
        # Calculate probabilities for bigrams
        bigram_probs = [smooth_ngram_probability(self.bigrams, b, self.total_bigrams, self.num_unique_bigrams) for
                        b in bigrams]

        # Calculate probabilities for unigrams
        unigram_probs = [smooth_ngram_probability(self.unigrams, u, self.total_unigrams, self.num_unique_unigrams)
                         for u in unigrams]


        # Compute conditional probabilities for each bigram based on its preceding unigram
        conditional_probs = [bp / up for bp, up in zip(bigram_probs, unigram_probs)]  # Exclude the last unigram, which has no following bigram
        #print(bigram_probs, unigram_probs, conditional_probs)

        # Calculate perplexity
        if conditional_probs:
            total_log_prob = np.sum(np.log(conditional_probs))
            # Add the log probability of the first unigram to start the chain properly
            total_log_prob += np.log(unigram_probs[0])
            perplexity = np.exp(-total_log_prob / len(bigrams))
        else:
            # Only one word, use its probability directly to estimate perplexity
            perplexity = np.exp(-np.log(unigram_probs[0]))

        # Calculate entropy of the token window
        entropy = calculate_entropy(token_window)

        return perplexity, entropy

    def apply_filter(self, text, window_size, metric_name, threshold, verbose=False, tokens=None, return_metrics=False):

        if tokens is None:
            tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]

        if isinstance(tokens[0], list):
            tokens = tokens[0]

        metrics = self.calculate_metrics(tokens, window_size)
        if verbose:
            print(metrics)

        if return_metrics:
            return not self.is_jailbreak(metrics[metric_name], threshold)[0], metrics[metric_name]
        else:
            return not self.is_jailbreak(metrics[metric_name], threshold)[0]

    def calculate_metrics(self, tokens, window_size):
        metrics = {"perplexity": [], "entropy": []}
        for i in range(len(tokens) - window_size + 1):
            window = tokens[i:i + window_size]
            perplexity, entropy = self.calculate_window_metrics(window)
            metrics["perplexity"].append(-1*perplexity)
            metrics["entropy"].append(entropy)

        return metrics

    def adapt_string_adaptive_not_working(self, input_text, window_size, metric_name, threshold, max_retries=5000,
                     select_from=None, full_shape=None, ids_full=None, seed=0):
        print('Using_seed', seed)
        random.seed(seed)
        if ids_full is not None:
            tokens = ids_full.cpu().tolist()
        else:
            tokens = self.tokenizer(input_text, add_special_tokens=False, padding=False)["input_ids"]

        if isinstance(tokens[0], list):  # Flatten in case of batch tokens
            tokens = tokens[0]

        metrics = self.calculate_metrics(tokens, window_size)
        best_metric_min = min(metrics[metric_name])
        for attempt in tqdm(range(max_retries), desc='Mutating indices'):
            is_jailbreak, failing_windows = self.is_jailbreak(metrics[metric_name], threshold)
            consistent_flag = self.check_token_consistency(tokens)
            if not is_jailbreak and consistent_flag:
                print(f"Input text passes the filter after {attempt + 1} attempts.")
                return tokens[select_from:], True

            failing_indices = set()
            for window_start in failing_windows:
                failing_indices.update(range(window_start, window_start + window_size))

            if select_from is not None:
                failing_indices = list({idx for idx in failing_indices if select_from <= idx < len(tokens)})

            # Calculate weights for each index based on the inverse of bigram count
            bigram_weights = [1.0 / (self.bigrams.get((tokens[i], tokens[i+1]), 0.1) + 1) if i < len(tokens) - 1 else 1.0 / (self.bigrams.get((tokens[i-1], tokens[i]), 0.1) + 1)  for i in failing_indices]

            idx = random.choices(failing_indices, weights=bigram_weights, k=1)[0]
            original_token = tokens[idx]

            # Calculate replacement weights based on bigram counts
            replacement_weights = []
            for candidate in self.vocab_no_special_ascii:
                bigram_count = self.bigrams.get((tokens[idx-1], candidate), 0) if idx > 0 else self.bigrams.get((candidate, tokens[idx+1]), 0)
                replacement_weights.append(bigram_count + 1)  # Add 1 to avoid division by zero

            # Weighted choice of replacement
            tokens[idx] = random.choices(self.vocab_no_special_ascii, weights=replacement_weights, k=1)[0]

            updated_metrics = self.calculate_metrics(tokens, window_size)
            updated_metric_sum = min(updated_metrics[metric_name])

            consistent_flag = self.check_token_consistency(tokens, print_inconsistent=False)

            if updated_metric_sum > best_metric_min and consistent_flag:
                print('adapting, improved metric sum', updated_metric_sum, best_metric_min)
                best_metric_min = updated_metric_sum
                if best_metric_min > threshold:
                    print("Threshold surpassed successfully.")
                    return tokens[select_from:], True
            else:
                tokens[idx] = original_token  # Revert if no improvement

            consistent_flag = self.check_token_consistency(tokens, print_inconsistent=False)
            if best_metric_min > threshold and consistent_flag:
                print("Threshold surpassed successfully.")
                return tokens[select_from:], True

        print("Max retries reached. No suitable adaptation found.", threshold, best_metric_min)
        text_out = self.tokenizer.decode(tokens, add_special_tokens=False)
        return tokens[select_from:], False

    def adapt_string_drop_window(self, input_text, window_size, metric_name, threshold,
                     select_from=None, select_to=None, ids_full=None):

        assert type(ids_full) == list

        if ids_full is not None:
            tokens = ids_full
        else:
            tokens = self.tokenizer(input_text, add_special_tokens=False, padding=False)["input_ids"]

        if isinstance(tokens[0], list):
            tokens = tokens[0]

        # Apply selective slicing based on provided indexes
        if select_from is not None or select_to is not None:
            tokens = tokens[slice(select_from, select_to)]

        metrics = self.calculate_metrics(tokens, window_size)
        is_jailbreak, failing_windows = self.is_jailbreak(metrics[metric_name], threshold)

        tokens = np.array(tokens)
        for window_start in failing_windows:
            tokens[window_start: window_start + window_size] = -1

        tokens = tokens[tokens != -1]

        return tokens.tolist()

    def adapt_string(self, input_text, window_size, metric_name, threshold, max_retries=5000, #5000, #500 #5000, #10000,
                     select_from=None, full_shape=None, ids_full=None, seed=0, filler_tokens=None, behavior=None):

        print('Using_seed', seed)

        if select_from is None:
            # this potentially cuts one token too early
            select_from = len(self.tokenizer(behavior, add_special_tokens=False, padding=False)["input_ids"])
        random.seed(seed)
        if ids_full is not None:
            tokens = ids_full.cpu().tolist()
        else:
            tokens = self.tokenizer(input_text, add_special_tokens=False, padding=False)["input_ids"]

        if isinstance(tokens[0], list):
            tokens = tokens[0]

        metrics = self.calculate_metrics(tokens, window_size)
        best_metric_min = min(metrics[metric_name])

        if filler_tokens is not None:
            max_retries = len(filler_tokens)

        for attempt in tqdm(range(max_retries), desc='Mutating indices'):
            is_jailbreak, failing_windows = self.is_jailbreak(metrics[metric_name], threshold)
            consistent_flag = self.check_token_consistency(tokens)
            if not is_jailbreak and consistent_flag:
                print(f"Input text passes the filter after {attempt + 1} attempts.")
                return tokens[select_from:], True

            failing_indices = set()
            for window_start in failing_windows:
                failing_indices.update(range(window_start, window_start + window_size))

            if select_from is not None:
                failing_indices = {idx for idx in failing_indices if select_from <= idx < len(tokens)}

            if filler_tokens is not None:

                for idx in failing_indices:
                    tokens[idx] = filler_tokens[attempt]
                    metrics = self.calculate_metrics(tokens, window_size)
                    updated_metric_sum = min(metrics[metric_name])

                    if updated_metric_sum > best_metric_min:

                        consistent_flag = self.check_token_consistency(tokens, print_inconsistent=False)

                        if consistent_flag:
                            print('adapting, improved metric sum', updated_metric_sum, best_metric_min)

                            best_metric_min = updated_metric_sum
                            # mutated = True
                            if best_metric_min > threshold:
                                print("Threshold surpassed successfully.")
                                return tokens[select_from:], True

            else:
                for idx in random.sample(failing_indices, min(self.mutation_count, len(failing_indices))):
                    original_token = tokens[idx]
                    tokens[idx] = random.choice(self.vocab_no_special_ascii)
                    metrics = self.calculate_metrics(tokens, window_size)
                    updated_metric_sum = min(metrics[metric_name])

                    consistent_flag = self.check_token_consistency(tokens, print_inconsistent=False)

                    if updated_metric_sum > best_metric_min and consistent_flag:
                        print('adapting, improved metric sum', updated_metric_sum, best_metric_min)

                        best_metric_min = updated_metric_sum
                        #mutated = True
                        if best_metric_min > threshold:
                            print("Threshold surpassed successfully.")
                            return tokens[select_from:], True
                    else:
                        tokens[idx] = original_token  # Revert if no improvement

            consistent_flag = self.check_token_consistency(tokens, print_inconsistent=False)
            if best_metric_min > threshold and consistent_flag:
                print("Threshold surpassed successfully.")
                return tokens[select_from:], True

        print("Max retries reached. No suitable adaptation found.", threshold, best_metric_min)
        text_out = self.tokenizer.decode(tokens, add_special_tokens=False)
        return tokens[select_from:], False


    def adapt_string_old(self, input_text, window_size, metric_name, threshold, max_retries=5000, #10000,
                     select_from=None, full_shape=None, ids_full=None):
        if ids_full is not None:
            #print('select from', select_from)
            tokens = ids_full.cpu().tolist()
        else:
            tokens = self.tokenizer(input_text, add_special_tokens=False, padding=False)["input_ids"]

        if isinstance(tokens[0], list):
            tokens = tokens[0]

        #if full_shape is not None:
        #    print(tokens, full_shape, len(tokens))
        #    print('ids full', ids_full)
        #    assert len(tokens) == full_shape
        #tokens = ids_full

        metrics = self.calculate_metrics(tokens, window_size)
        for attempt in range(max_retries):
            is_jailbreak, failing_windows = self.is_jailbreak(metrics[metric_name], threshold)
            if not is_jailbreak and self.check_token_consistency(tokens):
                print(f"Input text passes the filter after {attempt + 1} attempts.")
                return tokens, True

            failing_indices = set()
            #print('failing windows', failing_windows)
            for window_start in failing_windows:
                failing_indices.update(range(window_start, window_start + window_size))

            if select_from is not None:
                #print('failing_indices before', len(failing_indices), failing_indices)
                failing_indices = {idx for idx in failing_indices if select_from <= idx < len(tokens)}
                #print('failing_indices after', len(failing_indices), failing_indices)

            mutated = False
            for idx in random.sample(failing_indices, min(self.mutation_count, len(failing_indices))):  # Mutate multiple tokens
                original_token = tokens[idx]
                tokens[idx] = random.choice([t for t in self.vocab if t not in self.special_ids])
                updated_metrics = self.calculate_metrics(tokens,
                                                         window_size)  # Recalculate metrics only for affected windows
                if sum(updated_metrics[metric_name]) > threshold:
                    mutated = True
                else:
                    tokens[idx] = original_token  # Revert if no improvement

            if mutated and self.check_token_consistency(tokens):
                print(f"Input text passes the filter after {attempt + 1} attempts.")
                return tokens, True

        print("Max retries reached. No suitable adaptation found.")
        text_out = self.tokenizer.decode(tokens, add_special_tokens=False)
        #return text_out
        return tokens[select_from:], False

def smooth_ngram_probability(dict_used, ngram, counts_all, num_ngrams, smoothing="laplace"):
    ngram = str(ngram)
    probability = dict_used.get(ngram, 0)
    if smoothing == "laplace":
        return ((probability * counts_all) + 1) / (counts_all + num_ngrams)

def calculate_entropy(window_tokens):
    token_counts = Counter(window_tokens)
    total_tokens = sum(token_counts.values())
    probabilities = [count / total_tokens for count in token_counts.values()]
    return -np.sum(p * np.log2(p) for p in probabilities if p > 0)


def get_nonascii_toks(tokenizer, device='cpu'):
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    if "Baichuan2" in tokenizer.name_or_path:
        ascii_toks += [i for i in range(101, 1000)]

    return set(ascii_toks)
