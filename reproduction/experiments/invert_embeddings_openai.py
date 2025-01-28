import argparse
import time
import torch
import nltk
import numpy as np
import random

from datasets import load_dataset
import evaluate
import vec2text
from vec2text import load_pretrained_corrector
from tqdm import tqdm

nltk.download("punkt", quiet=True)

# Load sacrebleu from Hugging Face's evaluate library
sacrebleu = evaluate.load("sacrebleu")


def token_f1_score(ref_tokens, hyp_tokens):
    """Compute a simple token-level F1 score between two lists of tokens."""
    ref_set = set(ref_tokens)
    hyp_set = set(hyp_tokens)
    common = ref_set.intersection(hyp_set)
    if len(ref_tokens) + len(hyp_tokens) == 0:
        return 0.0
    precision = len(common) / (len(hyp_tokens) if len(hyp_tokens) > 0 else 1)
    recall = len(common) / (len(ref_tokens) if len(ref_tokens) > 0 else 1)
    if (precision + recall) == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Name of dataset, e.g. 'arguana' => ryanott/arguana__openai_ada2")
    parser.add_argument("--beam_width", type=int, default=1,
                        help="Beam width for sequence search")
    parser.add_argument("--num_steps", type=int, default=1,
                        help="Number of correction steps")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="If set, push updated dataset back to Hugging Face")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="If set to a positive integer, randomly sample that many rows from the dataset instead of inverting all.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Loading dataset from Hugging Face
    full_name = f"ryanott/{args.dataset}__openai_ada2"
    print(f"Loading dataset: {full_name}")
    ds = load_dataset(full_name)

    # Use the 'train' split, or default to the first available
    split_name = "train"
    if split_name not in ds:
        split_name = list(ds.keys())[0]
    dataset_split = ds[split_name]

    # If 'inversion' column already exists, warn that it will be overwritten
    existing_columns = dataset_split.column_names
    if "inversion" in existing_columns:
        print(
            "WARNING: An 'inversion' column already exists in this dataset. "
            "The new values for inverted text will overwrite existing entries."
        )

    # Choose how many rows to process. If num_samples <= 0, do all.
    n = len(dataset_split)
    if args.num_samples > 0 and args.num_samples < n:
        sampled_indices = random.sample(range(n), args.num_samples)
        print(f"Randomly sampling {args.num_samples} out of {n} rows.")
    else:
        sampled_indices = list(range(n))
        print(f"No valid --num_samples specified, or too large; processing all {n} rows.")

    print("Loading vec2text corrector for 'text-embedding-ada-002'...")
    corrector = load_pretrained_corrector("text-embedding-ada-002")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare place-holders for results (None for unprocessed rows)
    inversions = [None] * n
    bleu_scores = [None] * n
    f1_scores = [None] * n
    orig_lens = [None] * n
    inv_lens = [None] * n
    gpu_allocated_diffs = [None] * n
    gpu_peaks = [None] * n
    times = []

    for idx in tqdm(sampled_indices, desc="Processing samples", unit="sample"):
        example = dataset_split[idx]
        text = example["text"]
        emb_list = example["embeddings_A"]  # The embedding array
        embeddings = torch.tensor(emb_list, dtype=torch.float).unsqueeze(0).to(device)

        # Reset peak memory stats for this iteration (if CUDA)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(device=device)
            torch.cuda.synchronize(device=device)

        start_time = time.time()
        start_alloc = torch.cuda.memory_allocated(device=device) if device == "cuda" else 0

        # Perform inversion
        inverted_text = vec2text.invert_embeddings(
            embeddings=embeddings,
            corrector=corrector,
            num_steps=args.num_steps,
            sequence_beam_width=args.beam_width,
        )[0]  # invert_embeddings returns a list

        # Measure end time, allocated, and peak usage
        if device == "cuda":
            torch.cuda.synchronize(device=device)
        end_time = time.time()
        end_alloc = torch.cuda.memory_allocated(device=device) if device == "cuda" else 0
        peak_alloc = torch.cuda.max_memory_allocated(device=device) if device == "cuda" else 0

        elapsed = end_time - start_time
        times.append(elapsed)
        alloc_diff_mb = (end_alloc - start_alloc) / (1024**2)  # MB
        peak_diff_mb = (peak_alloc - start_alloc) / (1024**2)  # MB

        # Compute token-level F1
        ref_tokens = nltk.word_tokenize(text.lower())
        hyp_tokens = nltk.word_tokenize(inverted_text.lower())
        f1 = token_f1_score(ref_tokens, hyp_tokens)

        # Compute sacreBLEU
        sacrebleu_result = sacrebleu.compute(
            predictions=[inverted_text.lower()],
            references=[[text.lower()]]
        )
        bleu = sacrebleu_result["score"]

        # Store results
        inversions[idx] = inverted_text
        bleu_scores[idx] = bleu
        f1_scores[idx] = f1
        orig_lens[idx] = len(ref_tokens)
        inv_lens[idx] = len(hyp_tokens)
        gpu_allocated_diffs[idx] = alloc_diff_mb
        gpu_peaks[idx] = peak_diff_mb

    # Calculate final averages only over the processed rows
    processed_count = len(times)
    if processed_count > 0:
        avg_time = np.mean(times)
        # Filter out None values in these columns
        valid_alloc = [x for x in gpu_allocated_diffs if x is not None]
        valid_peak = [x for x in gpu_peaks if x is not None]
        valid_bleu = [x for x in bleu_scores if x is not None]
        valid_f1 = [x for x in f1_scores if x is not None]
        avg_alloc = np.mean(valid_alloc) if len(valid_alloc) > 0 else 0
        avg_peak = np.mean(valid_peak) if len(valid_peak) > 0 else 0
        avg_bleu = np.mean(valid_bleu) if len(valid_bleu) > 0 else 0
        avg_f1 = np.mean(valid_f1) if len(valid_f1) > 0 else 0

        print(f"\n===== Results for {args.dataset} =====")
        print(f"Processed {processed_count} out of {n} rows.")
        print(f"Average per-phrase time (s): {avg_time:.4f}")
        print(f"Avg GPU allocated diff (MB): {avg_alloc:.4f}")
        print(f"Avg GPU peak usage (MB): {avg_peak:.4f}")
        print(f"Average BLEU: {avg_bleu:.4f}")
        print(f"Average token F1: {avg_f1:.4f}")
        total_time = np.sum(times)
        print(f"Total time (s): {total_time:.4f}")
    else:
        print("No rows were processed, skipping stats.")

    # Add / Overwrite columns in the dataset
    ds_split_modified = (
        ds[split_name]
        .add_column("inversion", inversions)
        .add_column("orig_len", orig_lens)
        .add_column("inv_len", inv_lens)
        .add_column("bleu_score", bleu_scores)
        .add_column("token_f1", f1_scores)
        .add_column("gpu_alloc_diff", gpu_allocated_diffs)
        .add_column("gpu_peak_diff", gpu_peaks)
    )

    ds[split_name] = ds_split_modified

    # Optionally push updated dataset back to Hub
    if args.push_to_hub:
        ds.push_to_hub(full_name)
        print(f"Updated dataset pushed to Hugging Face Hub at {full_name}")


if __name__ == "__main__":
    main()
