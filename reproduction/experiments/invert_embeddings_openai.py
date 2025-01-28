import argparse
import time
import torch
import nltk
import numpy as np

from datasets import load_dataset
import evaluate
import vec2text
from vec2text import load_pretrained_corrector
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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
    args = parser.parse_args()

    # Load corrector
    print("Loading vec2text corrector for 'text-embedding-ada-002'...")
    corrector = load_pretrained_corrector("text-embedding-ada-002")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare lists to store new data
    inversions = []
    times = []
    gpu_allocated_diffs = []  # difference in allocated memory
    gpu_peaks = []            # difference in peak memory
    bleu_scores = []
    f1_scores = []
    orig_lens = []
    inv_lens = []

    # Load your dataset from Hugging Face
    full_name = f"ryanott/{args.dataset}__openai_ada2"
    print(f"Loading dataset: {full_name}")
    ds = load_dataset(full_name)

    # We'll iterate over the TRAIN split; adjust if you have other splits
    split_name = "train"
    if split_name not in ds:
        # If the dataset does not have a 'train' split, just pick the first split
        split_name = list(ds.keys())[0]

    dataset_split = ds[split_name]

    for i, example in tqdm(enumerate(dataset_split), total=len(dataset_split), desc="Processing examples"):
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

        # Synchronise again to measure end time, allocated, and peak usage
        if device == "cuda":
            torch.cuda.synchronize(device=device)
        end_time = time.time()
        end_alloc = torch.cuda.memory_allocated(device=device) if device == "cuda" else 0
        peak_alloc = torch.cuda.max_memory_allocated(device=device) if device == "cuda" else 0

        elapsed = end_time - start_time
        alloc_diff_mb = (end_alloc - start_alloc) / (1024**2)  # MB
        peak_diff_mb = (peak_alloc - start_alloc) / (1024**2)  # MB

        # Compute token-level F1
        ref_tokens = nltk.word_tokenize(text.lower())
        hyp_tokens = nltk.word_tokenize(inverted_text.lower())
        f1 = token_f1_score(ref_tokens, hyp_tokens)
        f1_scores.append(f1)

        # Compute SacreBLEU
        sacrebleu_result = sacrebleu.compute(
            predictions=[inverted_text.lower()],
            references=[[text.lower()]]
        )
        bleu = sacrebleu_result["score"]
        bleu_scores.append(bleu)

        times.append(elapsed)
        gpu_allocated_diffs.append(alloc_diff_mb)
        gpu_peaks.append(peak_diff_mb)
        orig_lens.append(len(ref_tokens))
        inv_lens.append(len(hyp_tokens))
        inversions.append(inverted_text)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i+1} samples...")

    # Print average stats
    avg_time = np.mean(times)
    total_time = np.sum(times)
    avg_alloc = np.mean(gpu_allocated_diffs)
    avg_peak = np.mean(gpu_peaks)
    avg_bleu = np.mean(bleu_scores)
    avg_f1 = np.mean(f1_scores)

    print(f"\n===== Results for {args.dataset} =====")
    print(f"Average per-phrase time (s): {avg_time:.4f}")
    print(f"Total time (s): {total_time:.4f}")
    print(f"Avg GPU allocated diff (MB): {avg_alloc:.4f}")
    print(f"Avg GPU peak usage (MB): {avg_peak:.4f}")
    print(f"Average BLEU: {avg_bleu:.4f}")
    print(f"Average token F1: {avg_f1:.4f}")

    # Add columns to the dataset
    ds_new = ds[split_name].add_column("inversion", inversions)
    ds_new = ds_new.add_column("orig_len", orig_lens)
    ds_new = ds_new.add_column("inv_len", inv_lens)
    ds_new = ds_new.add_column("bleu_score", bleu_scores)
    ds_new = ds_new.add_column("token_f1", f1_scores)
    ds_new = ds_new.add_column("gpu_alloc_diff", gpu_allocated_diffs)
    ds_new = ds_new.add_column("gpu_peak_diff", gpu_peaks)

    # Replace the split in the original dataset
    ds[split_name] = ds_new

    # Optionally push updated dataset back to Hub
    if args.push_to_hub:
        ds.push_to_hub(full_name)
        print(f"Updated dataset pushed to Hugging Face Hub at {full_name}")


if __name__ == "__main__":
    main()
