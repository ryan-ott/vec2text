import argparse
import json
import os
import random
import time
from typing import List

import evaluate
import nltk
import numpy as np
import openai
import spacy
import torch
import wandb
from datasets import load_dataset
from tqdm import tqdm

import vec2text
from vec2text import load_pretrained_corrector


nltk.download("punkt", quiet=True)
sacrebleu = evaluate.load("sacrebleu")
nlp = spacy.load("en_core_web_trf")


def get_embeddings_openai(text_list: List[str], model: str = "text-embedding-ada-002") -> torch.Tensor:
    """Get embeddings from OpenAI API for a list of texts."""
    client = openai.OpenAI()
    response = client.embeddings.create(
        input=text_list,
        model=model,
        encoding_format="float",  # override default base64 encoding...
    )
    outputs = [e.embedding for e in response.data]
    return torch.tensor(outputs)


def token_f1_score(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    """Compute a simple token-level F1 score between two lists of tokens."""
    ref_set = set(ref_tokens)
    hyp_set = set(hyp_tokens)
    len_common = len(ref_set.intersection(hyp_set))
    if len(ref_tokens) + len(hyp_tokens) == 0:
        return 0.0
    precision = len_common / (len(hyp_tokens) if len(hyp_tokens) > 0 else 1)
    recall = len_common / (len(ref_tokens) if len(ref_tokens) > 0 else 1)
    if (precision + recall) == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def extract_entities(text: str) -> dict:
    """Extract named entities from a text using spaCy."""
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    return entities


def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


def calculate_entity_overlap(original_entities, reconstructed_entities):
    results = {}
    all_jaccard = []
    all_f1 = []
    entity_types = set(original_entities.keys()).union(reconstructed_entities.keys())
    for entity_type in entity_types:
        set1 = set(original_entities.get(entity_type, []))
        set2 = set(reconstructed_entities.get(entity_type, []))
        jaccard = jaccard_similarity(set1, set2)
        # Simple F1: treat membership as binary classification per entity in the union
        union_entities = set1.union(set2)
        f1 = (
            2 * len(set1.intersection(set2)) / (len(set1) + len(set2))
            if (len(set1) + len(set2)) > 0 else 0
        )
        results[entity_type] = {"jaccard": jaccard, "f1": f1}
        all_jaccard.append(jaccard)
        all_f1.append(f1)
    results["overall"] = {
        "jaccard": sum(all_jaccard) / len(all_jaccard) if all_jaccard else 0,
        "f1": sum(all_f1) / len(all_f1) if all_f1 else 0
    }
    return results


def main(args):
    # -------------------------------------------------------------------------
    # 1. Initialisation
    # -------------------------------------------------------------------------
    wandb.init(
        project="Vec2Text-Repro",
        name=args.run_name,
        config={
            "dataset": args.dataset,
            "beam_width": args.beam_width,
            "num_steps": args.num_steps,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "model": args.model,
        }
    )
    
    if wandb.run is None:
        print("WandB initialization failed.")
        return

    # Seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        print("Using CUDA")
        torch.cuda.manual_seed_all(args.seed)

    # -------------------------------------------------------------------------
    # 2. Dataset Loading
    # -------------------------------------------------------------------------
    full_name = f"ryanott/{args.dataset}__openai_ada2"
    try:
        ds = load_dataset(full_name)
        print(f"Loaded dataset: {full_name}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Decide which split
    split_name = "train"
    if split_name not in ds:
        split_name = list(ds.keys())[0]
    dataset_split = ds[split_name]

    # -------------------------------------------------------------------------
    # 3. Sampling
    # -------------------------------------------------------------------------
    n = len(dataset_split)
    if args.num_samples > 0 and args.num_samples < n:
        sampled_indices = random.sample(range(n), args.num_samples)
        print(f"Randomly sampling {args.num_samples} out of {n} rows.")
    else:
        sampled_indices = list(range(n))
        print(f"No valid --num_samples specified, or too large; processing all {n} rows.")

    # -------------------------------------------------------------------------
    # 4. Model/Corrector
    # -------------------------------------------------------------------------
    print(f"Loading vec2text corrector for '{args.model}'...")
    corrector = load_pretrained_corrector(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # 5. Experiment Loop
    # -------------------------------------------------------------------------
    run_suffix = args.run_name if args.run_name else "run"
    inversions_col = [None] * n
    bleu_col = [None] * n
    f1_col = [None] * n
    orig_lens_col = [None] * n
    inv_lens_col = [None] * n
    gpu_alloc_diffs_col = [None] * n
    gpu_peaks_col = [None] * n
    time_per_sample_col = [None] * n
    entity_jaccard_col = [None] * n
    entity_named_f1_col = [None] * n
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
        )[0]

        if device == "cuda":
            torch.cuda.synchronize(device=device)
        end_time = time.time()
        end_alloc = torch.cuda.memory_allocated(device=device) if device == "cuda" else 0
        peak_alloc = torch.cuda.max_memory_allocated(device=device) if device == "cuda" else 0

        elapsed = end_time - start_time
        times.append(elapsed)

        alloc_diff_mb = (end_alloc - start_alloc) / (1024**2)  # MB
        peak_diff_mb = (peak_alloc - start_alloc) / (1024**2)  # MB

        # Compute Token F1
        ref_tokens = nltk.word_tokenize(text.lower())
        hyp_tokens = nltk.word_tokenize(inverted_text.lower())
        f1_score_val = token_f1_score(ref_tokens, hyp_tokens)

        # Compute BLEU score
        sacrebleu_result = sacrebleu.compute(
            predictions=[inverted_text.lower()],
            references=[[text.lower()]]
        )
        bleu_val = sacrebleu_result["score"]

        # Compute entity overlap metrics
        orig_entities = extract_entities(text)
        inv_entities = extract_entities(inverted_text)
        entity_overlap = calculate_entity_overlap(orig_entities, inv_entities)
        entity_jaccard_col[idx] = entity_overlap["overall"]["jaccard"]
        entity_named_f1_col[idx] = entity_overlap["overall"]["f1"]

        # Store results
        inversions_col[idx] = inverted_text
        bleu_col[idx] = bleu_val
        f1_col[idx] = f1_score_val
        orig_lens_col[idx] = len(ref_tokens)
        inv_lens_col[idx] = len(hyp_tokens)
        gpu_alloc_diffs_col[idx] = alloc_diff_mb
        gpu_peaks_col[idx] = peak_diff_mb
        time_per_sample_col[idx] = elapsed

    # -------------------------------------------------------------------------
    # 6. Summarise & Log Results
    # -------------------------------------------------------------------------
    num_samples = len(times)
    result_dict = {
        "dataset": args.dataset,
        "model_used": args.model,
        "run_name": run_suffix,
        "num_samples": num_samples,
    }

    if num_samples > 0:
        valid_times = [t for t in time_per_sample_col if t is not None]
        avg_time = np.mean(valid_times)

        valid_alloc = [x for x in gpu_alloc_diffs_col if x is not None]
        valid_peak = [x for x in gpu_peaks_col if x is not None]
        valid_bleu = [x for x in bleu_col if x is not None]
        valid_f1 = [x for x in f1_col if x is not None]
        valid_entity_jaccard = [x for x in entity_jaccard_col if x is not None]
        valid_entity_named_f1 = [x for x in entity_named_f1_col if x is not None]

        avg_alloc = np.mean(valid_alloc) if len(valid_alloc) > 0 else 0
        avg_peak = np.mean(valid_peak) if len(valid_peak) > 0 else 0
        avg_bleu = np.mean(valid_bleu) if len(valid_bleu) > 0 else 0
        avg_f1 = np.mean(valid_f1) if len(valid_f1) > 0 else 0
        avg_entity_jaccard = np.mean(valid_entity_jaccard) if valid_entity_jaccard else 0
        avg_entity_named_f1 = np.mean(valid_entity_named_f1) if valid_entity_named_f1 else 0
        total_time = np.sum(valid_times)

        result_dict.update({
            "avg_time_per_sample": avg_time,
            "avg_gpu_alloc": avg_alloc,
            "avg_gpu_peak": avg_peak,
            "avg_bleu": avg_bleu,
            "avg_token_f1": avg_f1,
            "avg_entity_jaccard": avg_entity_jaccard,
            "avg_named_entity_f1": avg_entity_named_f1,
            "total_time": total_time,
            "seed": args.seed,
        })

        # Print to console
        print(f"\n===== Results for run: {run_suffix} =====")
        print(f"Processed {num_samples} out of {n} rows.")
        print(f"Average time per sample (s): {avg_time:.4f}")
        print(f"Avg GPU allocated diff (MB): {avg_alloc:.4f}")
        print(f"Avg GPU peak usage (MB): {avg_peak:.4f}")
        print(f"Average BLEU: {avg_bleu:.4f}")
        print(f"Average token F1: {avg_f1:.4f}")
        print(f"Average Named Entity Jaccard: {avg_entity_jaccard:.4f}")
        print(f"Average Named Entity F1: {avg_entity_named_f1:.4f}")
        print(f"Total time (s): {total_time:.4f}")

        # Log final aggregate metrics to wandb
        wandb.log({
            "num_samples": num_samples,
            "avg_time_per_sample": avg_time,
            "avg_gpu_alloc_mb": avg_alloc,
            "avg_gpu_peak_mb": avg_peak,
            "avg_bleu": avg_bleu,
            "avg_token_f1": avg_f1,
            "avg_entity_jaccard": avg_entity_jaccard,
            "avg_named_entity_f1": avg_entity_named_f1,
            "total_time": total_time,
            "seed": args.seed,
        })

    else:
        print("No rows were processed, skipping stats.")
        wandb.log({"num_samples": 0})

    # Write run metrics to JSON
    os.makedirs(args.out_dir, exist_ok=True)
    out_filename = os.path.join(args.out_dir, f"run_metrics_{run_suffix}.json")
    with open(out_filename, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"Saved run metrics to {out_filename}")
    
    artifact = wandb.Artifact(f"run_metrics_{run_suffix}.json", type="results")
    artifact.add_file(out_filename)
    wandb.log_artifact(artifact)

    # -------------------------------------------------------------------------
    # 7. Add inverted text to dataset & push
    # -------------------------------------------------------------------------
    if args.push_to_hub:
        new_col_name = f"inversion_{run_suffix}"

        # If the column already exists, remove it first to overwrite
        if new_col_name in ds[split_name].column_names:
            print(f"Overwriting values for column {new_col_name}")
            ds[split_name] = ds[split_name].remove_columns(new_col_name)

        # Add inverted text to dataset
        ds[split_name] = ds[split_name].add_column(new_col_name, inversions_col)
        ds.push_to_hub(f"ryanott/{args.dataset}__openai_ada2")
        print(f"Saved updated dataset to hub: ryanott/{args.dataset}__openai_ada2")
    
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Name of dataset, e.g. 'arguana' => ryanott/arguana__openai_ada2")
    parser.add_argument("--beam_width", type=int, default=1,
                        help="Beam width for sequence search")
    parser.add_argument("--num_steps", type=int, default=1,
                        help="Number of correction steps")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="If > 0, randomly sample that many rows from the dataset.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--run_name", type=str, default="",
                        help="Label for the experiment run")
    parser.add_argument("--model", type=str, choices=["text-embedding-ada-002"], # "gtr-base" doesn't work yet, need gtr embeddings
                        default="text-embedding-ada-002",
                        help="Which model to use in load_pretrained_corrector.")
    parser.add_argument("--push_to_hub",
                        action="store_true",
                        help="Push updated dataset to Huggingface Hub")
    parser.add_argument("--out_dir", type=str,
                        help="Directory to save output files to")
    args = parser.parse_args()

    main(args)
