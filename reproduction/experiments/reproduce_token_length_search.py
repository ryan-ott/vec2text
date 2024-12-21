import argparse
import nltk
import time
import tracemalloc
import os
import random
import sys
from pathlib import Path

# Get path to current directory and to project root
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Now import settings
from reproduction.settings import PROJECT_ROOT, OUTPUTS_DIR

from vec2text import analyze_utils, data_helpers
import pandas as pd


def main(args):
    # Ensure necessary NLTK data is available
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")

    # Load the pretrained experiment and trainer
    experiment, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(
        args.model,
    )

    # Load the specified BEIR dataset
    beir_dataset_name = args.beir_dataset
    print(f"Loading BEIR dataset: {beir_dataset_name}")
    dataset = data_helpers.load_beir_dataset(beir_dataset_name)

    # Optionally, limit the dataset size for faster evaluation
    if args.max_samples:
        print(f"Original dataset size: {len(dataset)}")

        # Set random seed for reproducibility
        random.seed(42)

        # Get random indices
        new_length = min(len(dataset), args.max_samples)
        dataset = dataset.select(range(new_length))

        print(f"cut dataset at {args.max_samples} samples.")

    # Tokenize the dataset with both tokenizers (model tokenizer and embedder tokenizer)
    def tokenize_function(examples):
        # Tokenize with the model's tokenizer
        model_tokens = trainer.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        print(f"Model tokens max_length: {model_tokens['input_ids'].size(-1)}")

        # Tokenize with the embedder's tokenizer
        embedder_tokens = trainer.embedder_tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,  # trainer.embedder_tokenizer.model_max_length,
            return_tensors="pt",
        )
        print(f"Embedder tokens max_length: {embedder_tokens['input_ids'].size(-1)}")

        return {
            "input_ids": model_tokens["input_ids"],
            "attention_mask": model_tokens["attention_mask"],
            "embedder_input_ids": embedder_tokens["input_ids"],
            "embedder_attention_mask": embedder_tokens["attention_mask"],
            "labels": model_tokens["input_ids"].clone(),  # For seq2seq tasks, typically input=target
        }

    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    # Set evaluation parameters
    trainer.args.per_device_eval_batch_size = args.batch_size
    trainer.sequence_beam_width = args.beam_width
    trainer.num_gen_recursive_steps = args.steps

    print("+++ Trainer Args Passed +++")
    print("trainer.num_gen_recursive_steps:", trainer.num_gen_recursive_steps)
    print("trainer.sequence_beam_width:", trainer.sequence_beam_width)
    print("Model name:", args.model)
    print("max_seq_length:", trainer.embedder_tokenizer.model_max_length)

    # Start memory and time tracking
    tracemalloc.start()
    start_time = time.time()
    print(dataset)
    # Run evaluation
    metrics = trainer.evaluate(eval_dataset=tokenized_dataset)

    # End memory and time tracking
    duration = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Print evaluation results
    print("+++ Evaluation Metrics +++")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("Time taken:", duration)
    print(f"Current memory usage: {current / 10**6:.2f}MB; Peak: {peak / 10**6:.2f}MB")

    # save metrics to a CSV file
    if args.output_csv:
        df = pd.DataFrame([metrics])
        # Use a relative path for saving the CSV.
        csv_output_path = str(OUTPUTS_DIR / "csv" / f"reproduce_token_length_search_{args.beir_dataset}_results.csv")
        os.makedirs(os.path.dirname(csv_output_path), exist_ok=True) # Create the folder if it doesnt exist
        df.to_csv(csv_output_path, index=False)
        print(f"Metrics saved to {csv_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce Out-of-Domain Experiments on BEIR Datasets"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Pretrained model identifier (alias)"
    )
    parser.add_argument(
        "--beir_dataset",
        type=str,
        required=True,
        choices=[
            "arguana",
            "climate-fever",
            "cqadupstack",
            "dbpedia-entity",
            "fever",
            "fiqa",
            "hotpotqa",
            "msmarco",
            "nfcorpus",
            "nq",
            "quora",
            "scidocs",
            "scifact",
            "trec-covid",
            "webis-touche2020",
            "robust04",
            "bioasq",
        ],
        help="BEIR dataset to evaluate on",
    )
    parser.add_argument(
        "--beam-width", type=int, default=1, help="Beam width for sequence generation"
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of recursive generation steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (optional)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to save the evaluation metrics as a CSV file",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenizers",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenizers",
    )
    args = parser.parse_args()

    print("+++ Arguments Passed +++")
    print(args)
    main(args)