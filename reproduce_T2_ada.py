import argparse
import nltk
import time
import tracemalloc
from vec2text import data_helpers, load_pretrained_corrector
import pandas as pd

def main(args):
    # Ensure necessary NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    
    # Load the pretrained experiment and trainer
    corrector = load_pretrained_corrector("text-embedding-ada-002")
    
    # Load the specified BEIR dataset
    print(f"Loading BEIR dataset: {args.beir}")
    dataset = data_helpers.load_beir_dataset(args.beir)

    # Optionally, limit the dataset size for faster evaluation
    if args.max_samples:
        print(f"Original dataset size: {len(dataset)}")
        dataset = dataset.select(range(args.max_samples))
        print(f"Selected first {args.max_samples} samples from the dataset.")

    # Tokenize the dataset with both tokenizers (model tokenizer and embedder tokenizer)
    def tokenize_function(examples):
        # Tokenize with the model's tokenizer
        model_tokens = corrector.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=corrector.model.config.max_seq_length if hasattr(corrector.model.config, 'max_seq_length') else 512,
            return_tensors="pt"
        )
    
        # Tokenize with the embedder's tokenizer
        embedder_tokens = corrector.embedder_tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=corrector.model.config.max_seq_length if hasattr(corrector.model.config, 'max_seq_length') else 512,
            return_tensors="pt"
        )
    
        # Ensure labels are properly formatted
        return {
            "input_ids": model_tokens["input_ids"].squeeze(),
            "attention_mask": model_tokens["attention_mask"].squeeze(),
            "embedder_input_ids": embedder_tokens["input_ids"].squeeze(),
            "embedder_attention_mask": embedder_tokens["attention_mask"].squeeze(),
            "labels": model_tokens["input_ids"].squeeze(),  # Squeeze to remove extra dimensions
        }
    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )
    
    # Set evaluation parameters
    corrector.args.per_device_eval_batch_size = args.batch_size
    corrector.sequence_beam_width = args.beam_width
    corrector.num_gen_recursive_steps = args.steps
    
    print("+++ Trainer Args Passed +++")
    print("num_gen_recursive_steps:", corrector.num_gen_recursive_steps)
    print("sequence_beam_width:", corrector.sequence_beam_width)
    print("Model name:", args.model)
    
    # Start memory and time tracking
    tracemalloc.start()
    start_time = time.time()
    print(tokenized_dataset)
    # Run evaluation
    metrics = corrector.evaluate(
        eval_dataset=tokenized_dataset
    )
    
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
    
    # Optionally, save metrics to a CSV file
    if args.output_csv:
        df = pd.DataFrame([metrics])
        df.to_csv(args.output_csv, index=False)
        print(f"Metrics saved to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Out-of-Domain Experiments on BEIR Datasets")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Pretrained model identifier (alias)",
    )
    parser.add_argument(
        "--beir",
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
            "signal1m",
            "trec-news",
            "robust04",
            "bioasq",
        ],
        help="BEIR dataset to evaluate on",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=1,
        help="Beam width for sequence generation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of recursive generation steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation",
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
    args = parser.parse_args()
    
    print("+++ Arguments Passed +++")
    print(args)
    main(args)