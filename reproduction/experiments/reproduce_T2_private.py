import argparse
import ijson
import logging
import nltk
import pandas as pd
import time
import tracemalloc
from datasets import Dataset
from vec2text import data_helpers, load_pretrained_corrector, analyze_utils

def load_bioasq_dataset(file_path, encoding='Windows-1252', max_samples=None):
    """
    Loads the BioASQ dataset, extracting the 'text' field by concatenating 'title' and 'abstractText'.

    :param file_path: Path to the JSON file.
    :param encoding: Encoding of the JSON file.
    :param max_samples: Maximum number of samples to load.
    :return: Hugging Face Dataset object with a 'text' column.
    """
    texts = []
    try:
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            parser = ijson.parse(f)
            in_articles_array = False
            for prefix, event, value in parser:
                if (prefix, event) == ('articles', 'start_array'):
                    in_articles_array = True
                    continue
                if in_articles_array:
                    if event == 'start_map':
                        current_article = {}
                    elif event == 'map_key':
                        current_key = value
                    elif event in ('string', 'number', 'start_array'):
                        current_article[current_key] = value
                    elif event == 'end_map':
                        # Extract and concatenate 'title' and 'abstractText'
                        title = current_article.get('title', '').strip()
                        abstract = current_article.get('abstractText', '').strip()
                        text = f"{title}. {abstract}" if title and abstract else title or abstract
                        if text:
                            texts.append(text)
                        else:
                            logging.warning(f"Sample {len(texts)+1} has empty 'text'. Skipping.")
                        if max_samples and len(texts) >= max_samples:
                            break
        dataset = Dataset.from_dict({'text': texts})
        return dataset

    except Exception as e:
        logging.error(f"Failed to load BioASQ dataset: {e}")
        return Dataset.from_dict({'text': []})  # Return empty dataset on failure


def main(args):
    # Ensure necessary NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    
    # Load the pretrained experiment and trainer
    _, corrector = analyze_utils.load_experiment_and_trainer_from_pretrained(args.model)
    
    # Load the specified BEIR dataset
    print(f"Loading BEIR dataset: {args.beir}")
    if args.beir == "bioasq":
        dataset = load_bioasq_dataset(
            "datasets/bioasq/allMeSH_2020.json",
            encoding='Windows-1252',
            max_samples=args.max_samples,
        )
    else:
        dataset = data_helpers.load_beir_dataset(args.beir)
        dataset = dataset.select(range(args.max_samples)) if args.max_samples else dataset
    
    print("+++ Dataset +++")
    print(dataset)

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
        default="jxm/gtr__nq__32__correct",
        help="Pretrained model identifier (alias)",
    )
    parser.add_argument(
        "--beir",
        type=str,
        default="bioasq",
        choices=[
            "msmarco",  # just for reference of a public dataset
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