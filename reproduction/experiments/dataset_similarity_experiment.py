from sentence_transformers import SentenceTransformer, util
import torch
import argparse
import random
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
import datasets
import os

def get_document_embeddings(dataset, model):
    document_texts = [example["text"] for example in dataset]
    document_embeddings = model.encode(document_texts, show_progress_bar=True, convert_to_tensor=True)
    return document_embeddings

def calculate_average_embedding(embeddings):
    return torch.mean(embeddings, dim=0)

def reduce_dataset(dataset, sample_size, random_seed=42):
    if sample_size > len(dataset):
        raise ValueError("Sample size cannot be larger than the dataset size.")

    random.seed(random_seed)
    indices = random.sample(range(len(dataset)), sample_size)
    sampled_dataset = dataset.select(indices)
    return sampled_dataset

def load_beir_dataset(dataset_name, data_dir="datasets"):
    dataset_dir = os.path.join(data_dir, dataset_name)

    if not os.path.exists(dataset_dir):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = beir_util.download_and_unzip(url, data_dir)
    else:
        data_path = dataset_dir

    
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    documents = [{"text": corpus[doc_id]['text'], "title": corpus[doc_id].get("title", "")} for doc_id in corpus]

    dataset = datasets.Dataset.from_dict({
        "text": [doc["text"] for doc in documents] + [queries[q_id] for q_id in queries],
        "data_type": ["document"] * len(documents) + ["query"] * len(queries)
    })

    return dataset

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
    if args.dataset_A == args.dataset_B: # Load only once if datasets are identical
        print("Datasets are identical. Loading only once.")
        dataset = load_beir_dataset(args.dataset_A)
        dataset_A = reduce_dataset(dataset, args.sample_size, random_seed=42)
        dataset_B = reduce_dataset(dataset, args.sample_size, random_seed=43) #Different seed for different samples to account for randomness

    else:
        if args.dataset_B == "bioasq":
        dataset_A = data_helpers.load_beir_dataset(args.dataset_A)
        dataset_B = load_bioasq_dataset(
             "/home/scur2868/IR2/datasets/bioasq/allMeSH_2020.json",
             encoding='Windows-1252',
             max_samples=args.max_samples,
        )
        else:
            print(f"Loading BEIR dataset: {args.dataset_A} for dataset A")
            dataset_A = load_beir_dataset(args.dataset_A)
            dataset_B = load_beir_dataset(args.dataset_B)

        if args.sample_size:
            dataset_A = reduce_dataset(dataset_A, args.sample_size)
            dataset_B = reduce_dataset(dataset_B, args.sample_size)

    # Load Sentence-BERT model
    model = SentenceTransformer(args.model_name)

    # Generate embeddings (only for documents, not queries)
    embeddings_A = get_document_embeddings(dataset_A.filter(lambda example: example["data_type"] == "document"), model)
    embeddings_B = get_document_embeddings(dataset_B.filter(lambda example: example["data_type"] == "document"), model)

    # Calculate average embeddings
    average_embedding_A = calculate_average_embedding(embeddings_A)
    average_embedding_B = calculate_average_embedding(embeddings_B)

    # Calculate and print similarity
    similarity = util.cos_sim(average_embedding_A, average_embedding_B).item()
    print(f"Cosine similarity between average embeddings of {args.dataset_A} and {args.dataset_B}: {similarity}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate similarity between two BEIR datasets using Sentence-BERT embeddings.")
    parser.add_argument("--dataset_A", type=str, required=True, help="Name of the first BEIR dataset (e.g., 'arguana')")
    parser.add_argument("--dataset_B", type=str, required=True, help="Name of the second BEIR dataset (e.g., 'scidocs')")
    parser.add_argument("--model_name", type=str, default="all-mpnet-base-v2", help="Name of the Sentence-BERT model to use")
    parser.add_argument("--sample_size", type=int, default=None, help="Size of the random sample to use from each dataset")
    args = parser.parse_args()
    main(args)