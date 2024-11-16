import datasets
import pandas as pd
import torch
import vec2text
from datasets import load_dataset
from vec2text.data_helpers import retain_dataset_columns
from vec2text.models import InversionModel, CorrectorEncoderModel

# Load models exactly as shown in README
inversion_model = vec2text.models.InversionModel.from_pretrained("jxm/gtr__nq__32")
corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained("jxm/gtr__nq__32__correct")
corrector = vec2text.load_corrector(inversion_model, corrector_model)

beir_datasets = [
    "quora", "signal1m", "msmarco", "climate-fever", "fever", "dbpedia-entity",
    "nq", "hotpotqa", "fiqa", "webis-touche2020", "cqadupstack", "arguana",
    "scidocs", "trec-covid", "robust04", "bioasq", "scifact", "nfcorpus", "trec-news"
]

results = []

for dataset_name in beir_datasets:
    dataset = load_dataset(f"BeIR/{dataset_name}")
    dataset = retain_dataset_columns(dataset["test"], ["text"])
    texts = dataset["text"][:100]
    
    avg_tokens = sum(len(text.split()) for text in texts) / len(texts)
    
    # Use sequence beam width as shown in README
    reconstructed = vec2text.invert_strings(
        texts,
        corrector=corrector,
        num_steps=20,
        sequence_beam_width=4
    )
    
    metrics = corrector.compute_metrics((reconstructed, texts))
    
    results.append({
        "dataset": dataset_name,
        "tokens": avg_tokens,
        "bleu": metrics["bleu"],
        "token_f1": metrics["token_f1"]
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("tokens")
print(results_df.to_string(index=False))