import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import os
import re
import sys
import evaluate
import numpy as np
# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings import PROJECT_ROOT, OUTPUTS_DIR
f1_metric = evaluate.load("f1")
nlp = spacy.load("en_core_web_trf")  # Or a larger model like en_core_web_lg
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")



def extract_labels_and_predictions(log_file_path):
    preds = {}
    labels = {}

    with open(log_file_path, 'r') as log_file:
        lines = log_file.readlines()

    preds_section = False
    labels_section = False

    for line in lines:
        if "Contents of decoded_preds:" in line:
            preds_section = True
            labels_section = False
            continue
        elif "Contents of decoded_labels:" in line:
            preds_section = False
            labels_section = True
            continue

        if preds_section:
            match = re.match(r"^Index (\d+): (.*)", line)
            if match:
                index = int(match.group(1))
                preds[index] = match.group(2)
        elif labels_section:
            match = re.match(r"^Index (\d+): (.*)", line)
            if match:
                index = int(match.group(1))
                labels[index] = match.group(2)

    return preds, labels

def extract_entities(text):
    """
    Extracts named entities from a given text using spaCy.

    Args:
        text: The input text.

    Returns:
        A dictionary where keys are entity types and values are lists of entities
        of that type.
    """
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        entities[ent.label_].append(ent.text)
    return entities

def extract_entities_ner(text):
    """
    Extracts named entities from a given text using the Hugging Face NER pipeline.

    Args:
        text: The input text.

    Returns:
        A dictionary where keys are entity types and values are lists of entities
        of that type.
    """
    # Use the NER pipeline to extract entities
    entities = ner_pipeline(text)
    entity_dict = {}
    
    for ent in entities:
        entity_type = ent['entity_group']
        entity_text = ent['word']
        
        if entity_type not in entity_dict:
            entity_dict[entity_type] = []
        
        entity_dict[entity_type].append(entity_text)
    
    return entity_dict

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def calculate_entity_overlap(original_entities, reconstructed_entities):
    """
    Calculates the average Jaccard similarity and F1-score across all
    entity types between the original and reconstructed entity sets.

    Args:
        original_entities: Entities extracted from the original text.
        reconstructed_entities: Entities extracted from the reconstructed text.

    Returns:
        A dictionary containing the average Jaccard similarity and F1-score
        for each entity type, as well as overall averages.
    """
    results = {}
    all_jaccard = []
    all_f1 = []

    entity_types = set(original_entities.keys()).union(reconstructed_entities.keys())

    for entity_type in entity_types:
        set1 = set(original_entities.get(entity_type, []))
        set2 = set(reconstructed_entities.get(entity_type, []))

        jaccard = jaccard_similarity(set1, set2)
        f1 = f1_metric.compute(
            predictions=[1 if e in set2 else 0 for e in set1.union(set2)],
            references=[1 if e in set1 else 0 for e in set1.union(set2)]
        )["f1"] if len(set1.union(set2)) > 0 else 0

        results[entity_type] = {"jaccard": jaccard, "f1": f1}
        all_jaccard.append(jaccard)
        all_f1.append(f1)

    results["overall"] = {
        "jaccard": sum(all_jaccard) / len(all_jaccard) if all_jaccard else 0,
        "f1": sum(all_f1) / len(all_f1) if all_f1 else 0
    }
    return results

def main(log_file_path):
    labels, preds = extract_labels_and_predictions(log_file_path)

    all_results = []
    # Dictionary to hold per-sample scores
    per_sample_scores = {}
    skipped_samples = 0

    for i, pred in list(preds.items()):
        if i not in labels:
            print(f"Warning: No corresponding label for prediction index {i}. Skipping.")
            skipped_samples += 1
            continue

        pred_entities = extract_entities_ner(pred)
        label = labels[i]
        label_entities = extract_entities_ner(label)

        # **Exclude samples where both predicted and label entities are empty**
        if not pred_entities and not label_entities:
            print(f"--- Index {i} ---")
            print("Both prediction and label have no entities. Skipping sample.\n")
            skipped_samples += 1
            continue

        # Calculate entity overlap
        overlap = calculate_entity_overlap(pred_entities, label_entities)
        all_results.append(overlap)

        # **Begin Per-Sample Scoring**
        # Uncomment the following lines if per-sample scores are needed

        # per_sample_scores[i] = overlap
        # print(f"Sample Index: {i}")
        # print(f"Prediction Entities: {pred_entities}")
        # print(f"Label Entities: {label_entities}")
        # print(f"Jaccard Similarity: {overlap['overall']['jaccard']:.4f}")
        # print(f"F1 Score: {overlap['overall']['f1']:.4f}\n")
        # **End Per-Sample Scoring**

    # Aggregate results across all entries
    final_results = {}
    for entity_type in set().union(*[r.keys() for r in all_results]):
        if entity_type != "overall":
            final_results[entity_type] = {
                "jaccard": sum(r[entity_type]["jaccard"] for r in all_results if entity_type in r) / len(all_results),
                "f1": sum(r[entity_type]["f1"] for r in all_results if entity_type in r) / len(all_results)
            }

    final_results["overall"] = {
        "jaccard": sum(r["overall"]["jaccard"] for r in all_results) / len(all_results),
        "f1": sum(r["overall"]["f1"] for r in all_results) / len(all_results)
    }

    print("Final Aggregate Results:")
    print(final_results)

    print(f"\nTotal samples skipped (no entities): {skipped_samples}")

    # If per-sample scores are enabled, you can access them via `per_sample_scores`
    # For example:
    # print(per_sample_scores)

if __name__ == "__main__":
    # --- Configuration ---
    log_dir = OUTPUTS_DIR / "tokenlengthsearch"
    output_dir = OUTPUTS_DIR / "tokenlengthsearch" / "plots"
    log_file = OUTPUTS_DIR / "tokenlengthsearch"/ "repro_T2_gtr-50steps-4beam_fiqa_500samples_32maxtoken.log"
    # ---------------------
    main(log_file)
