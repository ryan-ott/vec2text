import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from evaluate import load
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import BERTScorer

 # Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from settings import PROJECT_ROOT, OUTPUTS_DIR, PLOTS_DIR

# --- Helper Functions ---
def clean_line(line):
     """Removes 'Index X:' prefix and extra spacing from a line."""
     line = re.sub(r"^Index \d+: ", "", line)
     line = line.strip()
     line = re.sub(r"\s+", " ", line)
     line = re.sub(r"\s*/\s*", "/", line)
     return line


def calculate_token_f1(pred, label):
     """Calculates the token-level F1 score between two strings."""
     pred_tokens = set(pred.lower().split())
     label_tokens = set(label.lower().split())

     if not pred_tokens and not label_tokens:
         return 1.0  # Both empty
     if not pred_tokens or not label_tokens:
         return 0.0

     precision = len(pred_tokens.intersection(label_tokens)) / len(pred_tokens)
     recall = len(pred_tokens.intersection(label_tokens)) / len(label_tokens)
     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
     return f1


    # --- Metric Extraction and Plotting ---
def extract_metrics_from_log(log_file_path):
    """
    Extracts multiple evaluation metrics (exact match, token F1, SBERT score, BLEU) and max token
    from a log file and returns them in a structured dictionary.
    """
    preds = {}
    labels = {}
    log_file_path = str(log_file_path)

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

    if len(preds) != len(labels):
        print(f"Warning: Number of predictions ({len(preds)}) and labels ({len(labels)}) differ in {log_file_path}.")
         
    min_len = min(len(preds), len(labels))
     
    model = SentenceTransformer('all-mpnet-base-v2')
    bleu = load("sacrebleu")
     
    exact_matches = []
    token_f1s = []
    sbert_scores = []
    bleu_scores = []
    data_added_to_bleu = False

    for i in range(min_len):
        if i in preds and i in labels:
            cleaned_pred = clean_line(preds[i])
            cleaned_label = clean_line(labels[i])
            exact_match = 1.0 if cleaned_pred == cleaned_label else 0.0
            exact_matches.append(exact_match)
            
            token_f1 = calculate_token_f1(cleaned_pred, cleaned_label)
            token_f1s.append(token_f1)
            
            pred_embedding = model.encode(cleaned_pred, convert_to_tensor=True)
            label_embedding = model.encode(cleaned_label, convert_to_tensor=True)
            cosine_sim = util.pytorch_cos_sim(pred_embedding, label_embedding)
            sbert_scores.append(cosine_sim.item())
            
            bleu.add_batch(predictions=[cleaned_pred], references=[[cleaned_label]])
            data_added_to_bleu = True
             
    if data_added_to_bleu:
        bleu_results = bleu.compute()
        bleu_scores = [bleu_results["score"]] * min_len
    else:
         bleu_scores = [0.0] * min_len
         
     # Extract max token and dataset name from filename
    if "maxtoken" in log_file_path:
        max_token = int(log_file_path.split("_")[-1].replace("maxtoken.log", ""))
    else:
        max_token = None

    match = re.search(r"_(.*?)_500samples_", log_file_path)
    dataset_name = match.group(1) if match else None

    if max_token is not None and dataset_name is not None:
        return {
            "max_token": max_token,
            "dataset_name": dataset_name,
            "exact_match": exact_matches,
            "token_f1": token_f1s,
            "sbert_score": sbert_scores,
            "bleu": bleu_scores,
            }
    else:
        return None

def plot_metrics_vs_max_token(all_data, output_dir):
     """
     Plots all calculated metrics (exact match, token F1, SBERT, BLEU) against max token length
     for all datasets.
     """
     metrics = {
         "exact_match": "Exact Match",
         "token_f1": "Token F1",
         "sbert_score": "SBERT Score",
         "bleu": "BLEU"
     }
     
     for metric_name, metric_label in metrics.items():
         plt.figure(figsize=(10, 6))
         for dataset_name, data in all_data.items():
             max_tokens = sorted(data.keys())
             means = [np.mean(data[max_token][metric_name]) for max_token in max_tokens]
             stds = [np.std(data[max_token][metric_name]) for max_token in max_tokens]
             plt.errorbar(max_tokens, means, yerr=stds, fmt='o-', capsize=5, label=dataset_name)

         plt.xlabel("Max Token Length")
         plt.ylabel(metric_label)
         plt.title(f"{metric_label} vs. Max Token Length")
         plt.grid(True)
         plt.legend()
         output_path = output_dir / f"{metric_name}_vs_max_token_all_datasets.png"
         plt.savefig(output_path)
         print(f"{metric_label} plot saved to: {output_path}")
         plt.close()

# --- Qualitative Analysis ---
def extract_and_compare_samples(log_file_path):
     """
     Extracts predictions and labels, compares them, calculates BLEU and BERTScore,
     and prints a qualitative analysis of sample pairs.
     """
     
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
     
     if len(preds) != len(labels):
         print(f"Warning: Number of predictions ({len(preds)}) and labels ({len(labels)}) differ in {log_file_path}.")
     
     min_len = min(len(preds), len(labels))
     scorer = BERTScorer(lang="en", rescale_with_baseline=True)
     
     identical_count = 0
     bleu_scores = []
     bertscore_precisions = []
     bertscore_recalls = []
     bertscore_f1s = []
     
     for i in range(min_len):
         if i in preds and i in labels:
             cleaned_pred = clean_line(preds[i])
             cleaned_label = clean_line(labels[i])
             print(f"--- Index {i} ---")
             print(f"Prediction: {cleaned_pred}")
             print(f"Label:      {cleaned_label}")
             
             if cleaned_pred == cleaned_label:
                 identical_count += 1
                 print("Match: Yes")
             else:
                 print("Match: No")
             
             # BLEU Score
             pred_tokens = nltk.word_tokenize(cleaned_pred.lower())
             label_tokens = nltk.word_tokenize(cleaned_label.lower())
             bleu = sentence_bleu([label_tokens], pred_tokens)
             bleu_scores.append(bleu)
             
             # BERTScore
             P, R, F1 = scorer.score([cleaned_pred], [cleaned_label])
             bertscore_precisions.append(P.item())
             bertscore_recalls.append(R.item())
             bertscore_f1s.append(F1.item())
             
             print(f"BLEU Score: {bleu:.4f}")
             print(f"BERTScore - Precision: {P.item():.4f}, Recall: {R.item():.4f}, F1: {F1.item():.4f}")
     
     print("\n--- Score Calculation ---")
     print(f"Total number of comparisons: {min_len}")
     print(f"Number of identical pairs: {identical_count}")
     
     proportion_identical = identical_count / min_len if min_len > 0 else 0.0
     average_bleu = np.mean(bleu_scores) if bleu_scores else 0.0

     print(f"Proportion of identical lines: {proportion_identical}")
     print(f"Average BLEU Score: {average_bleu:.4f}")
     print(f"Median BLEU Score: {np.median(bleu_scores):.4f}")
     print(f"BLEU Score Standard Deviation: {np.std(bleu_scores):.4f}")

     print("\n--- BERTScore Metrics ---")
     print(f"Average BERTScore - Precision: {np.mean(bertscore_precisions):.4f}, Recall: {np.mean(bertscore_recalls):.4f}, F1: {np.mean(bertscore_f1s):.4f}")
     print(f"Median BERTScore - Precision: {np.median(bertscore_precisions):.4f}, Recall: {np.median(bertscore_recalls):.4f}, F1: {np.median(bertscore_f1s):.4f}")
     print(f"BERTScore Standard Deviation - Precision: {np.std(bertscore_precisions):.4f}, Recall: {np.std(bertscore_recalls):.4f}, F1: {np.std(bertscore_f1s):.4f}")
     
     return {
         "proportion_identical": proportion_identical,
         "average_bleu": average_bleu,
         "bertscore_precision": np.mean(bertscore_precisions),
         "bertscore_recall": np.mean(bertscore_recalls),
         "bertscore_f1": np.mean(bertscore_f1s)
     }


# --- Main Execution ---
def main():
     # --- Configuration ---
     log_dir = OUTPUTS_DIR / "tokenlengthsearch"
     output_dir = OUTPUTS_DIR / "tokenlengthsearch" / "plots"
     qual_log_file = OUTPUTS_DIR / "tokenlengthsearch"/ "repro_T2_gtr-50steps-4beam_dbpedia-entity_500samples_8maxtoken.log" # Example of one log for qualitative analysis
     # ---------------------
     
     print(f"Log directory: {log_dir}")
     print(f"Output directory: {output_dir}")
     
     # -- Experiment 1: Plotting metrics vs max token length
     all_data = {}
     for filename in os.listdir(log_dir):
         if filename.endswith(".log"):
             filepath = log_dir / filename
             result = extract_metrics_from_log(filepath)

             if result:
                 max_token = result["max_token"]
                 dataset_name = result["dataset_name"]
                 if dataset_name not in all_data:
                     all_data[dataset_name] = {}
                 if max_token not in all_data[dataset_name]:
                     all_data[dataset_name][max_token] = {
                         "exact_match": [],
                         "token_f1": [],
                         "sbert_score": [],
                         "bleu": []
                     }
                 all_data[dataset_name][max_token]["exact_match"].extend(result["exact_match"])
                 all_data[dataset_name][max_token]["token_f1"].extend(result["token_f1"])
                 all_data[dataset_name][max_token]["sbert_score"].extend(result["sbert_score"])
                 all_data[dataset_name][max_token]["bleu"].extend(result["bleu"])
                 
     if all_data:
         os.makedirs(output_dir, exist_ok=True)
         plot_metrics_vs_max_token(all_data, output_dir)
     else:
         print("No valid data found in log files for plotting metrics vs max token length.")

     # -- Experiment 2: Qualitative Analysis of one log file
    #  print("\n--- Qualitative Analysis ---")
    #  try:
    #      nltk.data.find('tokenizers/punkt')
    #  except LookupError:
    #      nltk.download('punkt')

    #  try:
    #      extract_and_compare_samples(qual_log_file)
    #  except FileNotFoundError:
    #      print(f"The specified log file for qualitative analysis was not found. Please make sure to provide a valid file path. Given {qual_log_file}")

if __name__ == "__main__":
     main()