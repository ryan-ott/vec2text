import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from scipy.stats import pearsonr, linregress

def extract_dataset_and_tokens(filename):
    """
    Extracts the dataset name and max tokens from the filename.

    Args:
        filename: The name of the file.

    Returns:
        A tuple containing the dataset name and max tokens, or (None, None) if not found.
    """
    match = re.search(r"similarity_between_nq_(.+?)_samplesize(\d+)\.out", filename)  # Correct regex
    if match:
        dataset = match.group(1)
        max_tokens = int(match.group(2))
        return dataset, max_tokens
    return None, None

def extract_similarity_score(filepath):
    """
    Extracts the cosine similarity score from a log file.

    Args:
        filepath: The path to the log file.

    Returns:
        The similarity score as a float, or None if not found.
    """
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if "Cosine similarity" in line:
                    match = re.search(r"Cosine similarity.*: ([-+]?\d*\.\d+|\d+)", line)
                    if match:
                        similarity_score = float(match.group(1))
                        return similarity_score
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
    return None

def create_similarity_dataframe(log_dir):
    """
    Creates a dataframe of similarity scores from a directory of files.

    Args:
        log_dir: The directory containing the files.

    Returns:
        A pandas DataFrame with columns: Dataset, SampleSize, SimilarityScore
    """
    print(f"Checking files in directory: {log_dir}")
    
    all_files = os.listdir(log_dir)
    print(f"All files found in directory: {all_files}")

    files = [f for f in all_files if f.endswith(".out") and f.startswith("similarity_between_")]
    print(f"Filtered files: {files}")

    data = []

    for file in files:
        print(f"\nProcessing file: {file}")
        
        dataset, sample_size = extract_dataset_and_tokens(file)
        print(f"Extracted dataset: {dataset}, Sample size: {sample_size}")
        
        if dataset and sample_size:
            filepath = os.path.join(log_dir, file)
            similarity_score = extract_similarity_score(filepath)

            print(f"Extracted similarity score for {file}: {similarity_score}")

            if similarity_score is not None:
                data.append({
                    'Dataset': dataset,
                    'SampleSize': sample_size,
                    'SimilarityScore': similarity_score
                })
            else:
                print(f"Skipping file {file}: No similarity score found.")
        else:
            print(f"Skipping file {file}: Unable to extract dataset or sample size.")
    
    print(f"\nFinal data extracted: {data}")
    return pd.DataFrame(data)

def create_similarity_bar_plot(df, output_dir):
    """
    Creates a bar plot to visualize dataset similarities, handling missing data.

    Args:
        df: A DataFrame with columns: Dataset, SampleSize, SimilarityScore
        output_dir: The directory to save the plot.
    """
    # Define the desired dataset order (full list)
    dataset_order = [
        "hotpotqa", "nq", "dbpedia-entity", "nfcorpus", "climate-fever",
        "fever", "scifact", "scidocs", "arguana", "msmarco", "quora",
        "trec-covid", "fiqa", "webis-touche2020", "cquadupstack", "bioasq"
    ]

    # Filter out datasets with missing or zero similarity scores
    df_filtered = df.dropna(subset=['SimilarityScore'])
    df_filtered = df_filtered[df_filtered['SimilarityScore'] > 0]

    # 1. Create a DataFrame with ALL datasets in the desired order
    all_datasets_df = pd.DataFrame({'Dataset': dataset_order})

    # 2. Merge the filtered data with the full dataset DataFrame
    merged_df = pd.merge(all_datasets_df, df_filtered, on='Dataset', how='left')

    # 3. Fill missing SimilarityScore with 0 for plotting purposes
    merged_df['SimilarityScore'] = merged_df['SimilarityScore'].fillna(0)

    # Sort the merged DataFrame (not necessary, but for consistency)
    merged_df['Dataset'] = pd.Categorical(merged_df['Dataset'], categories=dataset_order, ordered=True)
    merged_df = merged_df.sort_values('Dataset')

    # --- Plotting ---
    plt.figure(figsize=(14, 7))

    # Create the bar plot using Seaborn
    ax = sns.barplot(x='Dataset', y='SimilarityScore', data=merged_df, color='#348ABD')

    # Customize the plot
    plt.title("Dataset Similarities to NQ", fontsize=16, fontweight='bold')
    plt.xlabel("Dataset", fontsize=12)
    plt.ylabel("Cosine Similarity Score", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    # Set y-axis limits based on the original filtered data (excluding added zeros)
    if not df_filtered.empty:
        plt.ylim(0, df_filtered['SimilarityScore'].max() * 1.1)

        # Add text labels above bars using the correct values from df_filtered
        for i, dataset in enumerate(merged_df['Dataset']):
            similarity_score = df_filtered[df_filtered['Dataset'] == dataset]['SimilarityScore'].values
            if len(similarity_score) > 0:
                # Only add a label if there's a corresponding value in the original data
                ax.text(i, similarity_score[0] + 0.02, f"{similarity_score[0]:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
            else:
                 # Add a zero label for the missing values
                 ax.text(i, 0 + 0.02, f"{0:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add grid lines and remove spines (same as before)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Save the plot ---
    output_path = os.path.join(output_dir, 'dataset_similarities.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Bar plot saved to: {output_path}")

def create_combined_plot(similarity_df, bleu_df, output_dir):
    """
    Creates a scatter plot of dataset similarity to NQ vs. BLEU score,
    including the linear regression line and Pearson correlation coefficient as title.

    Args:
        similarity_df: DataFrame with dataset similarity scores (from create_similarity_dataframe)
        bleu_df: DataFrame with BLEU scores (loaded from the table data)
        output_dir: The directory to save the plot.
    """

    # Merge the two DataFrames on the 'Dataset' column
    merged_df = pd.merge(similarity_df, bleu_df, on='Dataset', how='inner')

    # Convert BLEU to a 0-1 scale
    merged_df['BLEU'] = merged_df['BLEU'] / 100.0

    # --- Calculate the Pearson correlation coefficient ---
    r_value, p_value = pearsonr(merged_df['SimilarityScore'], merged_df['BLEU'])

    # --- Calculate Linear Regression ---
    slope, intercept, _, _, _ = linregress(merged_df['SimilarityScore'], merged_df['BLEU'])

    # --- Plotting ---
    plt.figure(figsize=(12, 8))

    # Create the scatter plot using Seaborn
    ax = sns.scatterplot(x='SimilarityScore', y='BLEU', data=merged_df, s=100, color='#348ABD')

    # Plot the regression line
    sns.regplot(x='SimilarityScore', y='BLEU', data=merged_df, scatter=False, color='red', ci=None)  # Remove confidence interval

    # Customize the plot
    plt.title(f"Dataset Similarity to NQ vs. BLEU Score (Pearson's R = {r_value:.2f})", fontsize=16, fontweight='bold')
    plt.xlabel("Cosine Similarity to NQ", fontsize=12)
    plt.ylabel("BLEU Score", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Set both axis limits to 0-1
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Add annotations with collision avoidance using adjust_text
    texts = []
    for i, row in merged_df.iterrows():
        t = ax.text(row['SimilarityScore'], row['BLEU'], f"{row['Dataset']}",
                    ha='center', va='center', fontsize=9)
        texts.append(t)

    adjust_text(texts,
                x=merged_df['SimilarityScore'].values,
                y=merged_df['BLEU'].values,
                expand_points=(1.2, 1.2),
                arrowprops=dict(arrowstyle="-", color='black', lw=0.5),
                only_move={'points':'xy', 'text':'xy'},
                autoalign='xy')

    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7)

    # --- Save the plot ---
    output_path = os.path.join(output_dir, 'similarity_vs_bleu_with_regression_and_correlation.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Scatter plot with regression and correlation saved to: {output_path}")

def main():
    log_dir = "/home/jesse-wonnink/vec2text/scripts/outputs/similarities"
    output_dir = "/home/jesse-wonnink/vec2text/scripts/outputs/similarities/plots"
    similarity_df = create_similarity_dataframe(log_dir)

    # --- Load BLEU scores from the table data ---
    bleu_data = [
        ("scifact", 80.58),
        ("arguana", 78.10),
        ("msmarco", 63.41),
        ("bioasq", 39.31),
        ("nfcorpus", 81.77),
        ("fiqa", 51.38),
        ("cqadupstack", 46.00),
        ("dbpedia-entity", 84.25),
        ("scidocs", 78.24),
        ("nq", 88.79),
        ("hotpotqa", 89.43),
        ("webis-touche2020", 48.30),
        ("climate-fever", 80.74),
        ("fever", 80.74),
        ("trec-covid", 56.16),
        ("quora", 57.57)
    ]
    bleu_df = pd.DataFrame(bleu_data, columns=['Dataset', 'BLEU'])
    bleu_df['Dataset'] = bleu_df['Dataset'].str.lower()

    if similarity_df.empty:
        print("The similarity DataFrame is empty. Ensure the log directory contains valid files.")
        return

    create_similarity_bar_plot(similarity_df, output_dir)
    print("Bar plot created successfully.")

    create_combined_plot(similarity_df, bleu_df, output_dir)
    print("Combined plot created successfully.")

if __name__ == "__main__":
    main()