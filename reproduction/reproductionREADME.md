# Vec2Text Reproduction

This folder contains the scripts used to reproduce the results in the paper "Text Embeddings Reveal (Almost) As Much As Text", as well as results, analysis notebooks and other files generated during the reproduction process.

## Getting Started

We suggest creating a new virtual environment to run the scripts. To do so, run the following commands:

```bash
python -m venv v2t-env
source v2t-env/bin/activate
```

Then, install the required packages:

```bash
pip install -r requirements.txt
```

## Running Experiments

Assuming you're in the `vec2text` directory and you have a `v2t-env` set

### In-Domain Reproduction

To reproduce the in-domain experiments, run the following command:

```bash
python experiments/reproduce_T1.py
```

### Out-of-Domain Reproduction

#### Using the GTR model

To reproduce the out-of-domain experiments, run the following command:

```bash
python experiments/reproduce_T2.py
```

>Note: for private BEIR datasets (bioasq), you need to first gain access to the dataset and place it in a folder named `data` in the root directory of the repository. Then use the following command:

```bash
python experiments/reproduce_T2_private.py
```

#### Using the OpenAI `text-embeddings-ada002` model

To reproduce the Table 2 experiment results from the original paper, we provide a job script that computes the inversions of texts from the BEIR datasets from pre-computed embeddings from the OpenAI `text-embeddings-ada002` model. To run the script:

<!-- 1. Save your OpenAI API in your terminal session:

    ```bash
    export OPENAI_API_KEY=your-api-key
    ``` -->

1. Log into huggingface and weights-and-biases:

    ```bash
    huggingface-cli login
    wandb login
    ```

2. Run the job script:

    ```bash
    bash reproduction/job_scripts/reproduce_T2_openai.job
    ```

    There you can adjust the parameters such as number of samples per dataset, number of correction steps, beam width, etc.

    Alternatively, you can run the python script directly instead of looping over all datasets:

    ```bash
    python reproduction/experiments/invert_embeddings.py \
    --model "$MODEL" \
    --dataset "$DS" \
    --beam_width "$BW" \
    --num_steps "$STEPS" \
    --num_samples "$SAMPLES" \
    --seed "$SEED" \
    --run_name "$RUN_NAME" \
    --out_dir "reproduction/outputs/invert_embeddings/$RUN_NAME" \
    --push_to_hub
    ```

    The script loads the pre-computed embeddings from huggingface and computes the inversions for the selected number of samples & computes per sample the metrics like BLEU and token-F1. Results are tracked using weights-and-biases and optionally pushed back to huggingface with the `--push_to_hub` flag.

### Dataset Similarity Analysis

To verify our readings of the dataset similarity, run the following command:

```bash
python experiments/data_similarity_experiement.py
```

### Sequence Length Experiment

To reproduce the sequence length experiment, run the following command:

```bash
python tokenlength_investigation.py
```

## Plots

Our plots were generated using the scripts under `reproduction/experiments/` that end with `_investigation.py` as well as the notebook `reproduction/analysis.ipynb`.
