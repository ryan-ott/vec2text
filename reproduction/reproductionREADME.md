# Vec2Text Reproduction
This folder contains the scripts used to reproduce the results in the paper "Text Embeddings Reveal (Almost) As Much As Text", as well as results, analysis notebooks and other files generated during the reproduction process.

## Getting Started
Clone the repository using:


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


#### In-Domain Reproduction
To reproduce the in-domain experiments, run the following command:
```bash
python experiments/reproduce_T1.py
```

#### Out-of-Domain Reproduction
To reproduce the out-of-domain experiments, run the following command:
```bash
python experiments/reproduce_T2.py
```

>Note: for private BEIR datasets (bioasq), you need to first gain access to the dataset and place it in a folder named `data` in the root directory of the repository. Then use the following command:
```bash
python experiments/reproduce_T2_private.py
```

#### Dataset Similarity Analysis
To verify our readings of the dataset similarity, run the following command:
```bash
python experiments/data_similarity_experiement.py
```

#### Sequence Length Experiment
To reproduce the sequence length experiment, run the following command:
```bash
python tokenlength_investition.py
```

