from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download scifact.zip dataset and unzip the dataset
dataset = "scifact"  # BEIR-Name
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

#### Print some stats and a sample of the data
print("Corpus Size:", len(corpus))
print("Queries Size:", len(queries))
print("Qrels Size:", len(qrels))
print("-----------------------------")
print("Corpus Type:", type(corpus))
print("Queries Type:", type(queries))
print("Qrels Type:", type(qrels))
print("-----------------------------")
print("Corpus Items:", list(corpus.items())[:5])  # Print first 5 items
print("Queries Items:", list(queries.items())[:5])  # Print first 5 items
print("Qrels Items:", list(qrels.items())[:5])  # Print first 5 items
print("-----------------------------")
print("Sample Corpus:", list(corpus.items())[0])
print("Sample Query:", list(queries.items())[0])
print("Sample Qrel:", list(qrels.items())[0])

#### Load the SBERT model and retrieve using cosine-similarity
model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"), batch_size=16)
retriever = EvaluateRetrieval(model, score_function="dot") # or "cos_sim" for cosine similarity
results = retriever.retrieve(corpus, queries)

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

#### Nicely print the results
print("Evaluation Results:")
print("NDCG@k:", ndcg)
print("MAP@k:", _map)
print("Recall@k:", recall)
print("Precision@k:", precision)
