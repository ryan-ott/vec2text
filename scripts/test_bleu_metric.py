import evaluate
from pprint import pprint as pp

# Load the BLEU metric
bleu = evaluate.load("sacrebleu")

# Example texts
pred = "          Is it possible to store the energy of lightning?"
true = "Is it possible to store the energy of lightning?"

# Compute BLEU score
results = bleu.compute(predictions=[pred], references=[true])

# Pretty print the results
pp(results)