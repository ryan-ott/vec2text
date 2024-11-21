from sacrebleu import corpus_bleu
import evaluate
from nltk.tokenize import word_tokenize
import re
metrics_bleu = evaluate.load("sacrebleu")

def calculate_single_bleu(pred, ref):
    score = metrics_bleu.compute(predictions=[pred], references=[ref], lowercase=True)
    # return corpus_bleu([pred], [[ref]]).score
    return score['score']

def calculate_single_f1(pred, ref):
    pred_tokens = set(word_tokenize(pred.lower()))
    ref_tokens = set(word_tokenize(ref.lower()))
    
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
        
    intersection = len(pred_tokens.intersection(ref_tokens))
    precision = intersection / len(pred_tokens) if pred_tokens else 0
    recall = intersection / len(ref_tokens) if ref_tokens else 0
    
    return 2 * (precision * recall) / (precision + recall) if precision + recall else 0

def main():
    with open('quora.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    pairs = text.split('\n\nPair #')[1:]
    predictions = []
    references = []
    
    print("\nPer-pair Results:")
    print("-" * 80)
    
    for i, pair in enumerate(pairs, 1):
        pred_match = re.search(r'\[pred\](.*?)\[true\]', pair, re.DOTALL)
        true_match = re.search(r'\[true\](.*?)\n\-{2,}', pair, re.DOTALL)
        
        if pred_match and true_match:
            pred = pred_match.group(1).strip()
            true = true_match.group(1).strip()
            
            predictions.append(pred)
            references.append(true)
            
            # Calculate individual metrics
            single_bleu = calculate_single_bleu(pred, true)
            single_f1 = calculate_single_f1(pred, true)
            
            print(f"\nPair #{i}")
            print(f"Pred: {pred}")
            print(f"True: {true}")
            print(f"BLEU: {single_bleu:.2f}")
            print(f"F1: {single_f1:.2f}")
            print("-" * 40)
    
    # Calculate overall metrics
    total_bleu = corpus_bleu(predictions, [references]).score
    
    f1_scores = [calculate_single_f1(pred, ref) for pred, ref in zip(predictions, references)]
    total_f1 = sum(f1_scores) / len(f1_scores)
    
    print("\nOverall Results:")
    print("-" * 80)
    print(f"Total pairs evaluated: {len(predictions)}")
    print(f"Overall BLEU Score: {total_bleu:.2f}")
    print(f"Overall Token F1 Score: {total_f1:.2f}")

if __name__ == '__main__':
    main()
