import math
from typing import Dict

import torch
import transformers

from vec2text.trainers.base import BaseTrainer


class InversionTrainerNonAutoregressive(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ######################################################
        self.tokenizer = self.model.tokenizer
        self.embedder_tokenizer = self.model.embedder_tokenizer
        self.call_embedding_model = self.model.call_embedding_model

    def generate(self, inputs: Dict, generation_kwargs: Dict) -> torch.Tensor:
        return self.model.generate(inputs=inputs, generation_kwargs=generation_kwargs)

    def evaluation_loop(
        self, *args, **kwargs
    ) -> transformers.trainer_utils.EvalLoopOutput:
        """
        Run evaluation and returns metrics.
        """
        output = super().evaluation_loop(*args, **kwargs)

        metric_key_prefix = kwargs["metric_key_prefix"]
        
        # Calculate existing perplexity metric
        try:
            perplexity = math.exp(output.metrics[f"{metric_key_prefix}_loss"])
        except KeyError:
            perplexity = -1
        except OverflowError:
            perplexity = float("inf")
        output.metrics[f"{metric_key_prefix}_perplexity"] = perplexity

        # Add NDCG@10 calculation
        try:
            predictions = output.predictions
            labels = output.label_ids
            
            # Calculate relevance scores (1 for exact match, 0 otherwise)
            relevance = (predictions == labels).float()
            
            # Calculate NDCG@10
            k = 10
            dcg = torch.sum((2**relevance - 1) / torch.log2(torch.arange(2, k + 2)), dim=1)
            idcg = torch.sum((2**torch.sort(relevance, descending=True)[0] - 1) / torch.log2(torch.arange(2, k + 2)), dim=1)
            ndcg = (dcg / idcg).mean().item()
            
            output.metrics[f"{metric_key_prefix}_ndcg@10"] = ndcg
        except:
            output.metrics[f"{metric_key_prefix}_ndcg@10"] = -1

        return output
