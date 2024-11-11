import argparse
import math
import os
import torch
import vec2text
from openai import OpenAI
from utils import halo


def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return api_key


@halo("Loading OpenAI embeddings")
def get_embeddings_openai(text_list, model="text-embedding-ada-002") -> torch.Tensor:
    client = OpenAI(api_key=get_openai_api_key())
    batches = math.ceil(len(text_list) / 128)
    outputs = []
    for batch in range(batches):
        text_list_batch = text_list[batch * 128 : (batch + 1) * 128]
        response = client.embeddings.create(  # Updated API call
            input=text_list_batch,
            model=model,
        )
        outputs.extend([e.embedding for e in response.data])  # Updated response structure
    return torch.tensor(outputs)


@halo("Loading corrector model")
def get_corrector(model_name: str) -> vec2text.models.CorrectorEncoderModel:
    if model_name == "ada":
        corrector = vec2text.load_pretrained_corrector("text-embedding-ada-002")
    else:
        raise ValueError(f"Unknown corrector model name: {model_name}")
    return corrector


def main(args):
    corrector = get_corrector(args.model)
    embeddings = get_embeddings_openai(args.text)
    output_text = vec2text.invert_embeddings(
        embeddings=embeddings.cuda(),
        corrector=corrector,
    )
    print(output_text)
    print(type(output_text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ada", help="Corrector model name")
    parser.add_argument("--text", type=str, nargs="+", default=["Jack Morris is a PhD student at Cornell Tech in New York City", "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity"], help="List of strings to embed")
    args = parser.parse_args()
    
    main(args)
