import os
import vec2text
import torch
from openai import OpenAI
import math

os.environ["OPENAI_API_KEY"] = ""  # Set your API key here 

def get_embeddings_openai(text_list, model="text-embedding-ada-002") -> torch.Tensor:
    client = OpenAI()  # Initialize client
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

corrector = vec2text.load_pretrained_corrector("text-embedding-ada-002")
embeddings = get_embeddings_openai([
       "Jack Morris is a PhD student at Cornell Tech in New York City",
       "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity"
])
print(embeddings.shape)
text = vec2text.invert_embeddings(
    embeddings=embeddings.cuda(),
    corrector=corrector
)

print(text)