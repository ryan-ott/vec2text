import argparse
import nltk
import time
import torch
import tracemalloc
from vec2text import analyze_utils

def main(args):
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    
    
    experiment, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(
        args.model,
    )
    train_datasets = experiment._load_train_dataset_uncached(
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        embedder_tokenizer=trainer.embedder_tokenizer
    )

    val_datasets = experiment._load_val_datasets_uncached(
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        embedder_tokenizer=trainer.embedder_tokenizer
    )

    print("+++ Training datasets +++")
    print(train_datasets)
    print(type(train_datasets))
    print(vars(train_datasets))

    print("+++ Validation datasets +++")
    print(val_datasets)
    print(type(val_datasets))
    print(vars(val_datasets))

    trainer.args.per_device_eval_batch_size = 16
    trainer.sequence_beam_width = args.beam_width
    trainer.num_gen_recursive_steps = args.steps
    print("+++ Trainer Args Passed +++")
    print("trainer.num_gen_recursive_steps:", trainer.num_gen_recursive_steps)
    print("trainer.sequence_beam_width:", trainer.sequence_beam_width)
    print("Model name:", args.model)
    torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    start = time.time()
    
    trainer.evaluate(
        eval_dataset=train_datasets["validation"]
    )
    
    print("Time taken:", time.time() - start)
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 10**6}MB; Peak: {peak / 10**6}MB")
    tracemalloc.stop()
    
    if torch.cuda.is_available():
        peak_gpu_mem = torch.cuda.max_memory_allocated() / 10**6
        print(f"Peak GPU memory usage: {peak_gpu_mem} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="jxm/gtr__nq__32__correct",
        choices=["jxm/gtr__nq__32__correct", "jxm/vec2text__openai_ada002__msmarco__msl128__corrector"],
        help="Model name"
    )
    parser.add_argument("--beam-width", type=int, default=1, help="Beam width")
    parser.add_argument("--steps", type=int, default=50, help="Number of recursive steps")
    args = parser.parse_args()
    
    print("+++ Args +++")
    print(args)
    main(args)