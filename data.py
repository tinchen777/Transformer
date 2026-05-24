
from datasets import load_dataset
from transformers import AutoTokenizer


dataset = load_dataset("locuslab/TOFU", name="full", split="train")

print(dataset[0])

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
