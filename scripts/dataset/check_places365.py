from datasets import load_dataset
from itertools import islice

dataset = load_dataset(
    "Andron00e/Places365-custom",
    split="train",
    streaming=True
)

for i, sample in enumerate(islice(dataset, 3)):
    print("sample keys:", sample.keys())
    print(sample)
    print("-" * 50)