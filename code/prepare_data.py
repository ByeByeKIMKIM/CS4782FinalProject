import os
from datasets import load_dataset
from transformers import BertTokenizer
def prepare_datasets():
    print("Initializing BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print("\nDownloading IMDb dataset...")
    imdb = load_dataset("imdb")
    os.makedirs("./data", exist_ok=True)
    imdb.save_to_disk("./data/imdb")
    print("IMDb dataset saved to ./data/imdb")
    print("\nDownloading BookCorpus dataset (Community Parquet Mirror)...")
    bookcorpus = load_dataset("Yuti/bookcorpus", split="train")
    print(f"Original BookCorpus size: {len(bookcorpus):,} examples")
    print("Extracting 10% subset for scaled-down pre-training...")
    subset_size = len(bookcorpus) // 10
    bookcorpus_subset = bookcorpus.select(range(subset_size))
    print(f"Subset size: {len(bookcorpus_subset):,} examples")
    print("\nTokenizing the BookCorpus subset (this will take some time)...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_special_tokens_mask=True
        )
    tokenized_bookcorpus = bookcorpus_subset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=bookcorpus_subset.column_names 
    )
    print("Saving tokenized BookCorpus subset to disk...")
    tokenized_bookcorpus.save_to_disk("./data/bookcorpus_10pct")
    print("BookCorpus subset saved to ./data/bookcorpus_10pct")
    print("\nPhase 2 Complete. Data is ready for Pre-training.")
if __name__ == "__main__":
    prepare_datasets()
