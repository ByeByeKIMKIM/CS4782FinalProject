# Data Storage Information

This directory is designated for datasets used for training and evaluating our DistilBERT re-implementation. 

Due to the immense size of the corpora, the raw dataset files are explicitly ignored via `.gitignore` to prevent inflating the repository size. 

## Obtaining the Datasets

You do **not** need to manually download or format the datasets. Our data preparation pipeline utilizes the [Hugging Face Datasets](https://huggingface.co/docs/datasets/) library to automatically fetch, cache, and tokenize the required splits.

To automatically populate this folder with the IMDb sentiment dataset and the 10% subset of the BookCorpus:
```bash
# From the project root, run:
cd code/
python prepare_data.py
```

### Manual Links (For Reference)
- [IMDb Dataset](https://huggingface.co/datasets/imdb)
- [BookCorpus (Parquet Community Mirror)](https://huggingface.co/datasets/Yuti/bookcorpus)
