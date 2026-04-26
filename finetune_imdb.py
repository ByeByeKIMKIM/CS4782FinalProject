import argparse
import numpy as np
from datasets import load_from_disk
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

def main(model_path):
    print("Loading tokenizer and IMDb dataset...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    imdb = load_from_disk("./data/imdb")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    print("Tokenizing dataset...")
    tokenized_imdb = imdb.map(tokenize_function, batched=True, num_proc=4)

    print(f"Loading pre-trained student model from {model_path}...")
    # num_labels=2 for sentiment classification (positive/negative)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./distilbert-imdb",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        # Set to True manually if not automatically detecting GPU
        # fp16=True, 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        compute_metrics=compute_metrics,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Evaluating on test set...")
    eval_results = trainer.evaluate()
    print(f"Test Accuracy: {eval_results['eval_accuracy']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./distilbert-student-pretrained", help="Path to the pre-trained student model")
    args = parser.parse_args()
    main(args.model_path)
