import os
import argparse
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True, help="Path al file CSV del test set")
    parser.add_argument("--model_path", type=str, required=True, help="Directory contenente modello e tokenizer fine-tuned")
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator()

    # --- Caricamento test set --------------------------------------------
    df = pd.read_csv(args.test_file)
    if "tweet" not in df.columns:
        raise ValueError("Il CSV deve contenere una colonna 'tweet'.")

    has_labels = "ideology_binary" in df.columns and df["ideology_binary"].notna().any()

    label_mapping = {"left": 0, "right": 1}
    id2label = {v: k for k, v in label_mapping.items()}

    texts = df["tweet"].astype(str).tolist()
    dataset_dict = {"text": texts}

    if has_labels:
        df = df[df["ideology_binary"].isin(label_mapping.keys())]
        labels = df["ideology_binary"].map(label_mapping).tolist()
        dataset_dict["label"] = labels

    dataset = Dataset.from_dict(dataset_dict)

    # --- Tokenizer e modello ---------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)

    # --- Preprocessing ---------------------------------------------------
    def preprocess(example):
        return tokenizer(example["text"], truncation=True, padding=True)

    remove_cols = ["text"]  # Non rimuovere 'label' per mantenere batch["labels"]
    dataset = dataset.map(preprocess, batched=True, remove_columns=remove_cols)


    # --- DataLoader ------------------------------------------------------
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=data_collator)
    model, dataloader = accelerator.prepare(model, dataloader)

    # --- Inference -------------------------------------------------------
    model.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        preds = accelerator.gather(preds)
        all_preds.extend(preds.cpu().numpy())

        if has_labels:
            labels = accelerator.gather(batch["labels"])
            all_labels.extend(labels.cpu().numpy())

    # --- Salvataggio predizioni ------------------------------------------
    df["prediction"] = [id2label[int(p)] for p in all_preds]
    df.to_csv("predictions.csv", index=False)
    print("\n predizioni salvate in: predictions.csv")

    # --- Metriche se disponibili -----------------------------------------
    if has_labels:
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
        report = classification_report(all_labels, all_preds, target_names=["left", "right"])

        print(f"\n Metriche sul test set:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print("\nClassification Report:\n", report)

if __name__ == "__main__":
    main()

