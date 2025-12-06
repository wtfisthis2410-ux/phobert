# train.py
import os
import argparse
import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from model_multihead import MultiHeadPhoBERT
import torch.nn.functional as F

def prepare_dataset(df_path, tokenizer, max_len=128, label_maps=None):
    ds = load_dataset("csv", data_files={"train": df_path})["train"]
    def preprocess(ex):
        enc = tokenizer(ex["text"], truncation=True, padding="max_length", max_length=max_len)
        enc["labels_bullying"] = label_maps["bullying"][ex["bullying_label"]]
        enc["labels_severity"] = label_maps["severity"][ex["severity"]]
        enc["labels_emotion"] = label_maps["emotion"][ex["emotion"]]
        return enc
    ds = ds.map(preprocess, batched=False)
    ds = ds.remove_columns([c for c in ds.column_names if c not in ["input_ids","attention_mask","labels_bullying","labels_severity","labels_emotion"]])
    ds.set_format(type="torch", columns=["input_ids","attention_mask","labels_bullying","labels_severity","labels_emotion"])
    return ds

class WrapperTrainer(Trainer):
    # override compute_loss to accept model returning dict
    def compute_loss(self, model, inputs, return_outputs=False):
        labels_b = inputs.pop("labels_bullying")
        labels_s = inputs.pop("labels_severity")
        labels_e = inputs.pop("labels_emotion")
        outputs = model(**inputs, labels_bullying=labels_b, labels_severity=labels_s, labels_emotion=labels_e)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

def main(args):
    base_model = args.base_model
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # define label maps (you should edit these to match your dataset)
    # Example label sets:
    bullying_labels = ["physical","verbal","sexual","social","cyber","none"]
    severity_labels = ["low","medium","high","critical"]
    emotion_labels = ["neutral","sad","angry","fear","happy","other"]

    label_maps = {
        "bullying": {k:i for i,k in enumerate(bullying_labels)},
        "severity": {k:i for i,k in enumerate(severity_labels)},
        "emotion": {k:i for i,k in enumerate(emotion_labels)}
    }

    ds = prepare_dataset(args.data_csv, tokenizer, max_len=args.max_len, label_maps=label_maps)

    model = MultiHeadPhoBERT(base_model, n_labels_bullying=len(bullying_labels),
                             n_labels_severity=len(severity_labels),
                             n_labels_emotion=len(emotion_labels))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="no",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = WrapperTrainer(model=model, args=training_args, train_dataset=ds)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training done. Model saved to", args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default="train_multi.csv", help="CSV with columns text,bullying_label,severity,emotion")
    parser.add_argument("--base_model", type=str, default="vinai/phobert-base", help="base model")
    parser.add_argument("--output_dir", type=str, default="./distilphobert-checkpoint", help="where to save")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()
    main(args)
