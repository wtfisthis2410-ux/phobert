# model_multihead.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class MultiHeadPhoBERT(nn.Module):
    def __init__(self, base_model_name, n_labels_bullying, n_labels_severity, n_labels_emotion, dropout=0.2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name, config=self.config)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        # Head 1: bullying (multi-class)
        self.head_bullying = nn.Linear(hidden_size, n_labels_bullying)
        # Head 2: severity (multi-class)
        self.head_severity = nn.Linear(hidden_size, n_labels_severity)
        # Head 3: emotion (multi-class)
        self.head_emotion = nn.Linear(hidden_size, n_labels_emotion)

        # Loss functions (we compute loss inside forward if labels passed)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels_bullying=None, labels_severity=None, labels_emotion=None):
        # encoder forward
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # use CLS pooling (first token)
        cls = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls)

        logits_bullying = self.head_bullying(x)
        logits_severity = self.head_severity(x)
        logits_emotion = self.head_emotion(x)

        loss = None
        if labels_bullying is not None and labels_severity is not None and labels_emotion is not None:
            loss_b = self.ce(logits_bullying, labels_bullying)
            loss_s = self.ce(logits_severity, labels_severity)
            loss_e = self.ce(logits_emotion, labels_emotion)
            # weight losses if needed
            loss = loss_b + loss_s + loss_e

        return {
            "loss": loss,
            "logits_bullying": logits_bullying,
            "logits_severity": logits_severity,
            "logits_emotion": logits_emotion,
        }
