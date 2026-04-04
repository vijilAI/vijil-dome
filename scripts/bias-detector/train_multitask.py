# /// script
# dependencies = [
#     "transformers>=4.45.0",
#     "datasets>=3.0.0",
#     "torch>=2.1.0",
#     "scikit-learn>=1.3.0",
#     "accelerate>=0.25.0",
#     "trackio",
# ]
# ///

"""
ModernBERT v3: Multi-task bias classifier
- Head 1: Binary bias detection (safe/unsafe)
- Head 2: EEOC class prediction (6 classes)
- Trained on Safeguard-labeled data with hard example mining
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import trackio
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    PreTrainedModel,
    PretrainedConfig,
)
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# ── Config ──
MODEL_NAME = "answerdotai/ModernBERT-base"
DATASET_NAME = "ciphertext/vijil-bias-detection-eeoc-v4"
HUB_MODEL_ID = "ciphertext/vijil-bias-detector-v4"
MAX_LENGTH = 512
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EEOC_CLASSES = ["age", "disability", "national_origin", "race_color", "religion", "sex_gender"]
EEOC_TO_IDX = {c: i for i, c in enumerate(EEOC_CLASSES)}
BINARY_WEIGHT = 0.7  # weight for binary loss
EEOC_WEIGHT = 0.3    # weight for EEOC class loss

# ── Auth ──
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    from huggingface_hub import login
    login(token=hf_token)

# ── Multi-task model ──
class BiasDetectorConfig(PretrainedConfig):
    model_type = "bias_detector"
    def __init__(self, base_model_name="answerdotai/ModernBERT-base", num_eeoc_classes=6, **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.num_eeoc_classes = num_eeoc_classes

class BiasDetectorModel(PreTrainedModel):
    config_class = BiasDetectorConfig

    def __init__(self, config):
        super().__init__(config)
        self.base = AutoModel.from_pretrained(config.base_model_name)
        hidden_size = self.base.config.hidden_size
        self.binary_head = nn.Linear(hidden_size, 2)
        self.eeoc_head = nn.Linear(hidden_size, config.num_eeoc_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None, labels=None, eeoc_labels=None, **kwargs):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])

        binary_logits = self.binary_head(cls_output)
        eeoc_logits = self.eeoc_head(cls_output)

        loss = None
        if labels is not None:
            binary_loss = nn.CrossEntropyLoss()(binary_logits, labels)
            loss = BINARY_WEIGHT * binary_loss
            if eeoc_labels is not None:
                eeoc_loss = nn.CrossEntropyLoss()(eeoc_logits, eeoc_labels)
                loss += EEOC_WEIGHT * eeoc_loss

        # Return only binary_logits as "logits" — Trainer uses this for compute_metrics
        # EEOC head contributes to loss but we don't evaluate it separately
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(loss=loss, logits=binary_logits)


# ── Load data ──
print("Loading dataset...")
ds = load_dataset(DATASET_NAME)
print(f"Train: {len(ds['train'])}, Val: {len(ds['validation'])}, Test: {len(ds['test'])}")

# ── Tokenizer ──
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    texts = [
        f"{p} [SEP] {r}"
        for p, r in zip(batch["input_prompt"], batch["response"])
    ]
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors=None,
    )
    encoded["labels"] = [int(x) for x in batch["is_biased"]]
    encoded["eeoc_labels"] = [EEOC_TO_IDX.get(c, 0) for c in batch["eeoc_class"]]
    return encoded

print("Tokenizing...")
tokenized = ds.map(tokenize, batched=True, batch_size=1000, remove_columns=ds["train"].column_names)
print(f"Columns: {tokenized['train'].column_names}")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels", "eeoc_labels"])

# ── Model ──
print("Creating multi-task model...")
config = BiasDetectorConfig(base_model_name=MODEL_NAME, num_eeoc_classes=len(EEOC_CLASSES))
model = BiasDetectorModel(config)

# ── Metrics ──
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": (preds == labels).mean(),
        "f1": f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary"),
        "recall": recall_score(labels, preds, average="binary"),
    }

# ── Trackio ──
trackio.init(project="vijil-bias-detector")

# ── Training ──
training_args = TrainingArguments(
    output_dir="vijil-bias-detector-v4",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=64,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=50,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,
    report_to="trackio",
    fp16=True,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    label_names=["labels"],
)

if hf_token and not training_args.hub_token:
    training_args.hub_token = hf_token

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

print("Starting training...")
trainer.train()

# ── Final eval ──
print("\nEvaluating on test set...")
test_results = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
print(f"Test results: {json.dumps(test_results, indent=2)}")

predictions = trainer.predict(tokenized["test"])
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids
print("\nClassification Report:")
print(classification_report(labels, preds, target_names=["unbiased", "biased"]))

# ── Push ──
trainer.push_to_hub(commit_message="v3: multi-task with hard example mining")
tokenizer.push_to_hub(HUB_MODEL_ID, commit_message="Add tokenizer")

trackio.finish()
print("\nDone! Model saved to Hub:", HUB_MODEL_ID)
