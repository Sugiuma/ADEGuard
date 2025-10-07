from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
import random
from sklearn.model_selection import train_test_split

# -------------------------------
# STEP 1: Detect labels from files
# -------------------------------
def collect_labels(files):
    labels = set()
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                labels.add(parts[-1])  # last col = label
    return sorted(list(labels))

label_list = collect_labels(["/content/sample_data/project1.conll"])#, "/content/sample_data/gold.conll"])
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}
print("Detected labels:", label_list)

# -------------------------------
# STEP 2: Reader for your format
# -------------------------------
def read_conll(filepath):
    examples = []
    tokens, tags = [], []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    examples.append({"tokens": tokens, "ner_tags": [label2id[tag] for tag in tags]})
                    tokens, tags = [], []
                continue

            parts = line.split()
            token, tag = parts[0], parts[-1]   # first col = token, last col = label
            tokens.append(token)
            tags.append(tag)

        if tokens:  # last sentence
            examples.append({"tokens": tokens, "ner_tags": [label2id[tag] for tag in tags]})

    return examples

# -------------------------------
# STEP 3: Build dataset
# -------------------------------
silver_data = read_conll("/content/sample_data/project1.conll")   # weak silver labels
#gold_data   = read_conll("/content/sample_data/gold.conll")     # gold labels

#print("Gold examples:", len(gold_data))
print("Silver examples:", len(silver_data))

random.shuffle(silver_data)
#random.shuffle(gold_data)


# First split: train vs temp (val+test)
train_data, temp_data = train_test_split(
    silver_data,
    test_size=0.15,   # 15% goes to val+test
    random_state=42,  # reproducibility
    shuffle=True
)

# Second split: val vs test
val_data, test_data = train_test_split(
    temp_data,
    test_size=0.5,    # half of 15% → 7.5% test, 7.5% val
    random_state=42,
    shuffle=True
)

print(f"Train: {len(train_data)}")
print(f"Val:   {len(val_data)}")
print(f"Test:  {len(test_data)}")


#train_data = silver_data + gold_data[:350]  # mix weak + gold
#val_data   = gold_data[350:425]  # ~75 examples
#test_data  = gold_data[425:]     # ~75 examples

dataset = DatasetDict({
   "train": Dataset.from_list(train_data),
   "validation": Dataset.from_list(val_data),
    "test": Dataset.from_list(test_data),
})

#print(dataset)

from collections import Counter

def print_label_counts(dataset, split_name, key="ner_tags"):
    """
    Print label counts for a dataset split.
    - dataset: HuggingFace Dataset
    - split_name: "train", "validation", or "test"
    - key: column containing labels ("ner_tags" before tokenization, "labels" after)
    """
    counts = Counter()
    for labels in dataset[split_name][key]:
        counts.update(labels)

    # remove padding ignore index (-100) if present
    if -100 in counts:
        del counts[-100]

    print(f"\n🔹 {split_name} counts (from '{key}'):")
    for label_id, cnt in sorted(counts.items()):
        print(f"  {label_id:>2} : {cnt}")

from datasets import Dataset, DatasetDict
from collections import Counter
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
import evaluate
import numpy as np

# -----------------------------
# 3️⃣ Load your dataset
# -----------------------------
dataset = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data),
    "test": Dataset.from_list(test_data),
})

label_column = "ner_tags"


for split in ["train", "validation", "test"]:
    print_label_counts(dataset, split, key="ner_tags")

import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from collections import Counter

# ------------------------
# 1. Define labels
# ------------------------
label_list = ["B-ADE", "B-DRUG","I-ADE","I-DRUG","O"]
num_labels = len(label_list)
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}

# ------------------------
# 2. Compute class weights from dataset
# ------------------------
def compute_class_weights(dataset, label_column="ner_tags"):
    counts = Counter()
    for split in ["train"]:
        for seq in dataset[split][label_column]:
            counts.update(seq)
    total = sum(counts.values())
    weights = []
    for i in range(num_labels):
        # weight = total / (num_labels * class_count)
        weights.append(total / (num_labels * counts[i]) if counts[i] > 0 else 1.0)
    return torch.tensor(weights, dtype=torch.float)

class_weights = compute_class_weights(dataset)
print("Class weights:", class_weights)

weights = torch.tensor([ 2.6494, 22.1747,  4.8549, 51.0511,  0.2298])
weights = weights / weights.max()  # normalize so max = 1
print("weights",weights)

print("I-DRUG / O weight ratio:", weights[3]/weights[4])

model_name = "dmis-lab/biobert-base-cased-v1.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
num_labels = len(label_list)  # e.g., 5 labels
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    return_dict=True  # ensures outputs.logits exists
)

# Freeze all parameters
for param in model.bert.parameters():
    param.requires_grad = False

# Unfreeze the last 4 encoder layers
for layer in model.bert.encoder.layer[-4:]:
    for param in layer.parameters():
        param.requires_grad = True

# Classifier head is always trainable
for param in model.classifier.parameters():
    param.requires_grad = True


for name, param in model.named_parameters():
    print(name, param.requires_grad)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=512,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                # for subword tokens: keep same label if not O, else O
                label_ids.append(label[word_idx] if label[word_idx] != label2id["O"] else label2id["O"])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Normalized weights tensor
weights = torch.tensor([0.0519, 0.4344, 0.0951, 1.0000, 0.0045]).to("cuda")

# Overwrite model's forward via Trainer
def compute_loss(model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
    return (loss, outputs) if return_outputs else loss

data_collator = DataCollatorForTokenClassification(tokenizer)

from seqeval.metrics import classification_report
import numpy as np

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_preds = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    report = classification_report(true_labels, true_preds, output_dict=True)
    return {
        "precision": report["micro avg"]["precision"],
        "recall": report["micro avg"]["recall"],
        "f1": report["micro avg"]["f1-score"]
    }

training_args = TrainingArguments(
    output_dir="./ner_biobert",
    eval_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    logging_steps=100,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to=[],
    push_to_hub=False
)

from transformers import Trainer

class WeightedTrainer(Trainer):
    def __init__(self, weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

trainer = WeightedTrainer(
    weights=weights,
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

metrics = trainer.evaluate(tokenized_datasets["test"])
print(metrics)

save_path = "biobert-ner-final"

# Save model
model.save_pretrained(save_path)

# Save tokenizer
tokenizer.save_pretrained(save_path)

print(f"✅ Model and tokenizer saved to '{save_path}'")

from seqeval.metrics import classification_report

preds_output = trainer.predict(tokenized_datasets["validation"])
preds = np.argmax(preds_output.predictions, axis=2)

true_labels = [[id2label[l] for l in label if l != -100] for label in preds_output.label_ids]
true_preds = [
    [id2label[p] for (p, l) in zip(pred, label) if l != -100]
    for pred, label in zip(preds, preds_output.label_ids)
]

print(classification_report(true_labels, true_preds, digits=3))

