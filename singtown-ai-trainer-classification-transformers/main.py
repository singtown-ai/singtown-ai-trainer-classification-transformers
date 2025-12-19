from singtown_ai import SingTownAIClient
from singtown_ai import stdout_watcher
from singtown_ai import export_class_folder

client = SingTownAIClient()

@stdout_watcher(interval=1)
def on_stdout_write(content: str):
    client.log(content, end="")

from pathlib import Path
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, TrainerCallback
from datasets import load_dataset

class SingTownAICallback(TrainerCallback):
    def __init__(self, client):
        self.client = client
        self.metrics = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.metrics.append(logs)
            self.client.update_metrics(self.metrics)

RUN_PATH = Path("../run")
DATASET_PATH = Path("../dataset").absolute().resolve()
RUN_PATH.mkdir(parents=True, exist_ok=True)

LABELS = client.task.project.labels
MODEL_NAME = client.task.model_name
EPOCHS = client.task.epochs
BATCH_SIZE = client.task.batch_size
LEARNING_RATE = client.task.learning_rate
EXPORT_WIDTH = client.task.export_width
EXPORT_HEIGHT = client.task.export_height

export_class_folder(client, DATASET_PATH)

train_dataset = load_dataset(str(DATASET_PATH/"TRAIN"))["train"]
valid_dataset = load_dataset(str(DATASET_PATH/"VALID"))["train"]
test_dataset = load_dataset(str(DATASET_PATH/"TEST"))["train"]


processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    local_files_only=True
)

def preprocess(example):
    inputs = processor(example["image"], return_tensors="pt")
    example["pixel_values"] = inputs["pixel_values"][0]
    return example

train_dataset = train_dataset.map(preprocess)
valid_dataset = valid_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["pixel_values", "label"])
valid_dataset.set_format(type="torch", columns=["pixel_values", "label"])
test_dataset.set_format(type="torch", columns=["pixel_values", "label"])

id2label = {i: label for i, label in enumerate(LABELS)}
label2id = {label: i for i, label in enumerate(LABELS)}

model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir="./trainer_output",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["label"] for item in batch]
    return {
        "pixel_values": torch.stack(pixel_values),
        "labels": torch.tensor(labels),
    }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1)
    labels = torch.tensor(labels)
    acc = (preds == labels).float().mean().item()
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks=[SingTownAICallback(client)],
)
trainer.train()


model.to("cpu")
model.eval()
input_tensor = torch.rand((1, 3, EXPORT_HEIGHT, EXPORT_WIDTH), dtype=torch.float32).to("cpu")
torch.onnx.export(
    model, 
    input_tensor, 
    RUN_PATH / "best.onnx",
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
)