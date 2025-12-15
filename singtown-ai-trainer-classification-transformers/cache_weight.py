# cache ~/.cache/huggingface/hub

from transformers import AutoImageProcessor, AutoModelForImageClassification
models = [
    "timm/mobilenetv3_large_100.ra_in1k",
]

for model in models:
    AutoImageProcessor.from_pretrained(model)
    AutoModelForImageClassification.from_pretrained(model)