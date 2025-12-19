from singtown_ai import SingTownAIClient
from singtown_ai import stdout_watcher

client = SingTownAIClient()

@stdout_watcher(interval=1)
def on_stdout_write(content: str):
    client.log(content, end="")


from pathlib import Path
from rknn.api import RKNN
import os
import tarfile
import torch
import random

RUN_PATH = Path("../run")
IMAGES_PATH = Path("../dataset")/"TRAIN"
ONNX_PATH = RUN_PATH/'best.onnx'
RUN_PATH.mkdir(parents=True, exist_ok=True)

LABELS = client.task.project.labels
EXPORT_WIDTH = client.task.export_width
EXPORT_HEIGHT = client.task.export_height

print(f"CUDA available:  {torch.cuda.is_available()}")

rknn = RKNN()

rknn.config(mean_values=[123.675, 116.28, 103.53], std_values=[58.395, 58.395, 58.395], target_platform="rv1106")
ret = rknn.load_onnx(str(ONNX_PATH))
if ret != 0:
    raise Exception("Load model failed!")


with open(RUN_PATH/"dataset.txt", "w") as f:
    imgs = []
    for root, dirs, files in os.walk(IMAGES_PATH):
        for file in files:
            full_path = Path(root) / file
            imgs.append(str(full_path.resolve()))
    selected_imgs = random.sample(imgs, min(100, len(imgs)))
    for img in selected_imgs:
        f.write(img + "\n")

ret = rknn.build(do_quantization=True, dataset=RUN_PATH/"dataset.txt")
if ret != 0:
    raise Exception("Build model failed!")

ret = rknn.export_rknn(str(RUN_PATH/"best.rknn"))
if ret != 0:
    raise Exception("Export model failed!")

rknn.release()

with open(RUN_PATH/"labels.txt", "wb") as f:
    f.write("\n".join(LABELS).encode("utf-8"))

with tarfile.open(RUN_PATH/"mobilenet.tar", "w") as zipf:
    zipf.add(RUN_PATH/"labels.txt", arcname="labels.txt")
    zipf.add(RUN_PATH/"best.rknn", arcname="mobilenet.rknn")

client.upload_results_zip(RUN_PATH/"mobilenet.tar")
print("Finished")
