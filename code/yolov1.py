from roboflow import Roboflow
from ultralytics import YOLO
from pathlib import Path
from dotenv import load_dotenv
import os, glob, math
import cv2
import matplotlib.pyplot as plt


# project_root = Path("..").resolve()
env_path = ".env"
load_dotenv(env_path)
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise RuntimeError("ROBOFLOW_API_KEY not found")
print("API key loaded from .env")


WORKSPACE = "footballltracking"
PROJECT   = "football-updated-g0fwo-u67ns"
VERSION   = 1 

rf = Roboflow(api_key=api_key)
workspace = rf.workspace(WORKSPACE)
project   = workspace.project(PROJECT)
dataset   = project.version(VERSION).download("yolov8")

print("Dataset location:", dataset.location)


data_dir = Path(dataset.location)
data_yaml = data_dir / "data.yaml"

print("data.yaml path:", data_yaml)
# print("\n===== data.yaml contents =====\n")
# print(data_yaml.read_text())


model = YOLO("yolo11n.pt")

results = model.train(
    data=str(data_yaml),
    epochs=40,
    imgsz=1920,
    batch=16,
    project=str("experiments"),
    name="football_det_y11n_v1",
    cos_lr=True,
)

# metrics = model.val()
# print(metrics)

test_metrics = model.val(
    data=str(data_yaml),
    split="test",
)

names = model.names
print("Test mAP50-95:", test_metrics.box.map)
print("Test mAP50:   ", test_metrics.box.map50)

print("\nPer-class mAP50:")
for i, m in enumerate(test_metrics.box.maps):
    print(f"  {i} ({names[i]}): {m:.3f}")

