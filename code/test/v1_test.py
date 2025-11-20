from ultralytics import YOLO
from pathlib import Path
import glob, math, cv2
import matplotlib.pyplot as plt

best_weights = Path("experiments") / "football_det_y11n_v1" / "weights" / "best.pt"
model = YOLO(str(best_weights))

# custom_dir = Path("data") / "interim" / "bundesliga_frames"
# image_paths = sorted(glob.glob(str(custom_dir / "*.jpg")))
# print("Found", len(image_paths), "custom test images")

# out_root = Path("experiments") / "football_det_y11n_v1" / "custom_predictions"
# out_root.mkdir(parents=True, exist_ok=True)

# results = model.predict(
#     source=image_paths,
#     imgsz=960,
#     conf=0.25,
#     save=True,
#     project=str(out_root),
#     name="custom_preview",
#     classes=[0],
#     show_conf=False
# )

# print("Predictions saved in:", out_root / "custom_preview")