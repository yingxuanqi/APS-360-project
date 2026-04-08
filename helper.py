from pathlib import Path
import csv
from PIL import Image
import numpy as np

DATA_ROOT = Path(r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\test")

rgb_dir = DATA_ROOT / "images" / "rgb"
masks_root = DATA_ROOT / "labels"

MASK_CLASSES = [
    "water",
    "weed_cluster",

]

rgb_files = list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png"))

output_csv = DATA_ROOT / "labels.csv"

rows = []

for rgb_path in rgb_files:
    stem = rgb_path.stem

    abnormal = 0

    for cls in MASK_CLASSES:
        mask_path = masks_root / cls / f"{stem}.png"

        if mask_path.exists():
            mask = np.array(Image.open(mask_path))

            # 只要这个mask里有非零像素，就说明有异常
            if np.any(mask > 0):
                abnormal = 1
                break

    rows.append([rgb_path.name, abnormal])

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    writer.writerows(rows)

print(f"labels.csv 已生成: {output_csv}")
print(f"总样本数: {len(rows)}")
print(f"abnormal 数量: {sum(r[1] for r in rows)}")
print(f"normal 数量: {len(rows) - sum(r[1] for r in rows)}")