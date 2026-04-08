import random
import shutil
from pathlib import Path

# =========================
# 1. 路径设置
# =========================
SOURCE_ROOT = Path(r"C:\Users\qiyin\supervised\Agriculture-Vision-2021\train")

TARGET_ROOT_1 = Path(r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\val")
TARGET_ROOT_2 = Path(r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\train")
TARGET_ROOT_3 = Path(r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\test")

N1 = 750   # val
N2 = 3500  # train
N3 = 750   # test
SEED = 42

LABEL_CLASSES = [
    "double_plant",
    "drydown",
    "endrow",
    "nutrient_deficiency",
    "planter_skip",
    "storm_damage",
    "water",
    "waterway",
    "weed_cluster"
]

# =========================
# 2. 原始目录
# =========================
rgb_dir = SOURCE_ROOT / "images" / "rgb"
nir_dir = SOURCE_ROOT / "images" / "nir"
boundaries_dir = SOURCE_ROOT / "boundaries"
labels_root = SOURCE_ROOT / "labels"

if not rgb_dir.exists():
    raise FileNotFoundError(f"没找到 RGB 文件夹: {rgb_dir}")

print("rgb_dir exists:", rgb_dir.exists())
print("nir_dir exists:", nir_dir.exists())
print("boundaries_dir exists:", boundaries_dir.exists())
print("labels_root exists:", labels_root.exists())

# =========================
# 3. 找所有 RGB 图片
# =========================
rgb_files = sorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))

if len(rgb_files) == 0:
    raise ValueError("rgb 文件夹里没有找到图片，请检查路径。")

total_needed = N1 + N2 + N3
if len(rgb_files) < total_needed:
    raise ValueError(f"总图片数只有 {len(rgb_files)}，但你需要 {total_needed} 张，不够。")

print(f"找到 RGB 图片总数: {len(rgb_files)}")
print(f"需要抽取总数: {total_needed}")

# =========================
# 4. 一次性无重复抽样
# =========================
random.seed(SEED)
all_selected = random.sample(rgb_files, total_needed)

subset1 = all_selected[:N1]                 # val
subset2 = all_selected[N1:N1 + N2]         # train
subset3 = all_selected[N1 + N2:N1 + N2 + N3]  # test

print(f"val 数量:   {len(subset1)}")
print(f"train 数量: {len(subset2)}")
print(f"test 数量:  {len(subset3)}")

# =========================
# 5. 工具函数
# =========================
def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        shutil.copy2(src, dst)
        return True
    return False

def find_nir_file(stem: str):
    nir_jpg = nir_dir / f"{stem}.jpg"
    nir_png = nir_dir / f"{stem}.png"
    if nir_jpg.exists():
        return nir_jpg
    if nir_png.exists():
        return nir_png
    return None

def setup_target_dirs(target_root: Path):
    target_rgb_dir = target_root / "images" / "rgb"
    target_nir_dir = target_root / "images" / "nir"
    target_boundaries_dir = target_root / "boundaries"
    target_labels_root = target_root / "labels"

    target_rgb_dir.mkdir(parents=True, exist_ok=True)
    target_nir_dir.mkdir(parents=True, exist_ok=True)
    target_boundaries_dir.mkdir(parents=True, exist_ok=True)

    for cls in LABEL_CLASSES:
        (target_labels_root / cls).mkdir(parents=True, exist_ok=True)

    return {
        "rgb": target_rgb_dir,
        "nir": target_nir_dir,
        "boundaries": target_boundaries_dir,
        "labels": target_labels_root
    }

def copy_sample_group(sample_list, target_root: Path, group_name="group"):
    target_dirs = setup_target_dirs(target_root)
    copied_count = 0

    for rgb_path in sample_list:
        stem = rgb_path.stem

        # 1) copy rgb
        shutil.copy2(rgb_path, target_dirs["rgb"] / rgb_path.name)

        # 2) copy nir
        nir_path = find_nir_file(stem)
        if nir_path is not None:
            shutil.copy2(nir_path, target_dirs["nir"] / nir_path.name)

        # 3) copy boundary
        boundary_path = boundaries_dir / f"{stem}.png"
        copy_if_exists(boundary_path, target_dirs["boundaries"] / boundary_path.name)

        # 4) copy labels/<class>/
        for cls in LABEL_CLASSES:
            label_path = labels_root / cls / f"{stem}.png"
            copy_if_exists(label_path, target_dirs["labels"] / cls / label_path.name)

        copied_count += 1

        if copied_count % 200 == 0:
            print(f"{group_name}: 已完成 {copied_count}/{len(sample_list)}")

    print(f"{group_name} 复制完成，共 {copied_count} 组样本")
    print(f"保存位置: {target_root}")

# =========================
# 6. 开始复制
# =========================
copy_sample_group(subset1, TARGET_ROOT_1, "val")
copy_sample_group(subset2, TARGET_ROOT_2, "train")
copy_sample_group(subset3, TARGET_ROOT_3, "test")

print("\n全部完成")