import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import matplotlib.pyplot as plt

# =========================
# 1. Config
# =========================
TRAIN_CSV = r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\train\labels.csv"
TRAIN_RGB_DIR = r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\train\images\rgb"
TRAIN_NIR_DIR = r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\train\images\nir"

VAL_CSV = r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\val\labels.csv"
VAL_RGB_DIR = r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\val\images\rgb"
VAL_NIR_DIR = r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\val\images\nir"

MODEL_PATH = "best_agri_cnn_3_channels.pth"

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
NUM_CLASSES = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2. Dataset
# =========================
class AgriDatasetRGB(Dataset):
    def __init__(self, csv_path, rgb_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.rgb_dir = rgb_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row["filename"]
        label = int(row["label"])

        rgb_path = os.path.join(self.rgb_dir, img_name)
        rgb_image = Image.open(rgb_path).convert("RGB")

        if self.transform is not None:
            rgb_image = self.transform(rgb_image)

        return rgb_image, torch.tensor(label, dtype=torch.long), img_name
    


# =========================
# 4. DataLoaders
# =========================


val_dataset_rgb = AgriDatasetRGB(
    csv_path=VAL_CSV,
    rgb_dir=VAL_RGB_DIR,
    transform=transform_rgb
)
val_loader_rgb = DataLoader(val_dataset_rgb, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# 5. Model
# =========================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 224 -> 112
        x = self.pool(F.relu(self.conv2(x)))   # 112 -> 56

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
# =========================
# 6. Grad-CAM
# =========================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)  # [1,2]
        score = output[:, class_idx]
        score.backward()

        activations = self.activations[0]  # [C,H,W]
        gradients = self.gradients[0]      # [C,H,W]

        weights = gradients.mean(dim=(1, 2))  # [C]

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy()

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def tensor_to_rgb(img_tensor):
    rgb = img_tensor[:3].cpu().numpy()
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = rgb - rgb.min()
    if rgb.max() > 0:
        rgb = rgb / rgb.max()
    return rgb


def tensor_to_nir(img_tensor):
    nir = img_tensor[3].cpu().numpy()
    nir = nir - nir.min()
    if nir.max() > 0:
        nir = nir / nir.max()
    return nir


def resize_cam_nearest(cam, target_h, target_w):
    """
    Avoid cv2 dependency.
    Simple resize using torch interpolate.
    """
    cam_tensor = torch.tensor(cam, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
    cam_resized = F.interpolate(
        cam_tensor,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False
    )
    return cam_resized[0, 0].cpu().numpy()


def show_gradcam_rgb(model, dataset, sample_idx, class_idx=1, save_path=None):
    img, label, img_name = dataset[sample_idx]
    input_tensor = img.unsqueeze(0).to(DEVICE)

    gradcam = GradCAM(model, model.conv2)
    cam = gradcam.generate(input_tensor, class_idx=class_idx)
    gradcam.remove_hooks()

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = logits.argmax(dim=1).item()

    # RGB image only
    rgb_img = img.cpu().numpy()
    rgb_img = np.transpose(rgb_img, (1, 2, 0))
    rgb_img = rgb_img - rgb_img.min()
    if rgb_img.max() > 0:
        rgb_img = rgb_img / rgb_img.max()

    H, W, _ = rgb_img.shape
    cam_resized = resize_cam_nearest(cam, H, W)

    heatmap = plt.cm.jet(cam_resized)[:, :, :3]
    overlay = 0.6 * rgb_img + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(rgb_img)
    plt.title(f"RGB\n{img_name}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cam_resized, cmap="jet")
    plt.title(f"Grad-CAM\nclass {class_idx}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(
        f"Overlay\nlabel={label.item()}, pred={pred}\n"
        f"P(normal)={probs[0].item():.3f}, P(abnormal)={probs[1].item():.3f}"
    )
    plt.axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def find_samples_by_label(dataset, target_label, max_count=10):
    indices = []
    for i in range(len(dataset)):
        _, label, _ = dataset[i]
        if label.item() == target_label:
            indices.append(i)
        if len(indices) >= max_count:
            break
    return indices


def find_correct_abnormal_samples(model, dataset, max_count=10):
    model.eval()
    indices = []

    with torch.no_grad():
        for i in range(len(dataset)):
            img, label, _ = dataset[i]
            if label.item() != 1:
                continue

            logits = model(img.unsqueeze(0).to(DEVICE))
            pred = logits.argmax(dim=1).item()

            if pred == 1:
                indices.append(i)

            if len(indices) >= max_count:
                break

    return indices


def find_wrong_samples(model, dataset, max_count=10):
    model.eval()
    indices = []

    with torch.no_grad():
        for i in range(len(dataset)):
            img, label, _ = dataset[i]
            logits = model(img.unsqueeze(0).to(DEVICE))
            pred = logits.argmax(dim=1).item()

            if pred != label.item():
                indices.append(i)

            if len(indices) >= max_count:
                break

    return indices


# =========================
# 7. Main
# =========================
if __name__ == "__main__":
    print("Using device:", DEVICE)

    best_model = CNN().to(DEVICE)
    best_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    best_model.eval()

    print("\nSome label=1 samples:", find_samples_by_label(val_dataset_rgb, 1, max_count=10))
    print("Some correct abnormal samples:", find_correct_abnormal_samples(best_model, val_dataset_rgb, max_count=10))
    print("Some wrong samples:", find_wrong_samples(best_model, val_dataset_rgb, max_count=10))


    example_indices = find_wrong_samples(best_model, val_dataset_rgb, max_count=100)

    if len(example_indices) == 0:
        print("\nNo correctly predicted abnormal samples found in the first scan.")
        print("Try visualizing a manual index from val_dataset.")
    else:
        for idx in example_indices:
            show_gradcam_rgb(best_model, val_dataset_rgb, sample_idx=idx, class_idx=1)
