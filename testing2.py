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

VAL_CSV = r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\val\labels.csv"
VAL_RGB_DIR = r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\val\images\rgb"

TEST_CSV = r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\test\labels.csv"
TEST_RGB_DIR = r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\test\images\rgb"

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
class AgriDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        required_cols = {"filename", "label"}
        if not required_cols.issubset(set(self.data.columns)):
            raise ValueError(f"CSV must contain columns: {required_cols}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row["filename"]
        label = int(row["label"])

        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"RGB image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), img_name


# =========================
# 3. Transform
# =========================
transform_rgb = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# =========================
# 4. DataLoaders
# =========================
train_dataset = AgriDataset(
    csv_path=TRAIN_CSV,
    img_dir=TRAIN_RGB_DIR,
    transform=transform_rgb
)

val_dataset = AgriDataset(
    csv_path=VAL_CSV,
    img_dir=VAL_RGB_DIR,
    transform=transform_rgb
)

test_dataset = AgriDataset(
    csv_path=TEST_CSV,
    img_dir=TEST_RGB_DIR,
    transform=transform_rgb
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =========================
# 5. Model
# =========================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 输入从 4 通道改成 3 通道
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 224 -> 112 -> 56
        # 之后 crop 4:-4 => 56 -> 48
        self.fc1 = nn.Linear(32 * 48 * 48, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [B,16,112,112]
        x = self.pool(F.relu(self.conv2(x)))   # [B,32,56,56]

        # 去掉边缘一圈
        x = x[:, :, 4:-4, 4:-4]                # [B,32,48,48]

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =========================
# 6. Training utilities
# =========================
def get_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels, _ in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for imgs, labels, _ in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)

    return running_loss / len(dataloader.dataset)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, (imgs, labels, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        if batch_idx % 10 == 0:
            print(
                f"[Epoch {epoch+1}] Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}",
                flush=True
            )

    return running_loss / len(dataloader.dataset)


# =========================
# 7. Test utility
# =========================
def evaluate_test_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels, _ in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {acc:.4f}")
    return acc


# =========================
# 8. Grad-CAM
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

        output = self.model(input_tensor)
        score = output[:, class_idx]
        score.backward()

        activations = self.activations[0]   # [C,H,W]
        gradients = self.gradients[0]       # [C,H,W]

        weights = gradients.mean(dim=(1, 2))

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
    rgb = img_tensor.cpu().numpy()
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = rgb - rgb.min()
    if rgb.max() > 0:
        rgb = rgb / rgb.max()
    return rgb


def resize_cam(cam, target_h, target_w):
    cam_tensor = torch.tensor(cam, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    cam_resized = F.interpolate(
        cam_tensor,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False
    )
    return cam_resized[0, 0].cpu().numpy()


def show_gradcam(model, dataset, sample_idx, class_idx=1, save_path=None):
    img, label, img_name = dataset[sample_idx]
    input_tensor = img.unsqueeze(0).to(DEVICE)

    gradcam = GradCAM(model, model.conv2)
    cam = gradcam.generate(input_tensor, class_idx=class_idx)
    gradcam.remove_hooks()

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = logits.argmax(dim=1).item()

    rgb_img = tensor_to_rgb(img)
    H, W, _ = rgb_img.shape
    cam_resized = resize_cam(cam, H, W)

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


# =========================
# 9. Helper functions
# =========================
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
# 10. Main
# =========================
if __name__ == "__main__":
    print("Using device:", DEVICE)
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))

    # 先检查 dataset 输出
    img, label, name = train_dataset[0]
    print("Loaded sample:", name, label.item(), img.shape)   # 应该是 [3,224,224]

    model = CNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    best_val_acc = 0.0

    # # -------- Training --------
    # for epoch in range(NUM_EPOCHS):
    #     train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
    #     val_loss = evaluate_loss(model, val_loader, criterion, DEVICE)

    #     train_acc = get_accuracy(model, train_loader, DEVICE)
    #     val_acc = get_accuracy(model, val_loader, DEVICE)

    #     train_loss_list.append(train_loss)
    #     val_loss_list.append(val_loss)
    #     train_acc_list.append(train_acc)
    #     val_acc_list.append(val_acc)

    #     print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    #     print(f"Train Loss: {train_loss:.4f}")
    #     print(f"Val Loss:   {val_loss:.4f}")
    #     print(f"Train Acc:  {train_acc:.4f}")
    #     print(f"Val Acc:    {val_acc:.4f}")

    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         torch.save(model.state_dict(), MODEL_PATH)
    #         print("Best model saved.")

    # # -------- Plot learning curves --------
    # epochs = range(1, NUM_EPOCHS + 1)

    # plt.figure(figsize=(12, 5))

    # plt.subplot(1, 2, 1)
    # plt.plot(epochs, train_loss_list, label="Train Loss")
    # plt.plot(epochs, val_loss_list, label="Val Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Loss Curve")
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(epochs, train_acc_list, label="Train Acc")
    # plt.plot(epochs, val_acc_list, label="Val Acc")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.title("Accuracy Curve")
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

    # -------- Load best model --------
    print("\nLoading best model...")
    best_model = CNN().to(DEVICE)
    best_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    best_model.eval()

    # -------- Test accuracy --------
    evaluate_test_accuracy(best_model, test_loader, DEVICE)

    # -------- Suggest useful samples --------
    print("\nSome label=1 samples:", find_samples_by_label(val_dataset, 1, max_count=10))
    print("Some correct abnormal samples:", find_correct_abnormal_samples(best_model, val_dataset, max_count=10))
    print("Some wrong samples:", find_wrong_samples(best_model, val_dataset, max_count=10))

    # -------- Example Grad-CAM visualization --------
    example_indices = find_correct_abnormal_samples(best_model, val_dataset, max_count=10)

    if len(example_indices) == 0:
        print("\nNo correctly predicted abnormal samples found.")
    else:
        for idx in example_indices[:3]:
            show_gradcam(best_model, val_dataset, sample_idx=idx, class_idx=1)