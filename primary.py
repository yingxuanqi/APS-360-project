import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AgriDataset1(Dataset):

    def __init__(self, csv_path, rgb_dir, nir_dir, transform_rgb=None, transform_nir=None):
        self.data = pd.read_csv(csv_path)
        self.rgb_dir = rgb_dir
        self.nir_dir = nir_dir
        self.transform_rgb = transform_rgb
        self.transform_nir = transform_nir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["filename"]
        label = self.data.iloc[idx]["label"]

        rgb_path = os.path.join(self.rgb_dir, img_name)
        nir_path = os.path.join(self.nir_dir, img_name)

        rgb_image = Image.open(rgb_path).convert("RGB")
        nir_image = Image.open(nir_path).convert("L")   # NIR usually single channel

        if self.transform_rgb:
            rgb_image = self.transform_rgb(rgb_image)

        if self.transform_nir:
            nir_image = self.transform_nir(nir_image)

        # rgb_image: [3, 224, 224]
        # nir_image: [1, 224, 224]
        image = torch.cat((rgb_image, nir_image), dim=0)   # [4, 224, 224]

        label = torch.tensor(label, dtype=torch.long)

        return image, label


transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

transform_nir = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


train_dataset = AgriDataset1(
    csv_path=r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\train\labels.csv",
    rgb_dir=r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\train\images\rgb",
    nir_dir=r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\train\images\nir",
    transform_rgb=transform_rgb,
    transform_nir=transform_nir
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_dataset = AgriDataset1(
    csv_path=r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\val\labels.csv",
    rgb_dir=r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\val\images\rgb",
    nir_dir=r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\val\images\nir",
    transform_rgb=transform_rgb,
    transform_nir=transform_nir
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


class CNN4(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 3, padding=1)   # 4 channels now
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def get_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)   # [1, 2]

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[:, class_idx]
        score.backward()

        # [1, C, H, W] -> [C, H, W]
        activations = self.activations[0]
        gradients = self.gradients[0]

        # channel weights
        weights = gradients.mean(dim=(1, 2))   # [C]

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)

        # normalize to [0,1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy(), class_idx

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def tensor_to_rgb(img_tensor):
    """
    img_tensor: [4, H, W]
    only use first 3 channels for display
    """
    rgb = img_tensor[:3].detach().cpu().numpy()
    rgb = np.transpose(rgb, (1, 2, 0))  # CHW -> HWC

    rgb = rgb - rgb.min()
    if rgb.max() > 0:
        rgb = rgb / rgb.max()

    return rgb


def tensor_to_nir(img_tensor):
    """
    img_tensor: [4, H, W]
    4th channel is NIR
    """
    nir = img_tensor[3].detach().cpu().numpy()
    nir = nir - nir.min()
    if nir.max() > 0:
        nir = nir / nir.max()
    return nir


def show_gradcam(model, dataset, sample_idx=0, class_idx=1):
    """
    class_idx=1 means visualize abnormal class
    """
    model.eval()

    img, label = dataset[sample_idx]
    input_tensor = img.unsqueeze(0).to(device)

    grad_cam = GradCAM(model, model.conv2)
    cam, pred_class = grad_cam.generate(input_tensor, class_idx=class_idx)
    grad_cam.remove_hooks()

    rgb_img = tensor_to_rgb(img)
    nir_img = tensor_to_nir(img)

    H, W, _ = rgb_img.shape
    cam_resized = cv2.resize(cam, (W, H))

    # overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.6 * rgb_img + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)

    # prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred = logits.argmax(dim=1).item()

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(rgb_img)
    plt.title("RGB")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(nir_img, cmap="gray")
    plt.title("NIR")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(cam_resized, cmap="jet")
    plt.title(f"Grad-CAM\nfor class {class_idx}")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(overlay)
    plt.title(
        f"Overlay\nlabel={label}, pred={pred}\n"
        f"P(normal)={probs[0]:.3f}, P(abnormal)={probs[1]:.3f}"
    )
    plt.axis("off")

    plt.tight_layout()
    plt.show()







device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    model = CNN4().to(device)
    model.load_state_dict(torch.load("best_agri_cnn_4_channels.pth", map_location=device))
    model.eval()

    # 2. 可视化几张图
    print("Showing Grad-CAM...")
    for i in range(1):
        show_gradcam(model, val_dataset, sample_idx=i, class_idx=1)
    
"""    train_acc_list = []
    val_acc_list = []
    model = CNN4().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0

    for epoch in range(10):
        model.train()
        print(epoch)

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print("epoch done")

        train_acc = get_accuracy(model, train_loader, device)
        val_acc = get_accuracy(model, val_loader, device)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        print(f"Epoch {epoch+1}")
        print(f"Train Acc: {train_acc:.4f}")
        print(f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_agri_cnn_4_channels.pth")
            print("best model saved")


    import matplotlib.pyplot as plt

    epochs = range(1, len(train_acc_list)+1)

    plt.figure()

    plt.plot(epochs, train_acc_list, label="Train Accuracy")
    plt.plot(epochs, val_acc_list, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")

    plt.legend()

    plt.show()"""