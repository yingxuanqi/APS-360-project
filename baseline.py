import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class AgriDataset(Dataset):

    def __init__(self, csv_path, img_dir, transform=None):

        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_name = self.data.iloc[idx]["filename"]
        label = self.data.iloc[idx]["label"]

        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
  

from torchvision import transforms
from torch.utils.data import DataLoader
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_dataset = AgriDataset(
    csv_path=r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\train\labels.csv",
    img_dir=r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\train\images\rgb",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_dataset = AgriDataset(
    csv_path=r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\val\labels.csv",
    img_dir=r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\val\images\rgb",
    transform=transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)






import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(16,32,3,padding=1)

        self.fc1 = nn.Linear(32*56*56,128)
        self.fc2 = nn.Linear(128,2)

    def forward(self,x):
 
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0),-1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x





def get_accuracy(model, dataloader, device):

    model.eval()              # 进入 evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():     # 不计算梯度，速度更快

        for imgs, labels in dataloader:

            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total






import torch.optim as optim


device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    train_acc_list = []
    val_acc_list = []
    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(),lr=1e-4)

    best_val_acc = 0

    for epoch in range(10):
        print(epoch+1)
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
            torch.save(model.state_dict(), "best_agri_cnn_3_channels.pth")
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

    plt.show()