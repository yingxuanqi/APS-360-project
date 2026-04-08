import torch
from baseline import CNN, get_accuracy,AgriDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from primary import  AgriDataset1,CNN4
#from goodgoodgood import CNN4


def get_confusion(model, loader, device):
    model.eval()

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            pred = torch.argmax(logits, dim=1)

            for p, t in zip(pred, y):
                if p == 1 and t == 1:
                    TP += 1
                elif p == 0 and t == 0:
                    TN += 1
                elif p == 1 and t == 0:
                    FP += 1
                elif p == 0 and t == 1:
                    FN += 1

    return TP, TN, FP, FN
device = "cuda" if torch.cuda.is_available() else "cpu"


CNN_model = CNN()
CNN_model = CNN_model.to(device)

CNN_model.load_state_dict(torch.load("best_agri_cnn_3_channels.pth", map_location=device))

CNN_model.eval()

print("CNN3_model loaded")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

test_dataset = AgriDataset(
    csv_path=r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\test\labels.csv",
    img_dir=r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\test\images\rgb",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)
TP, TN, FP, FN = get_confusion(CNN_model, test_loader, device)

print("\n=== Confusion Matrix ===")
print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\n=== Metrics ===")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("CNN channel 3:",get_accuracy(CNN_model, test_loader, device))


transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

transform_nir = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_dataset = AgriDataset1(
    csv_path=r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\test\labels.csv",
    rgb_dir=r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\test\images\rgb",
    nir_dir=r"C:\Users\qiyin\OneDrive\Documents\Desktop\APS360\dataset\test\images\nir",
    transform_rgb=transform_rgb,
    transform_nir=transform_nir
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


CNN_model = CNN4()
CNN_model = CNN_model.to(device)

CNN_model.load_state_dict(torch.load("best_agri_cnn_4ch_clean.pth", map_location=device))

CNN_model.eval()

print("CNN4_model loaded")
TP, TN, FP, FN = get_confusion(CNN_model, test_loader, device)

print("\n=== Confusion Matrix ===")
print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\n=== Metrics ===")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("CNN channel 4:",get_accuracy(CNN_model, test_loader, device))