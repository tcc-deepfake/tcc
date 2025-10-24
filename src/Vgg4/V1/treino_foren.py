import torch
import time
import os
import sys
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets, transforms
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

# ---------- log ----------
log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs", "Vgg4", "V1", "log_treino_foren.txt"))
os.makedirs(os.path.dirname(log_path), exist_ok=True)

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data); s.flush()
    def flush(self):
        for s in self.streams: s.flush()

_log_file = open(log_path, "w", encoding="utf-8", buffering=1) 
sys.stdout = _Tee(sys.stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)

print(f"Gravando log em: {log_path}")

# ---------- seed ----------
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------- Modelo VGG4.5M ----------
class VGG4point5M(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG4point5M, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*128, 256),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(256, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# ---------- foren ----------
train_path_local = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "data", "foren", "treino"))
val_path_local   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "data", "foren", "validacao"))

# ---------- transforms ----------
# VGG4.5M - 224x224
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# data augmentation no treino
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------- datasets ----------
train_dataset = datasets.ImageFolder(root=train_path_local, transform=train_transform)
val_dataset   = datasets.ImageFolder(root=val_path_local, transform=transform)

print(f"Número de imagens no dataset de treino: {len(train_dataset)}")
print(f"Número de imagens no dataset de validação: {len(val_dataset)}")
print(f"\nClasses detectadas no treino: {train_dataset.classes}")
print(f"Mapeamento de classe para índice: {train_dataset.class_to_idx}")

# ---------- dataloaders ----------
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

# ---------- modelo ----------
# Inicializa VGG4.5M (treinamento do zero)
model = VGG4point5M(num_classes=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# ---------- treino + validacao ----------
num_epochs = 100
scaler = GradScaler(device='cuda' if device.type == 'cuda' else 'cpu')
best_acc = -1.0
bad = 0
patience = 15
best_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "Vgg4", "V1", "model_foren.pt"))
os.makedirs(os.path.dirname(best_path), exist_ok=True)

start_time = time.time()

for epoch in range(1, num_epochs + 1):
    # ---- train ----
    model.train()
    running = 0.0
    for i, (images, labels) in enumerate(train_loader, 1):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.item()
        if i % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs} | Step {i}/{len(train_loader)} | Loss {(running/100):.4f}")
            running = 0.0

    # ---- validation ----
    model.eval()
    total = correct = 0
    with torch.no_grad(), autocast(device_type=device.type, enabled=(device.type == 'cuda')):
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total if total else 0.0
    print(f"[Val] Epoch {epoch} | Acc={val_acc:.4f}")

    # ---- early stopping + best model ----
    if val_acc > best_acc:
        best_acc = val_acc
        bad = 0
        torch.save(model.state_dict(), best_path)
        print(f"[OK] Melhor modelo salvo! Val Acc: {val_acc:.4f}")
    else:
        bad += 1
        print(f"Sem melhora há {bad} épocas")
        if bad >= patience:
            print("Early stopping.")
            break

end_time = time.time()
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print(f"\nMelhor acc: {best_acc:.4f}")
print(f"Tempo total de treino: {minutes}m {seconds}s")