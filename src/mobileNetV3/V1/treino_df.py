import torch
import time
import timm
import os
import sys
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets, transforms
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from collections import Counter
import numpy as np

# ---------- log ----------
log_path = "logs/mobileNetV3/V1/log_treino_df.txt"
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

# ---------- df ----------
train_path_local = 'data/df/treino'
val_path_local   = 'data/df/validacao'

# ---------- transforms ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
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
val_dataset   = datasets.ImageFolder(root=val_path_local,   transform=transform)

print(f"Número de imagens no dataset de treino: {len(train_dataset)}")
print(f"Número de imagens no dataset de validação: {len(val_dataset)}")
print(f"\nClasses detectadas no treino: {train_dataset.classes}")
print(f"Mapeamento de classe para índice: {train_dataset.class_to_idx}")

# ---------- dataloaders ----------
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

# ---------- modelo ----------
model = timm.create_model('mobilenetv3_large_100', pretrained=True)
for param in model.parameters():
    param.requires_grad = False
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
model = model.to(device)

# ---------- class weights ----------
counts = Counter([y for _, y in train_dataset.samples])  # 0=fake,1=real
w = np.array([1.0 / counts[i] for i in range(len(counts))], dtype=np.float32)
w = w * (len(w) / w.sum())
class_weights = torch.tensor(w, device=device, dtype=torch.float32)
print("Class weights:", w)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# ---------- treino + validacao ----------
num_epochs = 3
scaler = GradScaler(device='cuda' if device.type == 'cuda' else 'cpu')
best_acc = -1.0
bad = 0
patience = 3
best_path = "models/mobileNetV3/V1/model_df.pt"
os.makedirs(os.path.dirname(best_path), exist_ok=True)

start_time = time.time()

for epoch in range(1, num_epochs + 1):
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

    if val_acc > best_acc:
        best_acc = val_acc
        bad = 0
        torch.save(model.state_dict(), best_path)
    else:
        bad += 1
        if bad >= patience:
            print("Early stopping.")
            break

elapsed = time.time() - start_time
print(f"Melhor acc: {best_acc:.4f}")
print(f"\nTempo total de treino: {int(elapsed//60)}m {int(elapsed%60)}s")
