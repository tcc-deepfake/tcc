import torch
import time
import os
import sys
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets, transforms, models
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

# ---------- log ----------
log_path = "C:\\Users\\Thiago Borges\\Desktop\\teste\\tcc\\logs\\vgg16\\v1\\log_treino_foren.txt"
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

# ---------- foren ----------
train_path_local = 'C:\\Users\\Thiago Borges\\Desktop\\teste\\tcc\\data\\foren\\treino'
val_path_local   = 'C:\\Users\\Thiago Borges\\Desktop\\teste\\tcc\\data\\foren\\validacao'

# ---------- transforms ----------
# VGG16 - 224x224
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
# Carrega VGG16 com pesos pré-treinados
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# Congela os parâmetros do feature extractor
for param in model.features.parameters():
    param.requires_grad = False

# Modifica o classificador para 2 classes
model.classifier[6] = nn.Linear(4096, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# ---------- treino + validacao ----------
num_epochs = 3
scaler = GradScaler(device='cuda' if device.type == 'cuda' else 'cpu')
best_acc = -1.0
bad = 0
patience = 3
best_path = "C:\\Users\\Thiago Borges\\Desktop\\teste\\tcc\\models\\vgg16\\V1\\model_foren.pt"
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
    else:
        bad += 1
        if bad >= patience:
            print("Early stopping.")
            break

end_time = time.time()
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print(f"Melhor acc: {best_acc:.4f}")
print(f"\nTempo total de treino: {minutes}m {seconds}s")