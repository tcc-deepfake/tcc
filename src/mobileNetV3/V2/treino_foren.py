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
from torchvision.transforms import InterpolationMode
from utils.augmentation import RandomJPEGReencode, RandomCenterCropResize

# ---------- log ----------
log_path = "logs/mobileNetV3/V2/log_treino_foren.txt"
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
train_path_local = 'data/foren/treino'
val_path_local   = 'data/foren/validacao'

# ---------- transforms ----------
# MobileNetV3 - 224x224
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# data augumentation no treino
train_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
    RandomCenterCropResize(scale_min=0.9, scale_max=1.0, out_size=(224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
    transforms.RandomApply([RandomJPEGReencode(qmin=60, qmax=90, p=1.0)], p=0.5),
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
batch_size = 64 
num_workers = 4 
pin_memory = True 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)

# ---------- modelo ----------
model = timm.create_model('mobilenetv3_large_100', pretrained=True)

# ---------- subsititui camada classificadora ----------
if hasattr(model, 'classifier'):
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 2)
    )
else:
    raise ValueError("MobileNetV3 não tem atributo classifier esperado.")

# ---------- verifica gpu ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
model = model.to(device)

# ---------- congela camada ----------
for p in model.parameters():
    p.requires_grad = False

# libera só o classifier na fase 1
for p in model.classifier.parameters():
    p.requires_grad = True


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-3, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

scaler = GradScaler(device='cuda' if device.type == 'cuda' else 'cpu')

# ---------- treino + validacao ----------
num_epochs = 15
best_acc = -1.0
bad = 0
patience = 3
best_path = "models/mobileNetV3/V2/model_foren.pt"
os.makedirs(os.path.dirname(best_path), exist_ok=True)

start_time = time.time()

for epoch in range(1, num_epochs + 1):
    # ---- train ----   
    if epoch == 5:
        for name, param in model.named_parameters():
            if any(tag in name for tag in ['classifier', 'conv_head', 'blocks.5', 'blocks.6']):
                param.requires_grad = True
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(num_epochs - epoch + 1))

    model.train()
    running = 0.0
    for i, (images, labels) in enumerate(train_loader, 1):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, labels)

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

    scheduler.step()

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
