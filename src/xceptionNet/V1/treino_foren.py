import os
import sys
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets, transforms
import timm
import time
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

# ---------- log ----------
log_path = "logs/xceptionNet/V1/log_treino_foren.txt"
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

# data augumentation no treino
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(299, scale=(0.9,1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1), 
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform = transforms.Compose([
    transforms.Resize((299, 299)), # Resize 299x299 pro Xception
    transforms.ToTensor(),         # Imagem para PyTorch Tensor
    # Normalização com média e desvio da ImageNet
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
model = timm.create_model('xception', pretrained=True)

# ---------- congela camada ----------
for name, param in model.named_parameters():
    if any(layer in name for layer in ['fc']):
        param.requires_grad = True
    else:
        param.requires_grad = False

# ---------- subsititui camada classificadora ----------
if hasattr(model, 'fc'):
    in_features = model.fc.in_features
    # Colocando dropout para ver se melhora accuracy
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 2)
    )
    trainable_params = model.fc.parameters()
elif hasattr(model, 'head'):
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, 2)
    trainable_params = model.head.parameters()
else:
    raise ValueError("Layer de classificação não encontrado.")

# ---------- verifica gpu ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
model = model.to(device)

# ---------- loss, optimizer, scheduler ----------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10) 

scaler = GradScaler(device=device.type if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda'))

# ---------- treino + validacao ----------
best_val_acc = 0.0
patience = 5        # Paciência de 5 épocas
bad_epochs = 0      # Contador pra early stopping
save_path = "models/xceptionNet/V1/model_foren.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# ---------- épocas ----------
num_epochs = 10

start_time = time.time()

# ---------- treino ----------
for epoch in range(num_epochs):

    print(f"\nEPOCH {epoch+1}/{num_epochs}")
    print("-" * 30)

    model.train()
    train_loss, train_correct, total = 0.0, 0, 0
    running = 0.0 # Para log a cada 100 steps

    for i, (images, labels) in enumerate(train_loader, 1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True) 
        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * images.size(0)
        running += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Log a cada 100 steps (padrão df.py)
        if i % 100 == 0:
            print(f"[Train] epoch {epoch+1}/{num_epochs} step {i}/{len(train_loader)} loss_mean={running/100:.4f}")
            running = 0.0

    train_acc = train_correct / total
    train_loss /= total

    # -------- validação --------
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    running_val = 0.0 

    with torch.no_grad():
        for j, (images, labels) in enumerate(val_loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            running_val += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            if j % 100 == 0:
                print(f"[Val]   epoch {epoch+1}/{num_epochs} step {j}/{len(val_loader)} loss_mean={running_val/100:.4f}")
                running_val = 0.0

    val_acc = val_correct / val_total
    val_loss /= val_total

    scheduler.step()

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        bad_epochs = 0
        torch.save(model.state_dict(), save_path)
        print(f" Best model saved to {save_path} (Val Acc: {val_acc:.4f})")
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print("Early stopping.")
            break

end_time = time.time()

elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print(f"Melhor acc: {best_val_acc:.4f}")
print(f"\nTempo total de treino: {minutes}m {seconds}s")
