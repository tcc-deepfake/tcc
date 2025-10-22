import os
import sys
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets, transforms
import timm
import time

from utils.model_compress import optimize_model # Pra aplicar prune e quant.

# ---------- log ----------
log_path = "logs/xceptionNet/V2/log_treino_foren.txt"
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
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform = transforms.Compose([
    transforms.Resize((299, 299)), # Resize 299x299 pro Xception
    transforms.ToTensor(),         # Imagem para PyTorch Tensor
    #transforms.Lambda(lambda x: x - transforms.GaussianBlur(5)(x)),  # residual
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
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

# ---------- modelo ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('xception', pretrained=True)
optimized_xception = optimize_model(model, model_name='xception', device=device)
# -------------------------------------------------------------------
# 4. Freeze ALL layers (safe baseline)
# -------------------------------------------------------------------
for param in model.parameters():
    param.requires_grad = False

# # Then unfreeze the final feature extraction layers and classifier head
# for name, param in model.named_parameters():
#     if any(key in name for key in ['block12', 'block13', 'block14', 'conv4', 'bn4', 'fc', 'head']):
#         param.requires_grad = True

# -------------------------------------------------------------------
# 5. Replace classifier head for binary classification (2 classes)
# -------------------------------------------------------------------
if hasattr(model, 'fc'):
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    trainable_params = model.fc.parameters()

elif hasattr(model, 'head'):
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, 2)
    trainable_params = model.head.parameters()
else:
    raise ValueError("Layer de classificação não encontrado.")

# -------------------------------------------------------------------
# Modelo pra GPU se disponível
# -------------------------------------------------------------------
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
model = model.to(device)

print("Parametros treináveis:")
for name, p in model.named_parameters():
    if p.requires_grad:
        print(f"  {name}")

# -------------------------------------------------------------------
# Loss, optimizer, scheduler
# -------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(trainable_params, lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# ---------- treino + validacao ----------
best_val_acc = 0.0
patience = 5        # Paciência de 5 épocas
bad_epochs = 0      # Contador pra early stopping
save_path = "models/xceptionNet/V2/model_foren.pt"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# ---------- épocas ----------
num_epochs = 10

start_time = time.time()
# -----------------------------
# Treino
# -----------------------------
for epoch in range(num_epochs):
    
    print(f"\nEPOCH {epoch+1}/{num_epochs}")
    print("-" * 30)

    model.train()
    train_loss, train_correct, total = 0.0, 0, 0

    # for images, labels in tqdm(train_loader, desc="Treinando", leave=False):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = train_correct / total
    train_loss = train_loss / total

    # -----------------------------
    # Validação
    # -----------------------------
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():

        # for images, labels in tqdm(val_loader, desc="Validação", leave=False):
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    val_loss = val_loss / val_total

    # -----------------------------
    # SCHEDULER STEP
    # -----------------------------
    scheduler.step()
    # -----------------------------
    # Resumo da época
    # -----------------------------
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
    # -----------------------------
    # Early stopping
    # -----------------------------
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
