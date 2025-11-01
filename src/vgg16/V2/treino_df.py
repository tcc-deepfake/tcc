import torch
import time
import os
import sys
import numpy as np

# adiciona a raiz do projeto ao python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# faz o import absoluto funcionar
from utils.model_compress import aplica_pruning, limpa_pesos, check_sparsity

from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets, transforms, models
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from collections import Counter

# ---------- log ----------
log_path = "logs\\Vgg16\\V2\\log_treino_df.txt" 
os.makedirs(os.path.dirname(log_path), exist_ok=True)

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data); s.flush()
    def flush(self):
        for s in self.streams: s.flush()

def main():
    _log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, _log_file)
    sys.stderr = _Tee(sys.stderr, _log_file)

    print(f"Gravando LOGs em: {log_path}")
    print(f"CUDA disponível: {torch.cuda.is_available()}")

    # ---------- seed ----------
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True 

    # ---------- paths ----------
    train_path_local = 'data\\df\\treino'
    val_path_local   = 'data\\df\\validacao'
    
    path_v1 = "models\\Vgg16\\V1\\model_df.pt"
    save_path = "models\\Vgg16\\V2\\model_df.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ---------- transforms ----------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9,1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ---------- datasets ----------
    train_dataset = datasets.ImageFolder(root=train_path_local, transform=train_transform)
    val_dataset   = datasets.ImageFolder(root=val_path_local, transform=transform)
    print(f"Dataset de treino: {len(train_dataset)} imagens")
    print(f"Dataset de validação: {len(val_dataset)} imagens")

    # ---------- dataloaders ----------
    batch_size = 256 
    num_workers = 4 
    pin_memory = True

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    )

    # ---------- modelo ----------
    model = models.vgg16(weights=None) 

    model.classifier[6] = nn.Linear(4096, 2)

    state_v1 = torch.load(path_v1, map_location='cpu')
    model.load_state_dict(state_v1, strict=True)

    model = aplica_pruning(model, prune_amount=0.2, incluir_convs=True, verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    model = model.to(device)
    
    for name, param in model.named_parameters():
        if name.startswith("features.24") or name.startswith("features.25") \
           or name.startswith("features.26") or name.startswith("features.27") \
           or name.startswith("features.28") or name.startswith("features.29") \
           or name.startswith("classifier"):
            param.requires_grad = True
        else:
            param.requires_grad = False


    # ---------- class weights ----------
    counts = Counter([y for _, y in train_dataset.samples]) 
    w = np.array([1.0 / counts[i] for i in range(len(counts))], dtype=np.float32)
    w = w * (len(w) / w.sum())
    class_weights = torch.tensor(w, device=device, dtype=torch.float32)
    print(f"Class weights: {w}\n")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=1e-4, momentum=0.9, weight_decay=1e-4)

    num_epochs = 3
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs) 

    # ---------- treino + validação ----------
    scaler = GradScaler(device='cuda' if device.type == 'cuda' else 'cpu')
    best_acc = -1.0
    bad = 0
    patience = 3 # Paciência menor para fine-tuning

    start_time = time.time()
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        
        # ---- train ----
        model.train()
        running = 0.0
        epoch_start = time.time()
        
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
            
            if i % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Step {i}/{len(train_loader)} | Loss {(running/50):.4f}")
                running = 0.0

        epoch_time = time.time() - epoch_start
        print(f"TEMPO DO EPOCH {epoch+1}: {epoch_time:.2f}s")

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
        print(f"[Val] Epoch {epoch+1} | Acc={val_acc:.4f}")

        scheduler.step()

        # ---- early stopping + best model ----
        if val_acc > best_acc:
            best_acc = val_acc
            bad = 0
            torch.save(model.state_dict(), save_path) 
            print(f"O NOVO MELHOR MODELO SALVO (acc={best_acc:.4f})")
        else:
            bad += 1
            print(f"X SEM MELHORA ({bad}/{patience})")
            if bad >= patience:
                print("EARLY STOPPING ATIVADO.")
                break
        
        print("-" * 60)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print(f"\n{'='*60}")
    print(f"Melhor acurácia (pós-pruning): {best_acc:.4f}")
    print(f"Tempo total de fine-tuning: {minutes}m {seconds}s")
    print(f"{'='*60}")
    
    model.load_state_dict(torch.load(save_path, map_location=device))
    model = limpa_pesos(model)
    torch.save(model.state_dict(), save_path)
    print(f"Modelo com pruning permanente salvo em: {save_path}")
    
    # Cleanup
    print("\nLimpando recursos...")
    del train_loader
    del val_loader
    torch.cuda.empty_cache()
    print("CLEANUP CONCLUÍDO.")
    
    _log_file.close()

if __name__ == '__main__':
    main()

