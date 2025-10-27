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
log_path = "logs\\vgg16\\v2\\log_treino_foren.txt"
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

    # ========== TESTE 1: DIAGNÓSTICO GPU ==========
    print("\n" + "="*60)
    print("DIAGNÓSTICO GPU:")
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    print(f"Versão CUDA: {torch.version.cuda}")
    print(f"Número de GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU atual: {torch.cuda.current_device()}")
        print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
    print("="*60 + "\n")

    # ---------- seed ----------
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False  # False para melhor performance
    torch.backends.cudnn.benchmark = True       # True otimiza para GPU

    # ---------- foren ----------
    train_path_local = 'data\\foren\\treino'
    val_path_local   = 'data\\foren\\validacao'

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
    val_dataset   = datasets.ImageFolder(root=val_path_local, transform=transform)

    print(f"Número de imagens no dataset de treino: {len(train_dataset)}")
    print(f"Número de imagens no dataset de validação: {len(val_dataset)}")
    print(f"\nClasses detectadas no treino: {train_dataset.classes}")
    print(f"Mapeamento de classe para índice: {train_dataset.class_to_idx}")

    # ---------- dataloaders ----------
    batch_size = 256  # Aumentado para usar mais GPU
    num_workers = 4   # Carrega dados em paralelo
    pin_memory = True # Acelera transferência CPU -> GPU

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
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Congela feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Modifica classificador para 2 classes
    model.classifier[6] = nn.Linear(4096, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Memória GPU disponível: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("\nX AVISO: Rodando na CPU! Verifique sua instalação do PyTorch.")
        
    model = model.to(device)

    # ========== TESTE 2: FORÇA CRIAR TENSORES NA GPU ==========
    print("\n" + "="*60)
    print("TESTE DE ALOCAÇÃO NA GPU:")
    test_tensor = torch.randn(1000, 1000).to(device)
    test_result = test_tensor @ test_tensor
    print(f"Tensor de Teste criado em: {test_result.device}")
    print(f"Memória GPU após teste: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    del test_tensor, test_result
    torch.cuda.empty_cache()
    print("="*60 + "\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    # ---------- treino + validação ----------
    num_epochs = 100
    scaler = GradScaler(device='cuda' if device.type == 'cuda' else 'cpu')
    best_acc = -1.0
    bad = 0
    patience = 5
    best_path = "models\\vgg16\\V2\\model_foren.pt"
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    start_time = time.time()
    print(f"{'='*60}")
    print(f"Iniciando treinamento com batch_size={batch_size}, num_workers={num_workers}")
    print(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):
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
                gpu_mem_alloc = torch.cuda.memory_allocated() / 1024**3 if device.type == 'cuda' else 0
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3 if device.type == 'cuda' else 0
                print(f"Epoch {epoch}/{num_epochs} | Step {i}/{len(train_loader)} | Loss {(running/50):.4f} | GPU Alloc: {gpu_mem_alloc:.2f}GB | Reserved: {gpu_mem_reserved:.2f}GB")
                running = 0.0

        epoch_time = time.time() - epoch_start
        print(f"TEMPO DO EPOCH {epoch}: {epoch_time:.2f}s")

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
    print(f"Melhor acurácia: {best_acc:.4f}")
    print(f"Tempo total de treino: {minutes}m {seconds}s")
    print(f"{'='*60}")
    
    # Cleanup dos workers do DataLoader
    print("\nLimpando recursos...")
    del train_loader
    del val_loader
    torch.cuda.empty_cache()
    print("CLEANUP CONCLUÍDO.")
    
    _log_file.close()

if __name__ == '__main__':
    main()