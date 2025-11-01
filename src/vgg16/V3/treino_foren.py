import os
import sys
import torch
import torch.nn as nn
import time
from utils.model_compress import aplica_quantizacao_estatica
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ---------- log ----------
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data); s.flush()
    def flush(self):
        for s in self.streams: s.flush()

def main():
    log_path = "logs/vgg16/V3/log_treino_foren.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, _log_file)
    sys.stderr = _Tee(sys.stderr, _log_file)

    print(f"Gravando log em: {log_path}")

    # ---------- paths ----------
    path_v2 = "models/vgg16/V2/model_foren.pt"
    path_v3 = "models/vgg16/V3/model_foren.pth" # .pth para JIT
    val_path_local = 'data/foren/validacao' 
    os.makedirs(os.path.dirname(path_v3), exist_ok=True)

    # ---------- modelo ----------
    model = models.vgg16(weights=None) 
        
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, 2)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.load_state_dict(torch.load(path_v2, map_location=device))
    model.eval()

    # ---------- dataloader (para calibração) ----------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = datasets.ImageFolder(root=val_path_local, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)
    print(f"Dataset de calibração (val) carregado: {len(val_dataset)} imagens.")

    # ---------- quantização estática ----------
    start_time = time.time()

    model_quantized = aplica_quantizacao_estatica(
        model, 
        val_loader, 
        input_size=(1, 3, 224, 224), 
        verbose=True
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    # ---------- salvando o modelo quantizado ----------
    print("\nSalvando Modelo Quantizado...")
    model_scripted = torch.jit.script(model_quantized)
    torch.jit.save(model_scripted, path_v3)

    print(f"Modelo salvo em: {path_v3}")
    print(f"Tempo Total de Quantização: {minutes}m {seconds}s")

    # Cleanup
    print("\nLimpando Recursos...")
    del val_loader
    torch.cuda.empty_cache()
    print("CLEANUP CONCLUÍDO.")
    
    _log_file.close()

if __name__ == '__main__':
    main()
    
