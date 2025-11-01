import os
import sys
import torch
import torch.nn as nn
import timm
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.model_compress import aplica_quantizacao_estatica

# ---------- log ----------
log_path = "logs/xceptionNet/V3/log_treino_foren.txt"
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

# ---------- paths ----------
path_v2 = "models/xceptionNet/V2/model_foren.pt"
path_v3 = "models/xceptionNet/V3/model_foren.pth" # .pth para JIT
val_path_local = 'data/foren/validacao' 
os.makedirs(os.path.dirname(path_v3), exist_ok=True)

# ---------- modelo ----------
model = timm.create_model('xception', pretrained=False) 
if hasattr(model, 'fc'):
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, 2))
else:
    raise ValueError("Layer de classificação não encontrado.")

model.load_state_dict(torch.load(path_v2, map_location='cpu'))
model.eval()

# ---------- dataloader (para calibração) ----------
transform = transforms.Compose([
    transforms.Resize((299, 299)),
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
    input_size=(1, 3, 299, 299), 
    verbose=True
)

end_time = time.time()

# ---------- salvando o modelo quantizado ----------
print("Salvando modelo quantizado")
model_scripted = torch.jit.script(model_quantized)
torch.jit.save(model_scripted, path_v3)

elapsed_time = end_time - start_time
print(f"Modelo salvo em: {path_v3}")

