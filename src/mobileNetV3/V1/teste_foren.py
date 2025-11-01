import torch
import os
import sys
import timm
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets, transforms
from sklearn.metrics import classification_report
from torch.amp.autocast_mode import autocast

# ---------- log ----------
log_path = "logs/mobileNetV3/V2/log_teste_foren.txt"
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

# ---------- bases ----------
test_path_local  = 'data/foren/teste'
df_path  = 'data/df/teste'
best_path = 'models/mobileNetV3/V2/model_foren.pt'

# ---------- transforms ----------
# MobileNetV3 - 224x224
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------- datasets ----------
test_dataset  = datasets.ImageFolder(root=test_path_local, transform=transform)
df_dataset = datasets.ImageFolder(root=df_path, transform=transform)

print(f"Número de imagens (Foren): {len(test_dataset)}")
print("Foren classes:", test_dataset.class_to_idx)   
print(f"Número de imagens (DF): {len(df_dataset)}")
print("DF classes   :", df_dataset.class_to_idx)     

# ---------- dataloaders ----------
batch_size = 32
num_workers = 4 
pin_memory = True 
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
df_loader = DataLoader(df_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)

# ---------- modelo ----------
model = timm.create_model('mobilenetv3_large_100', pretrained=True)

if hasattr(model, 'classifier'):
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2), # O Dropout estava em falta
        nn.Linear(in_features, 2)
    )
else:
    raise ValueError("MobileNetV3 não tem atributo classifier esperado.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(best_path, map_location=device))
model = model.to(device)
model.eval()

# ---------- teste Foren ----------
correct = 0
total = 0
all_labels = []
all_predicted = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True) 

        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            outputs = model(images) 
        predicted = torch.max(outputs, 1)[1]

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())

print(f"Acurácia no Teste (Foren): {100 * correct / total:.2f}%")
target_names = [k for k, v in sorted(test_dataset.class_to_idx.items(), key=lambda item: item[1])]
report = classification_report(all_labels, all_predicted, target_names=target_names)
print(report)

# ---------- teste DF ----------
df_correct = 0
df_total = 0
df_all_labels = []
df_all_predicted = []

with torch.no_grad():
    for images, labels in df_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True) 

        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            outputs = model(images) 
        predicted = torch.max(outputs, 1)[1]

        df_total += labels.size(0)
        df_correct += (predicted == labels).sum().item()
        df_all_labels.extend(labels.cpu().numpy())
        df_all_predicted.extend(predicted.cpu().numpy())

# Calcula e imprime as métricas finais
df_accuracy = 100 * df_correct / df_total
print(f"Acurácia no Teste (DeepfakeFaces): {df_accuracy:.2f}%")

# nome das classes
df_target_names = [k for k, v in sorted(df_dataset.class_to_idx.items(), key=lambda item: item[1])]
df_report = classification_report(df_all_labels, df_all_predicted, target_names=df_target_names)
print(df_report)
