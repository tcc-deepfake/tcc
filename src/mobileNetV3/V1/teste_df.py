import torch
import os
import sys
import timm
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets, transforms
from sklearn.metrics import classification_report

# ---------- log ----------
log_path = "logs/mobileNetV3/V1/log_teste_df.txt"
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
df_path         = 'data/df/teste'
foren_path      = 'data_old/foren/teste/person'
best_path       = 'models/mobileNetV3/V1/model_df.pt'

# ---------- transforms ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------- datasets ----------
df_dataset    = datasets.ImageFolder(root=df_path,    transform=transform)
foren_dataset = datasets.ImageFolder(root=foren_path, transform=transform)

print(f"Número de imagens (DF): {len(df_dataset)}")
print("DF classes:", df_dataset.class_to_idx)
print(f"Número de imagens (Foren): {len(foren_dataset)}")
print("Foren classes:", foren_dataset.class_to_idx)

# ---------- dataloaders ----------
batch_size = 32
df_loader    = DataLoader(df_dataset,    batch_size=batch_size, shuffle=False)
foren_loader = DataLoader(foren_dataset, batch_size=batch_size, shuffle=False)

# ---------- modelo ----------
model = timm.create_model('mobilenetv3_large_100', pretrained=True)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(best_path, map_location=device))
model = model.to(device)
model.eval()

# ---------- teste DF ----------
df_correct = df_total = 0
df_all_labels = []
df_all_predicted = []
with torch.no_grad():
    for images, labels in df_loader:
        images, labels = images.to(device), labels.to(device)
        predicted = model(images).argmax(1)
        df_total += labels.size(0)
        df_correct += (predicted == labels).sum().item()
        df_all_labels.extend(labels.cpu().numpy())
        df_all_predicted.extend(predicted.cpu().numpy())

print(f"Acurácia no Teste (DeepfakeFaces): {100 * df_correct / df_total:.2f}%")
df_target_names = [k for k, v in sorted(df_dataset.class_to_idx.items(), key=lambda item: item[1])]
print(classification_report(df_all_labels, df_all_predicted, target_names=df_target_names))

# ---------- teste Foren ----------
f_correct = f_total = 0
f_all_labels = []
f_all_predicted = []
with torch.no_grad():
    for images, labels in foren_loader:
        images, labels = images.to(device), labels.to(device)

        predicted = model(images).argmax(1)
        f_total += labels.size(0)
        f_correct += (predicted == labels).sum().item()
        f_all_labels.extend(labels.cpu().numpy())
        f_all_predicted.extend(predicted.cpu().numpy())

print(f"Acurácia no Teste (Foren): {100 * f_correct / f_total:.2f}%")
f_target_names = [k for k, v in sorted(foren_dataset.class_to_idx.items(), key=lambda item: item[1])]
print(classification_report(f_all_labels, f_all_predicted, target_names=f_target_names))
