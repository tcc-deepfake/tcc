import torch
import os
import sys
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report

# ---------- log ----------
log_path = "C:\\Users\\Thiago Borges\\Desktop\\teste\\tcc\\logs\\vgg16\\v1\\log_teste_foren.txt"
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
test_path_local = 'C:\\Users\\Thiago Borges\\Desktop\\teste\\tcc\\data\\foren\\teste'
df_path = 'C:\\Users\\Thiago Borges\\Desktop\\teste\\tcc\\data\\df\\teste'
best_path = 'C:\\Users\\Thiago Borges\\Desktop\\teste\\tcc\\models\\vgg16\\V1\\model_foren.pt'

# ---------- transforms ----------
# VGG16 - 224x224
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------- datasets ----------
test_dataset = datasets.ImageFolder(root=test_path_local, transform=transform)
df_dataset = datasets.ImageFolder(root=df_path, transform=transform)

print(f"Número de imagens (Foren): {len(test_dataset)}")
print("Foren classes:", test_dataset.class_to_idx)
print(f"Número de imagens (DF): {len(df_dataset)}")
print("DF classes   :", df_dataset.class_to_idx)

# ---------- dataloaders ----------
batch_size = 32
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
df_loader = DataLoader(df_dataset, batch_size=batch_size, shuffle=False)

# ---------- modelo ----------
# Carrega VGG16 com a mesma arquitetura do treino
model = models.vgg16(weights=None)  # Não carrega pesos pré-treinados
model.classifier[6] = nn.Linear(4096, 2)  # 2 classes: FAKE e REAL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU não detectada - usando CPU")

model.load_state_dict(torch.load(best_path, map_location=device))
model = model.to(device)
model.eval()

# ---------- teste Foren ----------
print("\n=== TESTE NA BASE FOREN ===")
correct = 0
total = 0
all_labels = []
all_predicted = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
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
print("\n=== TESTE NA BASE DEEPFAKE FACES ===")
df_correct = 0
df_total = 0
df_all_labels = []
df_all_predicted = []

with torch.no_grad():
    for images, labels in df_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.max(outputs, 1)[1]
        df_total += labels.size(0)
        df_correct += (predicted == labels).sum().item()
        df_all_labels.extend(labels.cpu().numpy())
        df_all_predicted.extend(predicted.cpu().numpy())

df_accuracy = 100 * df_correct / df_total
print(f"Acurácia no Teste (DeepfakeFaces): {df_accuracy:.2f}%")

df_target_names = [k for k, v in sorted(df_dataset.class_to_idx.items(), key=lambda item: item[1])]
df_report = classification_report(df_all_labels, df_all_predicted, target_names=df_target_names)
print(df_report)