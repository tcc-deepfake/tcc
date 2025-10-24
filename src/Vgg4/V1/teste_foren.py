import torch
import os
import sys
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets, transforms
from sklearn.metrics import classification_report

# ---------- log ----------
log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs", "Vgg4", "V1", "log_teste_foren.txt"))
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

# ---------- Modelo VGG4.5M ----------
class VGG4point5M(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG4point5M, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*128, 256),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(256, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# ---------- bases ----------
test_path_local = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "data", "foren", "teste"))
df_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "data", "df", "teste"))
best_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "Vgg4", "V1", "model_foren.pt"))

# ---------- transforms ----------
# VGG4.5M - 224x224
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
# Carrega VGG4.5M com a mesma arquitetura do treino
model = VGG4point5M(num_classes=2)

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
