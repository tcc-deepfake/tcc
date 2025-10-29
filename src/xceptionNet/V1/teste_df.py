import torch
import time
import os
import sys
import timm
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets, transforms
from sklearn.metrics import classification_report

# ---------- log ----------
log_path = "logs/xceptionNet/V1/log_teste_df.txt"
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
df_path  = 'data/df/teste'
foren_path  = 'data_old/foren/teste/person'
best_path = "models/xceptionNet/V1/model_df.pt"

# ---------- transform ----------
transform = transforms.Compose([
    transforms.Resize((299, 299)), # Resize 299x299 pro Xception
    transforms.ToTensor(),         # Imagem para PyTorch Tensor
    # Normalização com média e desvio da ImageNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------- datasets ----------
test_dataset  = datasets.ImageFolder(root=foren_path, transform=transform)
df_dataset = datasets.ImageFolder(root=df_path, transform=transform)

print(f"Número de imagens (Foren): {len(test_dataset)}")
print("Foren classes:", test_dataset.class_to_idx)   
print(f"Número de imagens (DF): {len(df_dataset)}")
print("DF classes   :", df_dataset.class_to_idx)

# ---------- dataloaders ----------
batch_size = 32
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
df_loader = DataLoader(df_dataset, batch_size=batch_size, shuffle=False)


# ---------- modelo ----------
model = timm.create_model('xception', pretrained=True)

if hasattr(model, 'fc'):
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 2)
    )
elif hasattr(model, 'head'):
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 2)
    )
else:
    raise RuntimeError("Layer de classificação não encontrada.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(best_path, map_location=device))
model = model.to(device)
model.eval()

# ---------- teste DF ----------
df_correct = 0
df_total = 0
df_all_labels = []
df_all_predicted = []

start_time_df = time.time()

with torch.no_grad():
    for images, labels in df_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        predicted = torch.max(outputs, 1)[1]

        df_total += labels.size(0)
        df_correct += (predicted == labels).sum().item()
        df_all_labels.extend(labels.cpu().numpy())
        df_all_predicted.extend(predicted.cpu().numpy())

end_time_df = time.time()
elapsed_time_df = end_time_df - start_time_df
minutes_df = int(elapsed_time_df // 60)
seconds_df = int(elapsed_time_df % 60)
print(f"Tempo total de inferência: {minutes_df}m {seconds_df}s")

# Calcula e imprime as métricas finais
df_accuracy = 100 * df_correct / df_total
print(f"Acurácia no Teste (DeepfakeFaces): {df_accuracy:.2f}%")

# nome das classes
df_target_names = [k for k, v in sorted(df_dataset.class_to_idx.items(), key=lambda item: item[1])]
df_report = classification_report(df_all_labels, df_all_predicted, target_names=df_target_names)
print(df_report)

# ---------- teste Foren ----------
correct = 0
total = 0
all_labels = []
all_predicted = []

start_time_foren = time.time()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.max(outputs, 1)[1]
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())

end_time_foren = time.time()
elapsed_time_foren = end_time_foren - start_time_foren
minutes_foren = int(elapsed_time_foren // 60)
seconds_foren = int(elapsed_time_foren % 60)
print(f"Tempo total de inferência: {minutes_foren}m {seconds_foren}s")


print(f"Acurácia no Teste (Foren): {100 * correct / total:.2f}%")

target_names = [k for k, v in sorted(test_dataset.class_to_idx.items(), key=lambda item: item[1])]
report = classification_report(all_labels, all_predicted, target_names=target_names)
print(report)

