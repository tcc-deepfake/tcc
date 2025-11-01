import torch
import time
import os
import sys
from utils.model_compress import check_sparsity
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report
from torch.amp.autocast_mode import autocast

# ---------- log ----------
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data); s.flush()
    def flush(self):
        for s in self.streams: s.flush()

def main():
    log_path = "logs\\vgg16\\V2\\log_teste_df.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
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
    df_path = 'data\\df\\teste'
    foren_path = 'data_old\\foren\\teste\\person'
    best_path = 'models\\vgg16\\V2\\model_df.pt'

    # ---------- transforms ----------
    # VGG16 - 224x224
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ---------- datasets ----------
    df_dataset = datasets.ImageFolder(root=df_path, transform=transform)
    foren_dataset = datasets.ImageFolder(root=foren_path, transform=transform)

    print(f"Número de imagens (DF): {len(df_dataset)}")
    print("DF classes:", df_dataset.class_to_idx)
    print(f"Número de imagens (Foren): {len(foren_dataset)}")
    print("Foren classes:", foren_dataset.class_to_idx)

    # ---------- dataloaders ----------
    batch_size = 128
    num_workers = 4 
    pin_memory = False 
    df_loader = DataLoader(df_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
    foren_loader = DataLoader(foren_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)

    # ---------- modelo ----------
    model = models.vgg16(weights=None)  

    model.classifier[6] = nn.Linear(4096, 2)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model = model.to(device).float()
    model.eval()

    check_sparsity(model, verbose=True)
    
    # ---------- teste DF ----------
    print("\n=== TESTE NA BASE DEEPFAKE FACES ===")
    df_correct = df_total = 0
    df_all_labels = []
    df_all_predicted = []

    start_time_df = time.time()

    with torch.inference_mode():
        for images, labels in df_loader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device)

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

    print(f"Acurácia no Teste (DeepfakeFaces): {100 * df_correct / df_total:.2f}%")
    df_target_names = [k for k, v in sorted(df_dataset.class_to_idx.items(), key=lambda item: item[1])]
    df_report = classification_report(df_all_labels, df_all_predicted, target_names=df_target_names)
    print(df_report)

    print(f"Tempo total de inferência: {minutes_df}m {seconds_df}s")
    
    # ---------- teste Foren ----------
    print("\n=== TESTE NA BASE FOREN ===")
    f_correct = f_total = 0
    f_all_labels = []
    f_all_predicted = []

    start_time_foren = time.time()

    with torch.inference_mode():
        for images, labels in foren_loader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device)

            outputs = model(images) 
            predicted = torch.max(outputs, 1)[1]

            f_total += labels.size(0)
            f_correct += (predicted == labels).sum().item()
            f_all_labels.extend(labels.cpu().numpy())
            f_all_predicted.extend(predicted.cpu().numpy())

    end_time_foren = time.time()
    elapsed_time_foren = end_time_foren - start_time_foren
    minutes_foren = int(elapsed_time_foren // 60)
    seconds_foren = int(elapsed_time_foren % 60)

    print(f"Acurácia no Teste (Foren): {100 * f_correct / f_total:.2f}%")

    f_target_names = [k for k, v in sorted(foren_dataset.class_to_idx.items(), key=lambda item: item[1])]
    f_report = classification_report(f_all_labels, f_all_predicted, target_names=f_target_names)
    print(f_report)

    print(f"Tempo total de inferência: {minutes_foren}m {seconds_foren}s")
    
    # Cleanup
    print("\nLimpando recursos...")
    del df_loader
    del foren_loader
    torch.cuda.empty_cache()
    print("CLEANUP CONCLUÍDO.")
    
    _log_file.close()

if __name__ == '__main__':
    main()

