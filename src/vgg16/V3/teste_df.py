import torch
import time
import os
import sys
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report

# ---------- log ----------
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data); s.flush()
    def flush(self):
        for s in self.streams: s.flush()

def main():
    log_path = "logs/vgg16/V3/log_teste_df.txt"
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
    test_path_local = 'data_old/foren/teste/person'
    df_path = 'data/df/teste'
    best_path = "models/vgg16/V3/model_df.pth"

    # ---------- Transform ----------
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
    print("DF classes    :", df_dataset.class_to_idx)

    # ---------- dataloaders ----------
    batch_size = 128
    num_workers = 4
    pin_memory = False 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
    df_loader = DataLoader(df_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)

    # ---------- modelo ----------
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = torch.jit.load(best_path, map_location=device)
    model.eval()

    # ---------- teste DF ----------
    print("\n=== TESTE NA BASE DEEPFAKE FACES ===")
    df_correct = 0
    df_total = 0
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

    df_accuracy = 100 * df_correct / df_total
    print(f"Acurácia no Teste (DeepfakeFaces): {df_accuracy:.2f}%")

    df_target_names = [k for k, v in sorted(df_dataset.class_to_idx.items(), key=lambda item: item[1])]
    df_report = classification_report(df_all_labels, df_all_predicted, target_names=df_target_names)
    print(df_report)

    print(f"Tempo total de inferência (DF): {minutes_df}m {seconds_df}s")

    # ---------- teste Foren ----------
    print("\n=== TESTE NA BASE FOREN ===")
    foren_correct = 0
    foren_total = 0
    foren_all_labels = []
    foren_all_predicted = []

    start_time_foren = time.time()

    with torch.inference_mode(): 
        for images, labels in test_loader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device)

            outputs = model(images)
            predicted = torch.max(outputs, 1)[1]

            foren_total += labels.size(0)
            foren_correct += (predicted == labels).sum().item()
            foren_all_labels.extend(labels.cpu().numpy())
            foren_all_predicted.extend(predicted.cpu().numpy())

    end_time_foren = time.time()
    elapsed_time_foren = end_time_foren - start_time_foren
    minutes_foren = int(elapsed_time_foren // 60)
    seconds_foren = int(elapsed_time_foren % 60)

    foren_accuracy = 100 * foren_correct / foren_total
    print(f"Acurácia no Teste (Foren): {foren_accuracy:.2f}%")

    foren_target_names = [k for k, v in sorted(test_dataset.class_to_idx.items(), key=lambda item: item[1])]
    foren_report = classification_report(foren_all_labels, foren_all_predicted, target_names=foren_target_names)
    print(foren_report)

    print(f"Tempo total de inferência (Foren): {minutes_foren}m {seconds_foren}s")

    # Cleanup
    print("\nLimpando recursos...")
    del test_loader
    del df_loader
    print("CLEANUP CONCLUÍDO.")
    
    _log_file.close()

if __name__ == '__main__':
    main()
    
