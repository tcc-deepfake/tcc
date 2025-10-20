import argparse, os, shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--src_root", default="data_old/df", help="pasta origem com faces_224 e metadata.csv")
ap.add_argument("--dst_root", default="data/df",    help="pasta destino para treino/validacao/teste")
ap.add_argument("--move", action="store_true",      help="mover em vez de copiar")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

SRC = Path(args.src_root).resolve()
DST = Path(args.dst_root).resolve()

faces_dir = SRC / "faces_224"
meta_csv  = SRC / "metadata.csv"

assert faces_dir.is_dir(), f"Origem não encontrada: {faces_dir}"
assert meta_csv.is_file(), f"metadata.csv não encontrado: {meta_csv}"

# limpar destino
for sub in ("treino", "validacao", "teste"):
    for cls in ("real", "fake"):
        d = DST / sub / cls
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

# ler metadata
df = pd.read_csv(meta_csv)
vid_col = next(c for c in df.columns if c.lower() in ("videoname", "video", "filename"))
lbl_col = next(c for c in df.columns if c.lower() in ("label", "class", "target"))

# indexar arquivos por stem
index = {}
for p in faces_dir.rglob("*"):
    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
        index[p.stem] = p

paths, labels = [], []
miss = 0
for _, row in df.iterrows():
    stem = Path(str(row[vid_col]).replace(".mp4", "")).stem
    src = index.get(stem)
    if not src:
        miss += 1
        continue
    y = str(row[lbl_col]).strip().lower()
    y = 1 if y == "fake" else 0  # 0=real, 1=fake
    paths.append(src)
    labels.append(y)

print(f"\nTotal pares válidos: {len(paths)}  | Sem correspondência: {miss}")
print("Distribuição geral:", Counter(labels))

# split estratificado 80/10/10
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
train_idx, hold_idx = next(sss1.split(paths, labels))

paths = pd.Series(paths)
labels = pd.Series(labels)

paths_hold = paths.iloc[hold_idx].to_numpy()
labels_hold = labels.iloc[hold_idx].to_numpy()

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed)
val_rel, test_rel = next(sss2.split(paths_hold, labels_hold))
val_idx  = [hold_idx[i] for i in val_rel]
test_idx = [hold_idx[i] for i in test_rel]

splits = {
    "treino": train_idx,
    "validacao": val_idx,
    "teste": test_idx,
}

# mover/copiar arquivos
op = shutil.move if args.move else shutil.copy2
counts = {"treino": [0,0], "validacao":[0,0], "teste":[0,0]}

for split_name, idxs in splits.items():
    for i in idxs:
        src = paths.iloc[i]
        y = labels.iloc[i]
        cls = "fake" if y == 1 else "real"
        dst = DST / split_name / cls / (src.stem + src.suffix.lower())
        op(src, dst)
        counts[split_name][y] += 1

# --- prints finais detalhados ---
print("\n--- Distribuição por conjunto ---")
for split_name, (real_count, fake_count) in counts.items():
    total = real_count + fake_count
    p_real = (real_count / total) * 100 if total else 0
    p_fake = (fake_count / total) * 100 if total else 0
    print(f"{split_name.upper():10} | total={total:6d} | real={real_count:6d} ({p_real:5.1f}%) | fake={fake_count:6d} ({p_fake:5.1f}%)")

total_real = sum(v[0] for v in counts.values())
total_fake = sum(v[1] for v in counts.values())
print(f"\nTOTAL FINAL -> real={total_real} | fake={total_fake}")
print(f"Saída gerada em: {DST}")
print("\nVerifique as pastas 'treino', 'validacao' e 'teste' em:", DST)
