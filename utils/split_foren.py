import argparse, os, shutil
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

IMG_EXT = {".jpg",".jpeg",".png",".bmp",".webp"}

ap = argparse.ArgumentParser()
ap.add_argument("--src_root", default="data_old/foren", help="tem subpastas treino/validacao/teste/person/...")
ap.add_argument("--dst_root", default="data/foren",     help="gera treino/validacao/teste/{real,fake}")
ap.add_argument("--move", action="store_true",          help="mover em vez de copiar")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

SRC = Path(args.src_root).resolve()
DST = Path(args.dst_root).resolve()

def list_images_with_labels(subdir: Path):
    base = subdir / "person"
    assert base.is_dir(), f"não achei {base}"
    paths, labels = [], []
    for cls_dir in sorted([d for d in base.iterdir() if d.is_dir()]):
        name = cls_dir.name.lower()
        if "fake" in name or name.startswith("1_"):
            y = 1
        else:
            y = 0  # assume real para 0_* 
        for p in cls_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                paths.append(p)
                labels.append(y)
    return paths, labels

def split_80_10_10(paths, labels, seed):
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, hold_idx = next(sss1.split(paths, labels))
    # dentro do 20%, metade para val e metade para teste
    hold_paths = [paths[i] for i in hold_idx]
    hold_labels = [labels[i] for i in hold_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_rel, test_rel = next(sss2.split(hold_paths, hold_labels))
    val_idx  = [hold_idx[i] for i in val_rel]
    test_idx = [hold_idx[i] for i in test_rel]
    return train_idx, val_idx, test_idx

def counts(labels):
    c = Counter(labels)
    return c.get(0,0), c.get(1,0)

# listar e dividir cada subset original
orig_sets = ["treino","validacao","teste"]
parts = {}  # nome - dict(indices)
for name in orig_sets:
    paths, labels = list_images_with_labels(SRC/name)
    r,f = counts(labels)
    total = len(labels)
    pr = (r/total*100) if total else 0; pf = (f/total*100) if total else 0
    print(f"[ORIG {name}] total={total} | real={r} ({pr:.1f}%) | fake={f} ({pf:.1f}%)")
    tr, va, te = split_80_10_10(paths, labels, args.seed)
    parts[name] = {
        "paths": paths,
        "labels": labels,
        "train_idx": tr,
        "val_idx": va,
        "test_idx": te,
    }
    for tag, idxs in [("80%", tr), ("10%_val", va), ("10%_test", te)]:
        rr, ff = counts([labels[i] for i in idxs])
        tt = len(idxs); pr = rr/tt*100 if tt else 0; pf = ff/tt*100 if tt else 0
        print(f"  - {name} {tag:8} | n={tt:6d} | real={rr:6d} ({pr:5.1f}%) | fake={ff:6d} ({pf:5.1f}%)")

# concatenar: 80% de cada - treino, 10% de cada - validacao, 10% de cada - teste
final_splits = {"treino":[], "validacao":[], "teste":[]}
for name in orig_sets:
    item = parts[name]
    P, Y = item["paths"], item["labels"]
    final_splits["treino"]    += [(P[i], Y[i], name) for i in item["train_idx"]]
    final_splits["validacao"] += [(P[i], Y[i], name) for i in item["val_idx"]]
    final_splits["teste"]     += [(P[i], Y[i], name) for i in item["test_idx"]]

# limpar destino e criar pastas
for split in ("treino","validacao","teste"):
    for cls in ("real","fake"):
        d = DST/split/cls
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

op = shutil.move if args.move else shutil.copy2
global_counts = {k:[0,0] for k in ("treino","validacao","teste")}

for split, items in final_splits.items():
    for src, y, origin in items:
        cls = "fake" if y==1 else "real"
        # prefixa com o subset de origem para evitar colisões
        dst = (DST/split/cls)/(f"{origin}__{src.stem}{src.suffix.lower()}")
        op(src, dst)
        global_counts[split][y] += 1

# prints 
print("\n--- DISTRIBUIÇÃO FINAL ---")
for split,(r_cnt,f_cnt) in global_counts.items():
    tot = r_cnt+f_cnt
    pr = r_cnt/tot*100 if tot else 0
    pf = f_cnt/tot*100 if tot else 0
    print(f"{split.upper():10} | total={tot:6d} | real={r_cnt:6d} ({pr:5.1f}%) | fake={f_cnt:6d} ({pf:5.1f}%)")

print(f"\nSaída: {DST}")
print("Estrutura: data/foren/{treino,validacao,teste}/{real,fake}")
