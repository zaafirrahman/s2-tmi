"""
01_extract_features.py
----------------------
Ekstraksi fitur 768-dim (token [CLS]) dari 4 backbone ViT-Base yang di-freeze:
  imgnet16 / imgnet32 : torchvision (ViT_B_16 / ViT_B_32 Weights.DEFAULT, input 224)
  hf16     / hf32     : HuggingFace google/vit-base-patch16-224 / google/vit-base-patch32-384
Dijalankan SEKALI. Hasilnya dipakai ulang di 02_run_replicates.py.

Kenapa boleh di-cache (sudah dicek dari notebook skripsi):
  - backbone di-freeze, hanya head yang dilatih
  - transform deterministik (TANPA augmentasi acak). Catatan: notebook HF lama memakai
    RandomHorizontalFlip, dan transform itu ikut terpakai di test set. Di sini flip dibuang.
  - dropout backbone = 0 (torchvision default 0.0; HF config 0.0, dicek lewat assert)

Semua 1200 citra = isi folder 80-20/train + 80-20/test. Split asli skripsi tidak dipakai;
split dibuat ulang di level SAMPEL di skrip 02.
"""
import os
import re
from collections import Counter, defaultdict

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# ====== KONFIGURASI (sesuaikan path Kaggle) ======
ROOT = "/kaggle/input/arabica-beans/arabica_beans/80-20"
PRED_ROOT = "/kaggle/input/prediction-arabica-class/pred_class"  # 60 citra prediksi (mutu1..mutu6/P1A.jpg ...)
OUT_DIR = "/kaggle/working/features"
BATCH = 32
TV_BACKBONES = {
    "imgnet16": (torchvision.models.vit_b_16, torchvision.models.ViT_B_16_Weights.DEFAULT),
    "imgnet32": (torchvision.models.vit_b_32, torchvision.models.ViT_B_32_Weights.DEFAULT),
}
HF_BACKBONES = {
    "hf16": "google/vit-base-patch16-224",
    "hf32": "google/vit-base-patch32-384",
}
# "native" = mean/std milik model HF (0.5/0.5), "imagenet" = 0.485.. seperti notebook skripsi
HF_NORM = "native"
# Pola nama file contoh: 1T-01.jpg -> mutu 1, sampel "T", foto ke-01
FNAME = re.compile(r"^(\d)([A-Za-z]+)-(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
# =================================================

os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


def collect_images():
    """Gabungkan train+test dari 80-20, parse ID sampel dari nama file, lalu validasi."""
    items = []
    for split in ("train", "test"):
        ds = datasets.ImageFolder(os.path.join(ROOT, split))
        assert ds.classes == [f"mutu{i}" for i in range(1, 7)], ds.classes
        items += ds.samples  # (path, label)
    items.sort(key=lambda t: t[0])  # urutan deterministik

    paths, labels, samples = [], [], []
    bad = []
    for p, y in items:
        m = FNAME.match(os.path.basename(p))
        if not m or int(m.group(1)) != y + 1:
            bad.append(os.path.basename(p))
            continue
        paths.append(p)
        labels.append(y)
        samples.append(f"{m.group(1)}{m.group(2).upper()}")  # misal "1T"
    if bad:
        raise ValueError(
            f"{len(bad)} nama file tidak cocok pola <mutu><huruf>-<nomor>.jpg, contoh: {bad[:5]}. "
            "Kabari aku format nama filenya supaya regex disesuaikan."
        )

    # ---- validasi struktur: harapannya 6 kelas x 20 sampel x 10 foto = 1200 ----
    per_class = defaultdict(set)
    for y, s in zip(labels, samples):
        per_class[y].add(s)
    photos = Counter(samples)
    print(f"Total citra       : {len(paths)} (harapan 1200)")
    print(f"Duplikat nama file: {len(paths) - len(set(map(os.path.basename, paths)))}")
    print("Sampel per kelas  :", {f"mutu{c+1}": len(v) for c, v in sorted(per_class.items())})
    print("Foto per sampel   :", dict(Counter(photos.values())), "(harapan {10: 120})")
    return paths, np.array(labels), np.array(samples)


def collect_pred():
    """Citra prediksi (set eksternal). Return (paths, y) atau None jika folder tidak ada."""
    if not os.path.isdir(PRED_ROOT):
        print(f"PRED_ROOT tidak ditemukan ({PRED_ROOT}); set prediksi dilewati")
        return None
    ds = datasets.ImageFolder(PRED_ROOT)
    assert ds.classes == [f"mutu{i}" for i in range(1, 7)], ds.classes
    items = sorted(ds.samples, key=lambda t: t[0])
    print(f"Citra prediksi: {len(items)} (harapan 60)")
    return [p for p, _ in items], np.array([y for _, y in items])


class PathDataset(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def embed(fwd, tf, paths):
    """fwd: fungsi batch-gambar -> fitur (B, 768)."""
    dl = DataLoader(PathDataset(paths, tf), batch_size=BATCH, shuffle=False, num_workers=2)
    out = []
    for i, xb in enumerate(dl):
        out.append(fwd(xb.to(device)).cpu().numpy().astype(np.float32))
        if i % 10 == 0:
            print(f"  batch {i+1}/{len(dl)}")
    return np.concatenate(out)


def save(name, tf, fwd, paths, y, samples, pred):
    print(f"\n[{name}] transform:\n{tf}")
    X = embed(fwd, tf, paths)
    np.savez_compressed(os.path.join(OUT_DIR, f"{name}.npz"),
                        X=X, y=y, sample=samples, path=np.array(paths))
    if pred is not None:  # transform SAMA dengan training (bukan Resize 224 langsung seperti notebook lama)
        Xp = embed(fwd, tf, pred[0])
        np.savez_compressed(os.path.join(OUT_DIR, f"{name}_pred.npz"),
                            X=Xp, y=pred[1], path=np.array(pred[0]))
    with open(os.path.join(OUT_DIR, f"{name}_transform.txt"), "w") as f:
        f.write(repr(tf))
    print(f"[{name}] fitur {X.shape} tersimpan")


def extract_tv(name, ctor, weights, paths, y, samples, pred):
    model = ctor(weights=weights)
    model.heads = nn.Identity()  # keluaran = token [CLS] 768-dim setelah LayerNorm akhir
    model.eval().to(device)
    save(name, weights.transforms(), model, paths, y, samples, pred)


def extract_hf(name, repo, paths, y, samples, pred):
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    proc = AutoImageProcessor.from_pretrained(repo)
    size = proc.size.get("height") or proc.size.get("shortest_edge")
    mean, std = (proc.image_mean, proc.image_std) if HF_NORM == "native" \
        else ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf = transforms.Compose([                      # tanpa RandomHorizontalFlip
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    model = AutoModelForImageClassification.from_pretrained(repo).eval().to(device)
    assert model.config.hidden_dropout_prob == 0 and model.config.attention_probs_dropout_prob == 0
    base = model.base_model  # ViTModel; last_hidden_state sudah melewati LayerNorm akhir

    def fwd(xb):
        return base(pixel_values=xb).last_hidden_state[:, 0]  # token [CLS], input classifier HF

    save(name, tf, fwd, paths, y, samples, pred)


if __name__ == "__main__":
    paths, y, samples = collect_images()
    pred = collect_pred()
    for name, (ctor, w) in TV_BACKBONES.items():
        extract_tv(name, ctor, w, paths, y, samples, pred)
    for name, repo in HF_BACKBONES.items():
        extract_hf(name, repo, paths, y, samples, pred)
