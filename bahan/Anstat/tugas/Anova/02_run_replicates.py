"""
02_run_replicates.py
--------------------
Desain: faktorial 2 x 2 x 4 dalam Rancangan Acak Kelompok (RAK).
  Faktor A : sumber pre-training (ImageNet torchvision, HuggingFace)
  Faktor B : patch size (16, 32)
  Faktor C : jumlah data latih (sampel per kelas) -> 16, 14, 10, 8  (setara 80/70/50/40 % dari 20 sampel)
  Blok     : replikasi r = 1..R. Tiap blok = satu pengacakan sampel fisik ke train/test.
             Di dalam satu blok, SEMUA perlakuan memakai test set yang sama (4 sampel/kelas = 240 citra),
             dan train set bersarang (8 sampel pertama termasuk di 10, 14, 16).
  Respon   : akurasi test (epoch terakhir, bukan epoch terbaik, supaya test set tidak ikut "memilih" model)
             + pred_acc = akurasi pada 60 citra prediksi eksternal (respon sekunder)

Split dilakukan di level SAMPEL FISIK (bukan per citra) supaya foto dari sampel yang sama
tidak bocor antara train dan test.

Protokol training head IDENTIK untuk semua perlakuan (supaya faktor A tidak tercampur dengan
perbedaan epoch/lr/dropout; di skripsi HF: 50 epoch, lr 1e-4, tanpa dropout head, ada flip):
  Dropout(0.1) + Linear(768->6), AdamW lr=1e-3 wd=3e-2, batch 10, 90 epoch, CrossEntropy.
Konvergensi dicatat per run (loss_final, loss_drop_last10).
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn

# ====== KONFIGURASI ======
FEAT_DIR = "/kaggle/working/features"
OUT_CSV = "/kaggle/working/results.csv"
SPLIT_CSV = "/kaggle/working/splits.csv"
BACKBONES = {                                  # nama file fitur -> (sumber, patch size)
    "imgnet16": ("ImageNet", 16), "imgnet32": ("ImageNet", 32),
    "hf16": ("HF", 16), "hf32": ("HF", 32),
}
TRAIN_N = [16, 14, 10, 8]                      # sampel per kelas untuk training
N_TEST = 4                                     # sampel per kelas, dikunci per blok
R = 10                                         # jumlah blok/replikasi
BASE_SEED = 2024
EPOCHS, BATCH, LR, WD, DROP = 90, 10, 1e-3, 3e-2, 0.1
DEVICE = "cpu"   # head sekecil ini lebih cepat di CPU daripada GPU (overhead launch kernel)
# =========================


def load(name):
    d = np.load(os.path.join(FEAT_DIR, f"{name}.npz"), allow_pickle=False)
    return d["X"], d["y"], d["sample"]


def make_block_split(y, sample, rng):
    """Acak urutan sampel di tiap kelas. Return {kelas: array id sampel terurut acak}."""
    order = {}
    for c in np.unique(y):
        ids = np.unique(sample[y == c])
        order[c] = rng.permutation(ids)
    return order


def select(order, k_train):
    """Ambil id sampel train (k pertama dari pool) dan test (N_TEST terakhir) per kelas."""
    tr, te = [], []
    for c, ids in order.items():
        pool = ids[: len(ids) - N_TEST]
        assert k_train <= len(pool), f"k_train={k_train} > pool={len(pool)}"
        tr += list(pool[:k_train])
        te += list(ids[len(ids) - N_TEST:])
    return np.array(tr), np.array(te)


def train_head(Xtr, ytr, Xte, yte, seed, Xpr=None, ypr=None):
    torch.manual_seed(seed)
    head = nn.Sequential(nn.Dropout(DROP), nn.Linear(768, 6)).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    g = torch.Generator().manual_seed(seed)
    n = len(Xtr)
    losses = []
    for _ in range(EPOCHS):
        head.train()
        perm = torch.randperm(n, generator=g)
        ep_loss = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = F.cross_entropy(head(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        losses.append(ep_loss / n)
    head.eval()
    with torch.no_grad():
        p_tr = head(Xtr).argmax(1).numpy()
        p_te = head(Xte).argmax(1).numpy()
        p_pr = head(Xpr).argmax(1).numpy() if Xpr is not None else None
    ytr_n, yte_n = ytr.numpy(), yte.numpy()
    out = {
        "train_acc": float((p_tr == ytr_n).mean()),
        "test_acc": float((p_te == yte_n).mean()),
        "test_macro_f1": float(f1_score(yte_n, p_te, average="macro")),
        "loss_final": losses[-1],
        "loss_drop_last10": (losses[-10] - losses[-1]) / losses[-10],  # ~0 = sudah konvergen
    }
    if p_pr is not None:  # set prediksi eksternal (60 citra, sampel terpisah dari 120 sampel)
        out["pred_acc"] = float((p_pr == ypr.numpy()).mean())
    return out


def main():
    feats = {b: load(b) for b in BACKBONES}
    preds = {}
    for b in BACKBONES:
        fp = os.path.join(FEAT_DIR, f"{b}_pred.npz")
        if os.path.exists(fp):
            d = np.load(fp, allow_pickle=False)
            preds[b] = (torch.from_numpy(d["X"]), torch.from_numpy(d["y"]).long())
    # semua backbone harus punya urutan citra & ID sampel yang sama
    ref = next(iter(feats.values()))
    for b, (_, y, s) in feats.items():
        assert np.array_equal(y, ref[1]) and np.array_equal(s, ref[2]), f"urutan {b} beda"
    y_all, s_all = ref[1], ref[2]

    done = set()
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        done = set(zip(prev.block, prev.source, prev.patch, prev.train_n))
        print(f"Lanjut dari {len(prev)} run yang sudah ada")
    split_rows = []

    for r in range(1, R + 1):
        rng = np.random.default_rng(BASE_SEED + r)
        order = make_block_split(y_all, s_all, rng)

        # catat split blok ini (untuk audit/laporan)
        _, te_ids = select(order, TRAIN_N[0])
        for c, ids in order.items():
            for rank, sid in enumerate(ids):
                split_rows.append({"block": r, "class": f"mutu{c+1}", "sample": sid,
                                   "rank": rank, "in_test": sid in set(te_ids)})

        # randomisasi urutan perlakuan di dalam blok
        treatments = [(b, k) for b in BACKBONES for k in TRAIN_N]  # 4 backbone x 4 level = 16
        treatments = [treatments[i] for i in rng.permutation(len(treatments))]

        for pos, (b, k) in enumerate(treatments, start=1):
            source, patch = BACKBONES[b]
            if (r, source, patch, k) in done:
                continue
            tr_ids, te_ids = select(order, k)
            X, _, _ = feats[b]
            m_tr, m_te = np.isin(s_all, tr_ids), np.isin(s_all, te_ids)
            assert not (m_tr & m_te).any(), "kebocoran: ada sampel di train dan test"
            seed = BASE_SEED * 1000 + r * 100 + pos
            Xpr, ypr = preds.get(b, (None, None))
            res = train_head(
                torch.from_numpy(X[m_tr]), torch.from_numpy(y_all[m_tr]).long(),
                torch.from_numpy(X[m_te]), torch.from_numpy(y_all[m_te]).long(), seed, Xpr, ypr,
            )
            row = {"block": r, "source": source, "patch": patch, "train_n": k, "run_order": pos, "seed": seed,
                   "n_train_img": int(m_tr.sum()), "n_test_img": int(m_te.sum()), **res}
            pd.DataFrame([row]).to_csv(OUT_CSV, mode="a", index=False, header=not os.path.exists(OUT_CSV))
            print(f"blok {r:2d} | {source:8s} patch {patch} | train_n {k:2d} | test_acc {res['test_acc']:.4f} "
                  f"| loss_drop_last10 {res['loss_drop_last10']:.3f}")

    pd.DataFrame(split_rows).to_csv(SPLIT_CSV, index=False)
    print("Selesai ->", OUT_CSV)


if __name__ == "__main__":
    main()
