"""
fungsi_cluster.py
=================
Fungsi-fungsi BANTU (generik) untuk analisis segmentasi dengan cluster (hierarkis, K-means, profiling).

Prinsip modul ini: hanya berisi alat yang tidak bergantung pada hasil analisis.
Hal-hal yang baru diketahui SETELAH data diproses (ambang anomali, variabel terpilih,
jumlah cluster, nama & warna segmen, bobot daya tarik, dst.) TIDAK ada di sini; semuanya
dideklarasikan dan dijelaskan langsung di notebook notebook analisis.

Isi modul (berurutan mengikuti alur notebook):
    1. Loader data
    2. Data quality & cleaning   (uji identitas, penandaan anomali berbasis aturan)
    3. Seleksi variabel          (skewness, transformasi + standarisasi, korelasi, VIF, grafik log)
    4. Cluster hierarkis         (linkage, dendrogram, agglomeration schedule, evaluasi k, silhouette)
    5. K-means & perbandingan    (algoritma Lloyd manual, centroid, pencocokan label, stabilitas)
    6. Profiling                 (rata-rata, z-score, ANOVA, Kruskal-Wallis, Chi-square, Cramer's V)
    7. Daya tarik segmen         (skor min-max, sensitivitas bobot)
    8. Helper visualisasi
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =============================================================================
# 1. LOADER
# =============================================================================
def load_dataset(path: str | Path, source: str | None = None) -> pd.DataFrame:
    """Baca file Excel data set. Baris pertama file berisi judul, header ada di baris ke-2."""
    df = pd.read_excel(path, header=1)
    if "customer_id" not in df.columns:  # jaga-jaga kalau header di baris pertama
        df = pd.read_excel(path, header=0)
    if source:
        df["source"] = source
    return df


# =============================================================================
# 2. DATA QUALITY & CLEANING
# =============================================================================
def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan cepat: tipe, missing, jumlah unik, min, max tiap kolom."""
    rows = []
    for c in df.columns:
        s = df[c]
        is_num = pd.api.types.is_numeric_dtype(s)
        rows.append({
            "kolom": c, "tipe": str(s.dtype), "missing": int(s.isna().sum()),
            "n_unik": int(s.nunique()),
            "min": s.min() if is_num else "-", "max": s.max() if is_num else "-",
        })
    return pd.DataFrame(rows).set_index("kolom")


def check_identities(df: pd.DataFrame, identities: dict) -> pd.DataFrame:
    """
    Uji identitas antar kolom. `identities` dibuat di notebook:
        {nama: (fungsi_rasio(df) -> Series, toleransi, jenis)}
    jenis:
        'sama_dengan_1' : lolos jika |rasio - 1| <= toleransi
        'konstan'       : lolos jika rasio berada dalam +-toleransi dari MEDIAN rasio (rasio tetap?)
        'maks_1'        : lolos jika rasio <= 1
    Kolom 'pct_dalam_toleransi' = persentase baris yang lolos.
    """
    rows = []
    for nama, (fn, tol, jenis) in identities.items():
        r = fn(df)
        if jenis == "sama_dengan_1":
            ok, tol_txt = ((r - 1).abs() <= tol).mean() * 100, f"+-{tol:.0%}"
        elif jenis == "konstan":
            ok, tol_txt = ((r / r.median() - 1).abs() <= tol).mean() * 100, f"+-{tol:.0%} dari median"
        elif jenis == "maks_1":
            ok, tol_txt = (r <= 1 + 1e-6).mean() * 100, "<= 1"
        else:
            raise ValueError(f"jenis tidak dikenal: {jenis}")
        rows.append({"identitas": nama, "median_rasio": r.median(), "rasio_min": r.min(),
                     "rasio_maks": r.max(), "toleransi": tol_txt, "pct_dalam_toleransi": ok})
    return pd.DataFrame(rows).set_index("identitas")


def flag_by_rules(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Tandai baris yang melanggar aturan. `rules` dibuat di notebook:
        {nama_aturan: (fungsi_mask(df) -> Series[bool], penjelasan)}
    Return DataFrame boolean: satu kolom per aturan + kolom 'any' (baris mana pun yang kena).
    """
    flags = pd.DataFrame({k: f(df) for k, (f, _) in rules.items()}, index=df.index)
    flags["any"] = flags.any(axis=1)
    return flags


def split_flagged(df: pd.DataFrame, flags: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kembalikan (data_bersih, data_dibuang). Data dibuang dilengkapi kolom 'alasan' (aturan yang dilanggar)."""
    removed = df[flags["any"]].copy()
    if removed.empty:
        removed["alasan"] = pd.Series(dtype=str)
    else:
        removed["alasan"] = flags.loc[flags["any"]].drop(columns="any").apply(
            lambda r: ", ".join(r.index[r]), axis=1)
    return df[~flags["any"]].copy().reset_index(drop=True), removed


# =============================================================================
# 3. SELEKSI VARIABEL
# =============================================================================
def skew_table(df: pd.DataFrame, cols: list[str], thr: float = 1.0) -> pd.DataFrame:
    """Skewness tiap variabel dan rekomendasi transformasi log bila skew > thr."""
    sk = df[cols].skew().rename("skewness")
    out = sk.to_frame()
    out["perlu_log"] = out["skewness"] > thr
    return out.sort_values("skewness", ascending=False)


def prepare_features(df: pd.DataFrame, cols: list[str], log_cols: list[str] | None = None,
                     scaler: StandardScaler | None = None):
    """
    Transformasi log1p (untuk variabel miring) lalu standarisasi z-score.
    Return: (Z sebagai DataFrame, scaler). Jika scaler diberikan, dipakai apa adanya (transform saja).
    """
    log_cols = log_cols or []
    X = df[cols].astype(float).copy()
    for c in log_cols:
        if c in X.columns:
            X[c] = np.log1p(X[c])
    if scaler is None:
        scaler = StandardScaler().fit(X)
    Z = pd.DataFrame(scaler.transform(X), columns=cols, index=df.index)
    return Z, scaler


def compute_vif(Z: pd.DataFrame) -> pd.Series:
    """Variance Inflation Factor tiap kolom (VIF > 10 = multikolinearitas serius, > 5 = waspada)."""
    Xc = np.c_[np.ones(len(Z)), Z.values]
    vals = [variance_inflation_factor(Xc, i + 1) for i in range(Z.shape[1])]
    return pd.Series(vals, index=Z.columns, name="VIF").sort_values(ascending=False)


def high_corr_pairs(corr: pd.DataFrame, thr: float = 0.8) -> pd.DataFrame:
    """Daftar pasangan variabel dengan |korelasi| >= thr."""
    rows = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= thr:
                rows.append({"variabel_1": cols[i], "variabel_2": cols[j], "korelasi": r})
    return (pd.DataFrame(rows).sort_values("korelasi", key=abs, ascending=False)
            .reset_index(drop=True))


# =============================================================================
# 4. CLUSTER HIERARKIS
# =============================================================================
def compare_linkages(Z: pd.DataFrame, k: int = 4,
                     methods=("single", "complete", "average", "ward")) -> pd.DataFrame:
    """
    Bandingkan metode aglomerasi (jarak Euclidean):
    cophenetic correlation (seberapa setia dendrogram pada jarak asli), silhouette, dan sebaran ukuran cluster.
    """
    d = pdist(Z.values, metric="euclidean")
    rows = []
    for m in methods:
        L = linkage(Z.values, method=m, metric="euclidean")
        coph, _ = cophenet(L, d)
        lab = fcluster(L, k, "maxclust")
        sizes = np.sort(np.bincount(lab)[1:])[::-1]
        rows.append({
            "metode": m, "cophenetic_corr": coph,
            f"silhouette_k{k}": silhouette_score(Z.values, lab),
            "ukuran_cluster": list(sizes), "cluster_terkecil": int(sizes.min()),
        })
    return pd.DataFrame(rows).set_index("metode")


def agglomeration_table(L: np.ndarray, max_k: int = 10) -> pd.DataFrame:
    """
    Agglomeration schedule versi ringkas.
    Untuk solusi dengan k cluster: 'biaya_gabung_berikutnya' = jarak (Ward) saat k cluster digabung
    menjadi k-1. Lonjakan besar artinya menggabungkan lagi memaksa dua kelompok yang sebenarnya beda.
    """
    N = L.shape[0] + 1
    rows = []
    for k in range(max_k, 1, -1):
        cost_next = L[N - k, 2]          # biaya menggabung k -> k-1
        cost_prev = L[N - k - 1, 2]      # biaya menggabung (k+1) -> k
        rows.append({"k": k, "biaya_gabung_berikutnya": cost_next,
                     "lonjakan_vs_sebelumnya": cost_next - cost_prev,
                     "kenaikan_pct": (cost_next / cost_prev - 1) * 100})
    return pd.DataFrame(rows).set_index("k")


def wss(Z: np.ndarray, labels: np.ndarray) -> float:
    """Within-cluster sum of squares."""
    total = 0.0
    for c in np.unique(labels):
        pts = Z[labels == c]
        total += ((pts - pts.mean(axis=0)) ** 2).sum()
    return total


def evaluate_k(Z: pd.DataFrame, ks=range(2, 9), method: str = "hier", L: np.ndarray | None = None,
               random_state: int = 42) -> pd.DataFrame:
    """
    Metrik pemilihan jumlah cluster: silhouette (tinggi = bagus), Calinski-Harabasz (tinggi),
    Davies-Bouldin (rendah), WSS (untuk elbow), serta ukuran cluster terkecil.
    method='hier' memakai linkage L (Ward); method='kmeans' menjalankan KMeans.
    """
    rows = []
    for k in ks:
        if method == "hier":
            lab = fcluster(L, k, "maxclust") - 1
        else:
            lab = KMeans(k, n_init=30, random_state=random_state).fit_predict(Z.values)
        rows.append({
            "k": k, "silhouette": silhouette_score(Z.values, lab),
            "calinski_harabasz": calinski_harabasz_score(Z.values, lab),
            "davies_bouldin": davies_bouldin_score(Z.values, lab),
            "wss": wss(Z.values, lab),
            "cluster_terkecil_pct": np.bincount(lab).min() / len(lab) * 100,
        })
    return pd.DataFrame(rows).set_index("k")


def plot_dendrogram(L: np.ndarray, k: int | None = None, ax=None, p: int = 40, ylabel: str = "Jarak"):
    """Dendrogram terpotong (p daun terakhir) dengan garis potong untuk k cluster."""
    ax = ax or plt.gca()
    if k is None:
        dendrogram(L, truncate_mode="lastp", p=p, no_labels=True, color_threshold=0,
                   above_threshold_color="#1f4e79", ax=ax)
    else:
        N = L.shape[0] + 1
        cut = (L[N - k, 2] + L[N - k - 1, 2]) / 2
        dendrogram(L, truncate_mode="lastp", p=p, color_threshold=cut, no_labels=True,
                   above_threshold_color="grey", ax=ax)
        ax.axhline(cut, color="red", ls="--", lw=1.2, label=f"garis potong (k = {k})")
        ax.legend(loc="upper right")
    ax.set_ylabel(ylabel)
    return ax


def plot_silhouette(Z: pd.DataFrame, labels, ax=None, names: dict | None = None, colors: dict | None = None):
    """Silhouette per anggota, dikelompokkan per cluster (batang melebar ke kanan = cocok di clusternya)."""
    labels = np.asarray(labels)
    sil = silhouette_samples(Z.values, labels)
    ax = ax or plt.gca()
    y = 0
    for c in np.unique(labels):
        v = np.sort(sil[labels == c])
        nm = (names or {}).get(c, f"Cluster {c}")
        ax.fill_betweenx(np.arange(y, y + len(v)), 0, v, color=(colors or {}).get(nm, None), alpha=0.8)
        ax.text(-0.02, y + len(v) / 2, f"{nm}\n(rata-rata {v.mean():.2f})", ha="right", va="center", fontsize=8)
        y += len(v) + 15
    ax.axvline(sil.mean(), color="red", ls="--", lw=1.2, label=f"rata-rata total = {sil.mean():.3f}")
    ax.set_yticks([]); ax.set_xlabel("Silhouette"); ax.legend(loc="lower right")
    return ax


# =============================================================================
# 5. K-MEANS, PERBANDINGAN & STABILITAS
# =============================================================================
def cluster_centroids(Z: pd.DataFrame, labels) -> pd.DataFrame:
    """Rata-rata z-score per cluster."""
    return Z.groupby(np.asarray(labels)).mean()


def match_clusters(ref_centroids: pd.DataFrame, new_centroids: pd.DataFrame) -> dict:
    """
    Samakan label cluster solusi baru dengan solusi acuan (Hungarian algorithm pada jarak centroid).
    Return dict {label_baru: label_acuan}.
    """
    cost = np.linalg.norm(new_centroids.values[:, None, :] - ref_centroids.values[None, :, :], axis=2)
    r, c = linear_sum_assignment(cost)
    return {new_centroids.index[i]: ref_centroids.index[j] for i, j in zip(r, c)}


def assign_to_centroids(Z: pd.DataFrame, centroids: pd.DataFrame) -> np.ndarray:
    """Tempatkan tiap baris ke centroid terdekat (validasi silang antar-sampel)."""
    d = np.linalg.norm(Z.values[:, None, :] - centroids.values[None, :, :], axis=2)
    return centroids.index.values[d.argmin(axis=1)]


def size_table(labels, names: dict | None = None) -> pd.DataFrame:
    """Jumlah dan persentase anggota tiap cluster."""
    s = pd.Series(labels).value_counts().sort_index()
    out = pd.DataFrame({"n": s, "persen": s / s.sum() * 100})
    if names:
        out.index = [names.get(i, i) for i in out.index]
    return out


def bootstrap_stability(Z: pd.DataFrame, k: int, n_boot: int = 100, frac: float = 0.8,
                        random_state: int = 42) -> np.ndarray:
    """
    Stabilitas cluster lewat resampling: cluster ulang 80% data secara acak (K-means),
    bandingkan dengan solusi utama pada sampel yang sama (Adjusted Rand Index). ARI ~1 = sangat stabil.
    """
    rng = np.random.default_rng(random_state)
    base = KMeans(k, n_init=30, random_state=random_state).fit_predict(Z.values)
    aris = []
    for b in range(n_boot):
        idx = rng.choice(len(Z), int(frac * len(Z)), replace=False)
        lab = KMeans(k, n_init=10, random_state=b).fit_predict(Z.values[idx])
        aris.append(adjusted_rand_score(base[idx], lab))
    return np.array(aris)


def run_spec(df1: pd.DataFrame, df2: pd.DataFrame, cols: list[str], log_cols: list[str],
             k: int = 4, random_state: int = 42) -> dict:
    """
    Jalankan satu spesifikasi lengkap: Ward pada A1 dan K-means pada A2, lalu ukur konsistensinya.
    Dipakai untuk membandingkan spesifikasi awal (naif) vs spesifikasi final (respesifikasi).
    """
    Z1, sc1 = prepare_features(df1, cols, log_cols)
    Z2, _ = prepare_features(df2, cols, log_cols)
    L = linkage(Z1.values, "ward")
    lab1 = fcluster(L, k, "maxclust") - 1
    lab2 = KMeans(k, n_init=30, random_state=random_state).fit_predict(Z2.values)

    # A2 ditempatkan ke centroid A1 (skala A1) dan dibandingkan dengan K-means A2
    Z2_on_1, _ = prepare_features(df2, cols, log_cols, scaler=sc1)
    pred = assign_to_centroids(Z2_on_1, cluster_centroids(Z1, lab1))
    return {
        "ukuran A1 (Ward)": list(np.sort(np.bincount(lab1))[::-1]),
        "ukuran A2 (K-means)": list(np.sort(np.bincount(lab2))[::-1]),
        "silhouette A1": silhouette_score(Z1.values, lab1),
        "silhouette A2": silhouette_score(Z2.values, lab2),
        "ARI (A2 K-means vs centroid A1)": adjusted_rand_score(lab2, pred),
        "cluster terkecil (A1)": int(np.bincount(lab1).min()),
    }


# =============================================================================
# 6. PROFILING
# =============================================================================
def cluster_means(df: pd.DataFrame, labels, cols: list[str], agg: str = "mean") -> pd.DataFrame:
    """Rata-rata (atau median) variabel per segmen."""
    return df.groupby(np.asarray(labels))[cols].agg(agg)


def zscore_profile(df: pd.DataFrame, labels, cols: list[str]) -> pd.DataFrame:
    """Rata-rata z-score (relatif terhadap seluruh member) per segmen -> untuk heatmap profil."""
    Z = (df[cols] - df[cols].mean()) / df[cols].std()
    return Z.groupby(np.asarray(labels)).mean()


def anova_table(df: pd.DataFrame, labels, cols: list[str], alpha: float = 0.05) -> pd.DataFrame:
    """
    Uji perbedaan antar segmen untuk variabel numerik.
    - ANOVA satu arah (F, p) + effect size eta-squared
    - Kruskal-Wallis (H, p) sebagai uji robust untuk data miring
    - p dikoreksi Bonferroni karena banyak variabel diuji sekaligus
    Aturan baca eta^2: >=0.14 besar, 0.06-0.14 sedang, <0.06 kecil.
    """
    labels = np.asarray(labels)
    groups_idx = [np.where(labels == g)[0] for g in np.unique(labels)]
    rows = []
    for c in cols:
        groups = [df[c].values[i] for i in groups_idx]
        F, p = stats.f_oneway(*groups)
        H, p_kw = stats.kruskal(*groups)
        grand = df[c].mean()
        ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_t = ((df[c] - grand) ** 2).sum()
        rows.append({"variabel": c, "SS_antar": ss_b, "SS_dalam": ss_t - ss_b,
                     "df_antar": len(groups) - 1, "df_dalam": len(df) - len(groups),
                     "F": F, "p_anova": p, "eta_sq": ss_b / ss_t,
                     "H_kruskal": H, "p_kruskal": p_kw})
    out = pd.DataFrame(rows).set_index("variabel")
    m = len(cols)
    out["p_bonferroni"] = (out["p_kruskal"] * m).clip(upper=1)
    out["p_bonf_anova"] = (out["p_anova"] * m).clip(upper=1)
    out["membedakan"] = out["p_bonferroni"] < alpha
    out["kekuatan"] = pd.cut(out["eta_sq"], [-1, 0.06, 0.14, 2], labels=["kecil", "sedang", "besar"])
    return out.sort_values("eta_sq", ascending=False)


def chi2_table(df: pd.DataFrame, labels, cols: list[str], alpha: float = 0.05) -> pd.DataFrame:
    """Uji Chi-square independensi (segmen x variabel kategorikal) + Cramer's V sebagai effect size."""
    rows = []
    for c in cols:
        ct = pd.crosstab(df[c], np.asarray(labels))
        chi2, p, dof, exp = stats.chi2_contingency(ct)
        n = ct.values.sum()
        v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
        rows.append({"variabel": c, "chi2": chi2, "dof": dof, "p_value": p, "cramers_v": v,
                     "sel_expected_<5_pct": (exp < 5).mean() * 100})
    out = pd.DataFrame(rows).set_index("variabel")
    out["p_bonferroni"] = (out["p_value"] * len(cols)).clip(upper=1)
    out["membedakan"] = out["p_bonferroni"] < alpha
    out["kekuatan"] = pd.cut(out["cramers_v"], [-1, 0.1, 0.3, 2], labels=["lemah", "sedang", "kuat"])
    return out.sort_values("cramers_v", ascending=False)


def crosstab_pct(df: pd.DataFrame, labels, var: str, order: list | None = None) -> pd.DataFrame:
    """Komposisi (%) kategori var di dalam tiap segmen (kolom = 100%). `order` = urutan baris kategori (opsional)."""
    ct = pd.crosstab(df[var], np.asarray(labels), normalize="columns") * 100
    if order is not None:
        ct = ct.reindex(order)
    return ct


# =============================================================================
# 7. DAYA TARIK SEGMEN
# =============================================================================
def _minmax_1_5(s: pd.Series) -> pd.Series:
    return 1 + 4 * (s - s.min()) / (s.max() - s.min())


def attractiveness_table(df: pd.DataFrame, seg_col: str, weights: dict, value_col: str,
                         access_cols: list[str], extra_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Skor daya tarik segmen (skala 1-5 per kriteria, min-max antar segmen):
      - ukuran : jumlah anggota segmen
      - nilai  : rata-rata `value_col` per anggota (mis. total belanja)
      - akses  : rata-rata dari tiap `access_cols` yang dinormalkan dengan nilai maksimumnya
                 (mis. persentase pemesanan langsung dan sesi aplikasi)
    `weights` = {"ukuran": w1, "nilai": w2, "akses": w3} (jumlah = 1); semua ditentukan di notebook.
    `extra_cols` = kolom tambahan yang hanya dilaporkan rata-ratanya (tidak ikut skor).
    """
    g = df.groupby(seg_col)
    raw = pd.DataFrame({
        "n_member": g.size(),
        "pct_member": g.size() / len(df) * 100,
        f"avg_{value_col}": g[value_col].mean(),
        "revenue_share_pct": g[value_col].sum() / df[value_col].sum() * 100,
    })
    for c in list(access_cols) + list(extra_cols or []):
        raw[c] = g[c].mean()
    akses_raw = sum(raw[c] / raw[c].max() for c in access_cols) / len(access_cols)
    score = pd.DataFrame({
        "skor_ukuran": _minmax_1_5(raw["n_member"]),
        "skor_nilai": _minmax_1_5(raw[f"avg_{value_col}"]),
        "skor_akses": _minmax_1_5(akses_raw),
    })
    score["skor_total"] = (weights["ukuran"] * score["skor_ukuran"] + weights["nilai"] * score["skor_nilai"]
                           + weights["akses"] * score["skor_akses"])
    out = raw.join(score)
    out["peringkat"] = out["skor_total"].rank(ascending=False).astype(int)
    return out.sort_values("skor_total", ascending=False)


def weight_sensitivity(df: pd.DataFrame, seg_col: str, schemes: dict, **kwargs) -> pd.DataFrame:
    """Peringkat segmen di bawah beberapa skema bobot (`schemes` = {nama_skema: weights}) -> apakah prioritas berubah?"""
    return pd.DataFrame({nm: attractiveness_table(df, seg_col, w, **kwargs)["peringkat"] for nm, w in schemes.items()})


# =============================================================================
# 8. VISUALISASI
# =============================================================================
def set_style():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False})


def save_fig(fig, name: str, folder: str | Path = "figures"):
    """Simpan grafik sebagai PNG resolusi tinggi (siap ditempel ke slide)."""
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=200, bbox_inches="tight")


def plot_corr_heatmap(corr: pd.DataFrame, ax=None, annot: bool = True):
    ax = ax or plt.gca()
    sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=annot, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.4, cbar_kws={"shrink": 0.7}, ax=ax)
    return ax


def plot_profile_heatmap(zprof: pd.DataFrame, ax=None, title: str = ""):
    """Heatmap profil segmen (z-score rata-rata; merah = di atas rata-rata, biru = di bawah)."""
    ax = ax or plt.gca()
    sns.heatmap(zprof, cmap="RdBu_r", center=0, vmin=-2, vmax=2, annot=True, fmt=".2f",
                linewidths=0.5, cbar_kws={"shrink": 0.7, "label": "z-score"}, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("")
    return ax


def plot_attractiveness_bubble(att: pd.DataFrame, colors: dict | None = None, ax=None):
    """Bubble chart: sumbu X = kemudahan dijangkau, Y = nilai bagi perusahaan, ukuran gelembung = jumlah member."""
    ax = ax or plt.gca()
    for seg, r in att.iterrows():
        ax.scatter(r["skor_akses"], r["skor_nilai"], s=r["n_member"] * 6, alpha=0.75,
                   color=(colors or {}).get(seg, "grey"), edgecolor="white", lw=1.5)
        ax.annotate(f"{seg}\n({r['pct_member']:.0f}% member)", (r["skor_akses"], r["skor_nilai"]),
                    ha="center", va="center", fontsize=8, color="black",
                    xytext=(0, -np.sqrt(r["n_member"] * 6) / 2 - 14), textcoords="offset points")
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.axhline(3, color="grey", ls=":", lw=1)
    ax.axvline(3, color="grey", ls=":", lw=1)
    ax.set_xlabel("Kemudahan dijangkau (skor 1-5)")
    ax.set_ylabel("Nilai bagi perusahaan (skor 1-5)")
    return ax


# =============================================================================
# 9. HELPER TAMBAHAN (grafik transformasi log & demonstrasi K-means)
# =============================================================================
def plot_before_after_log(df: pd.DataFrame, cols: list[str], pairs_per_row: int = 3, bins: int = 40):
    """Histogram sebelum (abu-abu) dan sesudah log1p (biru) untuk setiap variabel, lengkap dengan skewness."""
    rows = int(np.ceil(len(cols) / pairs_per_row))
    fig, axes = plt.subplots(rows, pairs_per_row * 2, figsize=(5.3 * pairs_per_row, 2.7 * rows))
    axes = np.atleast_2d(axes).reshape(rows, -1)
    for ax in axes.ravel():
        ax.set_visible(False)
    for i, c in enumerate(cols):
        r, j = divmod(i, pairs_per_row)
        a, b = axes[r, 2 * j], axes[r, 2 * j + 1]
        a.set_visible(True); b.set_visible(True)
        x, lx = df[c], np.log1p(df[c])
        sns.histplot(x, bins=bins, ax=a, color="#9aa5b1"); a.set_title(f"{c}\nskew = {x.skew():.2f}", fontsize=9)
        sns.histplot(lx, bins=bins, ax=b, color="#1f4e79"); b.set_title(f"log1p({c})\nskew = {lx.skew():.2f}", fontsize=9)
        for ax in (a, b):
            ax.set_xlabel(""); ax.set_ylabel(""); ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def lloyd_kmeans(X: np.ndarray, init_centroids: np.ndarray, max_iter: int = 100) -> list[dict]:
    """
    Algoritma K-means (Lloyd) ditulis eksplisit supaya tiap iterasi bisa dilihat.
    Tiap iterasi: (1) tiap titik ke centroid terdekat, (2) hitung WSS, (3) centroid = rata-rata anggotanya.
    Berhenti saat tidak ada titik yang pindah cluster. Return list dict per iterasi:
    iterasi, pindah (jumlah titik yang berganti cluster; iterasi 1 = semua titik), wss,
    centroid (yang dipakai untuk penugasan iterasi itu), label.
    """
    C = np.array(init_centroids, dtype=float).copy()
    lab, hist = None, []
    for it in range(1, max_iter + 1):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        new = d2.argmin(axis=1)
        moved = len(X) if lab is None else int((new != lab).sum())
        hist.append({"iterasi": it, "pindah": moved, "wss": float(d2[np.arange(len(X)), new].sum()),
                     "centroid": C.copy(), "label": new.copy()})
        if moved == 0:
            break
        lab = new
        C = np.array([X[lab == k].mean(axis=0) if (lab == k).any() else C[k] for k in range(len(C))])
    return hist


def plot_kmeans_snapshots(X: np.ndarray, history: list[dict], steps: list[int]):
    """
    Tampilkan iterasi tertentu pada proyeksi 2D (PCA). Titik berwarna = cluster saat itu,
    tanda X hitam = centroid yang dipakai pada iterasi itu. Return (fig, explained_variance_ratio).
    """
    pca = PCA(2).fit(X)
    P = pca.transform(X)
    fig, axes = plt.subplots(1, len(steps), figsize=(4.4 * len(steps), 4.2), sharex=True, sharey=True)
    for ax, i in zip(np.atleast_1d(axes), steps):
        h = history[i]
        C2 = pca.transform(h["centroid"])
        ax.scatter(P[:, 0], P[:, 1], c=h["label"], cmap="tab10", vmin=0, vmax=9, s=7, alpha=0.55)
        ax.scatter(C2[:, 0], C2[:, 1], marker="X", s=170, c="black", edgecolor="white", lw=1.2)
        ax.set_title(f"Iterasi {h['iterasi']}\n{h['pindah']} titik pindah | WSS {h['wss']:,.0f}", fontsize=9)
    fig.supxlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} variasi)", fontsize=9)
    fig.supylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})", fontsize=9)
    fig.tight_layout()
    return fig, pca.explained_variance_ratio_
