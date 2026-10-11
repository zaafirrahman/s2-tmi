"""
sem_helpers.py - fungsi bantu untuk notebook SEM (TI5002 Minggu 6).

Berisi output yang TIDAK disediakan langsung oleh semopy:
Mardia, Cronbach's alpha, CR, AVE, R^2, efek langsung/tidak langsung/total,
tabel evaluasi fit, uji selisih chi-kuadrat, dan regresi OLS pada skor rata-rata.
Hanya bergantung pada numpy, pandas, scipy.
"""
import numpy as np
import pandas as pd
from scipy import stats


# ---------- Asumsi data ----------
def skew_kurt(X: pd.DataFrame) -> pd.DataFrame:
    """Skewness dan kurtosis (excess; normal = 0) tiap butir."""
    return pd.DataFrame({"skewness": X.apply(stats.skew), "kurtosis": X.apply(stats.kurtosis)})


def mardia_kurtosis(X: pd.DataFrame) -> dict:
    """Uji Mardia untuk kurtosis multivariat. H0: kurtosis = p(p+2) (normal multivariat)."""
    n, p = X.shape
    Z = X.values - X.values.mean(axis=0)
    Sinv = np.linalg.inv(np.cov(X.values, rowvar=False, bias=True))
    d2 = np.einsum("ij,jk,ik->i", Z, Sinv, Z)
    b2p = float((d2 ** 2).mean())
    z = (b2p - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
    return {"b2p": b2p, "expected": p * (p + 2), "z": float(z),
            "p_value": float(2 * (1 - stats.norm.cdf(abs(z))))}


# ---------- Membaca keluaran semopy.inspect() ----------
def _classify(row, ind_set, lat_set):
    l, r, op = row["lval"], row["rval"], row["op"]
    if l in ind_set and r in lat_set:
        return "loading", r, l                   # (jenis, konstruk/dari, indikator/ke)
    if l in lat_set and r in ind_set:
        return "loading", l, r
    if op == "~" and l in lat_set and r in lat_set:
        return "path", r, l                      # r -> l
    if op == "~~" and l == r:
        return "variance", l, l
    if op == "~~" and l in lat_set and r in lat_set:
        return "cov", l, r
    return "other", l, r


def annotate_estimates(est: pd.DataFrame, items, constructs) -> pd.DataFrame:
    """Tambah kolom kind/a/b dan ubah kolom numerik ('-' -> NaN). Tahan terhadap dua konvensi semopy."""
    est = est.copy()
    for col in ("Estimate", "Est. Std", "Std. Err", "z-value", "p-value"):
        if col in est.columns:
            est[col] = pd.to_numeric(est[col], errors="coerce")
    ind_set, lat_set = set(items), set(constructs)
    info = est.apply(lambda r: pd.Series(_classify(r, ind_set, lat_set), index=["kind", "a", "b"]), axis=1)
    return pd.concat([est, info], axis=1)


def offending_estimates(est: pd.DataFrame, load_cut=0.95, se_cut=1.0) -> dict:
    """Variansi error negatif, loading terstandar >= load_cut, standard error > se_cut."""
    return {
        "variansi_negatif": est[(est.kind == "variance") & (est.Estimate < 0)],
        "loading_terlalu_tinggi": est[(est.kind == "loading") & (est["Est. Std"].abs() >= load_cut)],
        "se_besar": est[est["Std. Err"] > se_cut],
    }


# ---------- Reliabilitas & validitas ----------
def cronbach_alpha(d: pd.DataFrame) -> float:
    k = d.shape[1]
    return float(k / (k - 1) * (1 - d.var(ddof=1).sum() / d.sum(axis=1).var(ddof=1)))


def reliability_table(est: pd.DataFrame, df: pd.DataFrame, constructs, n_items=3) -> pd.DataFrame:
    """Alpha, Composite Reliability, AVE, loading minimum per konstruk (loading terstandar)."""
    load = est[est.kind == "loading"]
    rows = []
    for c in constructs:
        lam = load.loc[load.a == c, "Est. Std"].values
        theta = 1 - lam ** 2
        rows.append({"konstruk": c,
                     "alpha": cronbach_alpha(df[[f"{c}{i}" for i in range(1, n_items + 1)]]),
                     "CR": lam.sum() ** 2 / (lam.sum() ** 2 + theta.sum()),
                     "AVE": float((lam ** 2).mean()),
                     "loading min": float(lam.min())})
    return pd.DataFrame(rows).set_index("konstruk")


# ---------- Model struktural ----------
def structural_table(est: pd.DataFrame) -> pd.DataFrame:
    sp = est[est.kind == "path"].copy()
    sp["Jalur"] = sp["a"] + " → " + sp["b"]
    sp["Signifikan (p<0,05)?"] = np.where(sp["p-value"] < 0.05, "ya", "tidak")
    out = sp.set_index("Jalur")[["Estimate", "Est. Std", "Std. Err", "z-value", "p-value", "Signifikan (p<0,05)?"]]
    return out.rename(columns={"Estimate": "Koef. tak terstandar", "Est. Std": "Koef. terstandar"})


def r_squared(est: pd.DataFrame, endo) -> dict:
    """R^2 = 1 - variansi error struktural terstandar."""
    v = est[est.kind == "variance"]
    return {e: float(1 - v.loc[v.a == e, "Est. Std"].iloc[0]) for e in endo}


def std_paths(est: pd.DataFrame) -> dict:
    """{(dari, ke): koefisien terstandar}"""
    return {(r.a, r.b): r["Est. Std"] for _, r in est[est.kind == "path"].iterrows()}


def effects(std: dict, exo, endo) -> pd.DataFrame:
    """Efek langsung, tidak langsung, total dari eksogen ke endogen: total = (I-B)^-1 Gamma."""
    B = np.zeros((len(endo), len(endo)))
    G = np.zeros((len(endo), len(exo)))
    for (src, dst), v in std.items():
        if src in exo:
            G[endo.index(dst), exo.index(src)] = v
        else:
            B[endo.index(dst), endo.index(src)] = v
    total = np.linalg.inv(np.eye(len(endo)) - B) @ G
    cols = pd.MultiIndex.from_product([["Langsung", "Tidak langsung", "Total"], exo])
    return pd.DataFrame(np.hstack([G, total - G, total]), index=endo, columns=cols)


def ols_standardized(df: pd.DataFrame, constructs, struct_paths, n_items=3) -> dict:
    """Regresi OLS terstandar pada skor rata-rata butir (pembanding SEM; terkena attenuasi)."""
    comp = pd.DataFrame({c: df[[f"{c}{i}" for i in range(1, n_items + 1)]].mean(axis=1) for c in constructs})
    z = (comp - comp.mean()) / comp.std()
    out = {}
    for dst in sorted({d for _, d in struct_paths}, key=lambda d: [p[1] for p in struct_paths].index(d)):
        preds = [s for s, d in struct_paths if d == dst]
        b = np.linalg.lstsq(z[preds].values, z[dst].values, rcond=None)[0]
        out.update({(s, dst): float(v) for s, v in zip(preds, b)})
    return out


# ---------- Fit ----------
_RULES = {"chi2 p-value": (lambda v: v > 0.05, "p > 0,05"),
          "CFI": (lambda v: v >= 0.90, "≥ 0,90 (baik ≥ 0,95)"),
          "TLI": (lambda v: v >= 0.90, "≥ 0,90 (baik ≥ 0,95)"),
          "GFI": (lambda v: v >= 0.90, "≥ 0,90"),
          "NFI": (lambda v: v >= 0.90, "≥ 0,90"),
          "RMSEA": (lambda v: v <= 0.08, "≤ 0,08 (baik ≤ 0,06)")}


def fit_table(fit: pd.Series) -> pd.DataFrame:
    rows = {}
    for k, (rule, crit) in _RULES.items():
        if k in fit.index:
            rows[k] = {"Nilai": fit[k], "Kriteria": crit, "Lolos?": "ya" if rule(fit[k]) else "tidak"}
    return pd.DataFrame(rows).T


def chi_square_difference(small: pd.Series, big: pd.Series) -> dict:
    """Model bersarang: 'small' = lebih sedikit parameter (df lebih besar)."""
    d_chi = float(small["chi2"] - big["chi2"])
    d_df = int(small["DoF"] - big["DoF"])
    return {"delta_chi2": d_chi, "delta_df": d_df, "p_value": float(1 - stats.chi2.cdf(d_chi, d_df))}
