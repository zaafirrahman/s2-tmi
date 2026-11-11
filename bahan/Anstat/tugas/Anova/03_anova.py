"""
03_anova.py
-----------
Analisis RAK faktorial 2 (sumber pre-training) x 2 (patch) x 4 (jumlah data latih), blok = replikasi.
  model : y ~ C(block) + C(source) * C(patch) * C(train_n)   (kolom source dipakai jika ada)
  uji   : ANOVA, partial eta-squared, Shapiro-Wilk, Levene (Brown-Forsythe), Durbin-Watson,
          plot residual, Tukey HSD
Jalankan lokal atau di Kaggle:  RESULTS_CSV=results.csv python 03_anova.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.stattools import durbin_watson

CSV = os.environ.get("RESULTS_CSV", "results.csv")
OUT = os.environ.get("ANOVA_OUT", "anova_out")
RESP = "test_acc"       # atau "test_macro_f1"
ASIN_SQRT = False       # True jika asumsi gagal (respon proporsi) -> transformasi arcsin(sqrt(y))
ALPHA = 0.05

os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(CSV)
df["y"] = np.arcsin(np.sqrt(df[RESP])) if ASIN_SQRT else df[RESP]
df["block"] = df["block"].astype(str)
df["patch"] = df["patch"].astype(str)
df["train_n"] = df["train_n"].astype(str)
HAS_SRC = "source" in df.columns and df["source"].nunique() > 1
FACTORS = (["source"] if HAS_SRC else []) + ["patch", "train_n"]
df["cell"] = df[FACTORS].agg("|".join, axis=1)

log = []
def say(s=""):
    print(s)
    log.append(str(s))

say(f"Respon: {RESP}{' (arcsin-sqrt)' if ASIN_SQRT else ''} | n = {len(df)} run | blok = {df.block.nunique()}")
say(f"Run per sel (harus seimbang): {sorted(df.groupby('cell').size().unique())}\n")

# ---- deskriptif ----
desc = df.groupby(FACTORS)[RESP].agg(["count", "mean", "std"]).round(4)
say("Rata-rata per sel:\n" + desc.to_string() + "\n")

# ---- ANOVA ----
formula = "y ~ C(block) + " + " * ".join(f"C({f})" for f in FACTORS)
say(f"Model: {formula}")
model = smf.ols(formula, data=df).fit()
tab = anova_lm(model, typ=2)
tab["partial_eta_sq"] = tab["sum_sq"] / (tab["sum_sq"] + tab.loc["Residual", "sum_sq"])
tab.loc["Residual", "partial_eta_sq"] = np.nan
say("ANOVA (Type II; desain seimbang sehingga sama dengan Type I/III):\n" + tab.round(5).to_string() + "\n")

# ---- uji asumsi ----
resid = model.resid
sw = stats.shapiro(resid)
lv = stats.levene(*[g["y"].values for _, g in df.groupby("cell")], center="median")
df_sorted = df.assign(resid=resid).sort_values(["block", "run_order"])
dw = durbin_watson(df_sorted["resid"].values)
say("Uji asumsi:")
say(f"  Normalitas residual (Shapiro-Wilk): W={sw.statistic:.4f}, p={sw.pvalue:.4f} -> {'OK' if sw.pvalue > ALPHA else 'GAGAL'}")
say(f"  Homogenitas varians (Levene median): F={lv.statistic:.4f}, p={lv.pvalue:.4f} -> {'OK' if lv.pvalue > ALPHA else 'GAGAL'}")
say(f"  Independensi (Durbin-Watson): DW={dw:.3f} (sekitar 2 = tidak ada autokorelasi)\n")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(model.fittedvalues, resid, s=12); ax[0].axhline(0, c="k", lw=.8)
ax[0].set(xlabel="Fitted", ylabel="Residual", title="Residual vs fitted")
sm.qqplot(resid, line="s", ax=ax[1]); ax[1].set_title("Q-Q plot residual")
ax[2].plot(df_sorted["resid"].values, marker="o", ms=3, lw=.6); ax[2].axhline(0, c="k", lw=.8)
ax[2].set(xlabel="Urutan run", ylabel="Residual", title="Residual vs urutan run")
plt.tight_layout(); plt.savefig(f"{OUT}/residual_plots.png", dpi=150); plt.close()

# ---- interaction plot ----
fig, ax = plt.subplots(figsize=(6, 4))
line_keys = ["source", "patch"] if HAS_SRC else ["patch"]
for key, g in df.groupby(line_keys):
    key = key if isinstance(key, tuple) else (key,)
    m = g.groupby(g["train_n"].astype(int))[RESP].agg(["mean", "std"]).sort_index()
    ax.errorbar(m.index, m["mean"], yerr=m["std"], marker="o", capsize=3,
                label=" patch ".join(map(str, key)) if HAS_SRC else f"patch {key[0]}")
ax.set(xlabel="Sampel latih per kelas", ylabel=RESP, title="Rata-rata ± SD per perlakuan")
ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/interaction_plot.png", dpi=150); plt.close()

# ---- post hoc ----
# Catatan: Tukey di bawah mengabaikan efek blok, jadi sedikit konservatif.
inter_terms = [t for t in tab.index if ":" in t]
sig_inter = [t for t in inter_terms if tab.loc[t, "PR(>F)"] < ALPHA]
if sig_inter:
    say(f"Interaksi nyata: {sig_inter} -> Tukey HSD pada sel gabungan ({'|'.join(FACTORS)}):")
    say(pairwise_tukeyhsd(df["y"], df["cell"], alpha=ALPHA).summary().as_text())
else:
    sig_main = [f for f in FACTORS if tab.loc[f"C({f})", "PR(>F)"] < ALPHA and df[f].nunique() > 2]
    if sig_main:
        for f in sig_main:
            say(f"Tanpa interaksi nyata; faktor {f} nyata -> Tukey HSD pada {f}:")
            say(pairwise_tukeyhsd(df["y"], df[f], alpha=ALPHA).summary().as_text())
    else:
        say("Tidak ada interaksi nyata dan tidak ada faktor >2 level yang nyata -> tidak perlu uji lanjut.")

tab.round(6).to_csv(f"{OUT}/anova_table.csv")
with open(f"{OUT}/anova_report.txt", "w") as f:
    f.write("\n".join(log))
