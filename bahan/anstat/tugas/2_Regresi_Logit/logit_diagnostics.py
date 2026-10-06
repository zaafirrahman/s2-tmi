"""
Comprehensive Binary Logistic Regression Diagnostic Report
=============================================================
Fungsi all-in-one buat cek:
1. Korelasi Pearson antar variabel independen (+ korelasi ke Y, sekadar screening awal)
2. Ringkasan model (koefisien, SE, Wald z-test, p-value, Odds Ratio + CI)
3. Uji signifikansi model keseluruhan (Likelihood Ratio Test) & Pseudo R2 (McFadden)
4. Uji kecocokan model (Hosmer-Lemeshow Goodness-of-Fit Test)
5. Uji Multikolinearitas (VIF)
6. Classification Table (Confusion Matrix), Hit Rate, Sensitivity, Specificity
7. Press's Q Statistic (uji apakah hit rate signifikan lebih baik dari tebakan acak)

Cara pakai:
    from logit_diagnostics import full_diagnostic_report, validation_report, train_fit_report
    model, results = full_diagnostic_report(X_train, y_train, feature_names=feature_cols)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats


def _sig_flag(p):
    """Kasih bintang signifikansi ala output SPSS/Minitab."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.10:
        return "."
    return ""


# ---------------------------------------------------------------------------
# 1. KORELASI PEARSON
# ---------------------------------------------------------------------------
def pearson_correlation_report(X, y=None, y_name="BrandPilihan"):
    """Matriks korelasi Pearson antar variabel independen, + korelasi tiap X ke Y.
    Catatan: ini cuma screening awal (linear correlation). Untuk Y biner, korelasi
    point-biserial ini masih informatif sebagai indikasi arah hubungan, tapi
    signifikansi resmi tetap mengacu ke Wald test pada model logit."""
    print("=" * 70)
    print("1. PEARSON CORRELATION MATRIX (antar variabel independen)")
    print("=" * 70)
    corr_matrix = X.corr(method="pearson")
    print(corr_matrix.round(3).to_string())

    if y is not None:
        print("\n--- Korelasi tiap X terhadap Y (%s, 1=kategori target) ---" % y_name)
        rows = []
        for col in X.columns:
            r, p = stats.pointbiserialr(y, X[col]) if X[col].nunique() > 2 else stats.pearsonr(X[col], y)
            rows.append({"Variabel": col, "r": round(r, 4), "p-value": round(p, 4),
                         "Sig": _sig_flag(p)})
        corr_y = pd.DataFrame(rows)
        print(corr_y.to_string(index=False))
    print()
    return corr_matrix


# ---------------------------------------------------------------------------
# 2. MODEL SUMMARY: KOEFISIEN, WALD TEST, ODDS RATIO
# ---------------------------------------------------------------------------
def model_summary_report(model, feature_names=None):
    """Ringkasan koefisien, Wald test (z-test) tiap variabel, dan Odds Ratio + CI 95%."""
    print("=" * 70)
    print("2. MODEL SUMMARY (Coefficients, Wald Test, Odds Ratio)")
    print("=" * 70)

    conf = model.conf_int()
    conf.columns = ["CI_2.5%", "CI_97.5%"]

    coef_table = pd.DataFrame({
        "Coef (B)": model.params,
        "Std Error": model.bse,
        "Wald (z)": model.tvalues,
        "p-value": model.pvalues,
    })
    coef_table["Sig"] = coef_table["p-value"].apply(_sig_flag)
    coef_table["Odds Ratio"] = np.exp(model.params)
    coef_table["OR CI 2.5%"] = np.exp(conf["CI_2.5%"])
    coef_table["OR CI 97.5%"] = np.exp(conf["CI_97.5%"])

    print(coef_table.round(4).to_string())
    print("\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.10")
    print("\nInterpretasi Odds Ratio (OR):")
    print("  OR > 1 -> menaikkan peluang Y=1 (kategori target)")
    print("  OR < 1 -> menurunkan peluang Y=1")
    print("  OR = 1 (mendekati) -> tidak berpengaruh")
    print()
    return coef_table


# ---------------------------------------------------------------------------
# 3. UJI SIGNIFIKANSI MODEL KESELURUHAN (LR TEST) & PSEUDO R2
# ---------------------------------------------------------------------------
def overall_significance_report(model):
    """Likelihood Ratio Test (model vs model konstanta saja) + McFadden Pseudo R2."""
    print("=" * 70)
    print("3. UJI SIGNIFIKANSI MODEL KESELURUHAN (Likelihood Ratio Test)")
    print("=" * 70)

    print(f"Log-Likelihood (model)   : {model.llf:.4f}")
    print(f"Log-Likelihood (null)    : {model.llnull:.4f}")
    print(f"LR chi2({model.df_model:.0f})            : {model.llr:.4f}")
    print(f"Prob > chi2 (LR p-value) : {model.llr_pvalue:.6f}  {_sig_flag(model.llr_pvalue)}")
    if model.llr_pvalue < 0.05:
        print("  -> Model signifikan lebih baik daripada model tanpa prediktor (tolak H0)")
    else:
        print("  -> Model TIDAK signifikan lebih baik daripada model konstanta saja (gagal tolak H0)")

    print("\n--- Goodness of Fit ---")
    print(f"McFadden Pseudo R2       : {model.prsquared:.4f}")
    print("  Rule of thumb (McFadden): 0.2-0.4 -> good fit; >0.4 -> sangat baik "
          "(catatan: ambang ini jauh lebih rendah drpd R2 OLS, wajar untuk model logit)")
    print(f"AIC                      : {model.aic:.2f}")
    print(f"BIC                      : {model.bic:.2f}")
    print(f"N observations           : {int(model.nobs)}")
    print()
    return {
        "llf": model.llf, "llnull": model.llnull, "llr": model.llr,
        "llr_pvalue": model.llr_pvalue, "prsquared": model.prsquared,
        "aic": model.aic, "bic": model.bic,
    }


# ---------------------------------------------------------------------------
# 4. HOSMER-LEMESHOW GOODNESS-OF-FIT TEST
# ---------------------------------------------------------------------------
def hosmer_lemeshow_test(y_true, y_pred_prob, g=10):
    """
    Uji Hosmer-Lemeshow: membandingkan frekuensi observasi vs ekspektasi
    dalam g kelompok (biasanya decile) berdasarkan urutan predicted probability.
    H0: model FIT dengan data (tidak ada perbedaan signifikan observed vs expected).
    Ditolak jika p-value < 0.05 (artinya model TIDAK fit dengan baik).
    """
    print("=" * 70)
    print("4. UJI HOSMER-LEMESHOW (Goodness of Fit)")
    print("=" * 70)

    data = pd.DataFrame({"y": np.asarray(y_true), "prob": np.asarray(y_pred_prob)})
    data = data.sort_values("prob").reset_index(drop=True)

    try:
        data["group"] = pd.qcut(data["prob"], g, duplicates="drop")
    except ValueError:
        data["group"] = pd.qcut(data["prob"].rank(method="first"), g, duplicates="drop")

    n_groups = data["group"].nunique()

    grouped = data.groupby("group", observed=True)
    obs_1 = grouped["y"].sum()
    n_g = grouped["y"].count()
    exp_1 = grouped["prob"].sum()
    obs_0 = n_g - obs_1
    exp_0 = n_g - exp_1

    # hindari pembagian oleh nol
    eps = 1e-10
    hl_stat = (((obs_1 - exp_1) ** 2 / (exp_1 + eps)) + ((obs_0 - exp_0) ** 2 / (exp_0 + eps))).sum()
    df = n_groups - 2
    df = max(df, 1)
    hl_pvalue = 1 - stats.chi2.cdf(hl_stat, df)

    table = pd.DataFrame({
        "N": n_g, "Observed_1": obs_1, "Expected_1": exp_1.round(2),
        "Observed_0": obs_0, "Expected_0": exp_0.round(2),
    })
    print(table.to_string())
    print(f"\nHosmer-Lemeshow chi2({df}) = {hl_stat:.4f}, p-value = {hl_pvalue:.4f}")
    if hl_pvalue > 0.05:
        print("  -> Model FIT dengan baik terhadap data (gagal tolak H0)")
    else:
        print("  -> Model TIDAK fit dengan baik terhadap data (tolak H0) - waspada mis-spesifikasi")
    print("\nH0: tidak ada perbedaan signifikan antara observed vs expected | Ditolak jika p-value < 0.05")
    print()
    return {"hl_stat": hl_stat, "hl_pvalue": hl_pvalue, "df": df, "table": table}


# ---------------------------------------------------------------------------
# 5. MULTIKOLINEARITAS (VIF)
# ---------------------------------------------------------------------------
def multicollinearity_report(X):
    """Uji multikolinearitas pakai VIF. X tanpa konstanta.
    Catatan: VIF dihitung dari hubungan linear antar-X saja, jadi valid dipakai
    apa adanya meskipun model utamanya adalah logit (bukan OLS)."""
    print("=" * 70)
    print("5. UJI MULTIKOLINEARITAS (VIF)")
    print("=" * 70)

    X_const = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variabel"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i + 1)
                        for i in range(len(X.columns))]
    vif_data["Keterangan"] = vif_data["VIF"].apply(
        lambda v: "AMAN" if v < 5 else ("WASPADA" if v < 10 else "BERMASALAH")
    )
    print(vif_data.round(3).to_string(index=False))
    print("\nRule of thumb: VIF < 5 aman, 5-10 waspada, > 10 multikolinearitas serius")
    print()
    return vif_data


# ---------------------------------------------------------------------------
# 6. CLASSIFICATION TABLE / CONFUSION MATRIX / HIT RATE
# ---------------------------------------------------------------------------
def classification_report_logit(y_true, y_pred_prob, threshold=0.5,
                                  labels=("0", "1"), label_name="Y"):
    """Confusion matrix, hit rate (accuracy), sensitivity, specificity, precision."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_pred_prob) >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n = tp + tn + fp + fn

    hit_rate = (tp + tn) / n
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan   # recall utk kelas 1
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan   # recall utk kelas 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan

    cm = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=[f"Actual {labels[0]}", f"Actual {labels[1]}"],
        columns=[f"Pred {labels[0]}", f"Pred {labels[1]}"],
    )

    print(f"--- Classification Table (threshold = {threshold}) ---")
    print(cm.to_string())
    print(f"\nN                  : {n}")
    print(f"Hit Rate (Akurasi) : {hit_rate:.4f}  ({hit_rate*100:.2f}%)")
    print(f"Sensitivity (Recall {labels[1]}) : {sensitivity:.4f}")
    print(f"Specificity (Recall {labels[0]}) : {specificity:.4f}")
    print(f"Precision ({labels[1]})          : {precision:.4f}")

    # Proportional chance criterion (Hair et al.) sebagai pembanding hit rate
    p1 = y_true.mean()
    p0 = 1 - p1
    prop_chance = p1 ** 2 + p0 ** 2
    print(f"\nProportional Chance Criterion : {prop_chance:.4f} "
          f"(hit rate model sebaiknya > 1.25x ini = {1.25*prop_chance:.4f}, aturan Hair et al.)")
    print(f"  -> {'Model MEMENUHI kriteria (>1.25x chance)' if hit_rate > 1.25*prop_chance else 'Model BELUM memenuhi kriteria 1.25x chance'}")
    print()

    return {
        "confusion_matrix": cm, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "hit_rate": hit_rate, "sensitivity": sensitivity, "specificity": specificity,
        "precision": precision, "prop_chance": prop_chance, "n": n,
    }


# ---------------------------------------------------------------------------
# 7. PRESS'S Q STATISTIC
# ---------------------------------------------------------------------------
def press_q_test(n_correct, n_total, k=2):
    """
    Press's Q Statistic (Hair et al.): menguji apakah hit rate klasifikasi
    signifikan lebih baik daripada tebakan acak (chance classification).
    Q = [N - (n_correct * K)]^2 / [N * (K - 1)]  ~ chi2(1)
    Ditolak (signifikan) jika Q > 3.841 (chi2 kritis, df=1, alpha=0.05).
    """
    print("--- Press's Q Statistic ---")
    q = ((n_total - (n_correct * k)) ** 2) / (n_total * (k - 1))
    crit = stats.chi2.ppf(0.95, df=1)
    p_value = 1 - stats.chi2.cdf(q, df=1)
    print(f"Press's Q = {q:.4f}  (nilai kritis chi2(1, 0.05) = {crit:.4f}, p-value = {p_value:.4f})")
    if q > crit:
        print("  -> Signifikan (Q > 3.841): hit rate model JAUH LEBIH BAIK dari tebakan acak")
    else:
        print("  -> TIDAK signifikan: hit rate model belum terbukti lebih baik dari tebakan acak")
    print()
    return {"press_q": q, "critical_value": crit, "p_value": p_value}


# ---------------------------------------------------------------------------
# FULL DIAGNOSTIC REPORT (TRAINING)
# ---------------------------------------------------------------------------
def full_diagnostic_report(X, y, feature_names=None, y_name="BrandPilihan",
                            model_label="", threshold=0.5, labels=("0", "1"), hl_groups=10):
    """
    Jalanin semua uji sekaligus dan return (model, dict_of_results).
    X: DataFrame variabel independen (TANPA konstanta, konstanta ditambah otomatis)
    y: Series/array variabel dependen BINER (0/1)
    """
    if feature_names is None:
        feature_names = list(X.columns)

    print("\n" + "#" * 70)
    print(f"#  FULL DIAGNOSTIC REPORT (Binary Logit) {'- ' + model_label if model_label else ''}")
    print("#" * 70 + "\n")

    X_const = sm.add_constant(X)
    model = sm.Logit(y, X_const).fit(disp=0)

    results = {}
    results["correlation"] = pearson_correlation_report(X, y, y_name=y_name)
    results["model_summary"] = model_summary_report(model, feature_names)
    results["overall"] = overall_significance_report(model)

    y_pred_prob = model.predict(X_const)
    results["hosmer_lemeshow"] = hosmer_lemeshow_test(y, y_pred_prob, g=hl_groups)
    results["vif"] = multicollinearity_report(X)

    print("=" * 70)
    print("6. CLASSIFICATION TABLE (In-Sample, Training)")
    print("=" * 70)
    results["classification"] = classification_report_logit(
        y, y_pred_prob, threshold=threshold, labels=labels, label_name=y_name)

    print("=" * 70)
    print("7. PRESS'S Q STATISTIC (Training)")
    print("=" * 70)
    n_correct = results["classification"]["tp"] + results["classification"]["tn"]
    results["press_q"] = press_q_test(n_correct, results["classification"]["n"])

    print("=" * 70)
    print("RINGKASAN CEPAT (checklist kualitas model logit)")
    print("=" * 70)
    checklist = {
        "Model signifikan (LR test p<0.05)": results["overall"]["llr_pvalue"] < 0.05,
        "Model fit dgn data (Hosmer-Lemeshow p>0.05)": results["hosmer_lemeshow"]["hl_pvalue"] > 0.05,
        "Tidak ada multikolinearitas (semua VIF<5)": (results["vif"]["VIF"] < 5).all(),
        "Hit rate > 1.25x proportional chance": results["classification"]["hit_rate"] > 1.25 * results["classification"]["prop_chance"],
        "Press's Q signifikan (> chance)": results["press_q"]["press_q"] > results["press_q"]["critical_value"],
    }
    for k_, v_ in checklist.items():
        print(f"  [{'OK' if v_ else 'X '}] {k_}")
    print()

    return model, results


# ---------------------------------------------------------------------------
# VALIDATION REPORT (HOLD-OUT / TEST SET)
# ---------------------------------------------------------------------------
def validation_report(model, X_test, y_test, label="Test Set", threshold=0.5, labels=("0", "1")):
    """
    Evaluasi model di data validasi (out-of-sample): classification table,
    hit rate, sensitivity, specificity, dan Press's Q.
    """
    print("=" * 70)
    print(f"VALIDASI MODEL - {label}")
    print("=" * 70)

    X_test_const = sm.add_constant(X_test, has_constant="add")
    X_test_const = X_test_const[model.params.index]

    y_pred_prob = model.predict(X_test_const)

    cls = classification_report_logit(y_test, y_pred_prob, threshold=threshold, labels=labels)
    n_correct = cls["tp"] + cls["tn"]
    pq = press_q_test(n_correct, cls["n"])

    return {"y_pred_prob": y_pred_prob, "classification": cls, "press_q": pq}


def train_fit_report(model, X_train, y_train, label="Training Set", threshold=0.5, labels=("0", "1")):
    """
    Evaluasi model di data training itu sendiri (in-sample fit), dengan struktur
    yang sama seperti validation_report(), supaya gap train-vs-valid mudah dicek.
    """
    print("=" * 70)
    print(f"FIT MODEL DI DATA TRAINING - {label}")
    print("=" * 70)

    X_train_const = sm.add_constant(X_train, has_constant="add")
    X_train_const = X_train_const[model.params.index]

    y_pred_prob = model.predict(X_train_const)

    cls = classification_report_logit(y_train, y_pred_prob, threshold=threshold, labels=labels)
    n_correct = cls["tp"] + cls["tn"]
    pq = press_q_test(n_correct, cls["n"])

    return {"y_pred_prob": y_pred_prob, "classification": cls, "press_q": pq}


def train_test_comparison(train_result, test_result):
    """Ringkasan cepat perbandingan hit rate train vs test."""
    print("=" * 70)
    print("RINGKASAN PERBANDINGAN TRAIN vs TEST")
    print("=" * 70)

    hr_train = train_result["classification"]["hit_rate"]
    hr_test = test_result["classification"]["hit_rate"]
    gap = hr_train - hr_test
    flag = "  <- indikasi overfitting" if gap > 0.05 else ""
    print(f"{'Hit Rate':12s}{'Train':>10s}{'Test':>10s}{'Gap':>12s}")
    print(f"{'':12s}{hr_train:10.4f}{hr_test:10.4f}{gap:+12.4f}{flag}")
    print()
