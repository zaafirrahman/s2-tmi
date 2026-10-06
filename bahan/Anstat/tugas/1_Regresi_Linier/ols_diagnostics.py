"""
Comprehensive OLS Diagnostic Report
=====================================
Fungsi all-in-one buat cek:
1. Korelasi Pearson antar variabel (X-X dan X-Y)
2. Ringkasan model (koefisien, SE, t-stat, p-value, F-test, R², Adj R²)
3. Uji Normalitas Residual (Shapiro-Wilk, Kolmogorov-Smirnov, Jarque-Bera)
4. Uji Homoskedastisitas (Breusch-Pagan, Glejser)
5. Uji Multikolinearitas (VIF)
6. Uji Autokorelasi (Durbin-Watson, Breusch-Godfrey)
7. Outlier & Influential Points (Cook's Distance, Leverage, Studentized Residual)

Cara pakai:
    from ols_diagnostics import full_diagnostic_report
    model, report = full_diagnostic_report(X_train, y_train, feature_names=feature_cols)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
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


def pearson_correlation_report(X, y=None, y_name="Occupancy"):
    """Matriks korelasi Pearson antar variabel independen, + korelasi tiap X ke Y."""
    print("=" * 70)
    print("1. PEARSON CORRELATION MATRIX (antar variabel independen)")
    print("=" * 70)
    corr_matrix = X.corr(method="pearson")
    print(corr_matrix.round(3).to_string())

    if y is not None:
        print("\n--- Korelasi tiap X terhadap Y (%s) ---" % y_name)
        rows = []
        for col in X.columns:
            r, p = stats.pearsonr(X[col], y)
            rows.append({"Variabel": col, "r": round(r, 4), "p-value": round(p, 4),
                         "Sig": _sig_flag(p)})
        corr_y = pd.DataFrame(rows)
        print(corr_y.to_string(index=False))
    print()
    return corr_matrix


def model_summary_report(model, feature_names):
    """Ringkasan koefisien, t-test tiap variabel, dan F-test keseluruhan model."""
    print("=" * 70)
    print("2. MODEL SUMMARY (Coefficients, t-test, F-test)")
    print("=" * 70)

    coef_table = pd.DataFrame({
        "Coef": model.params,
        "Std Error": model.bse,
        "t-stat": model.tvalues,
        "p-value": model.pvalues,
    })
    coef_table["Sig"] = coef_table["p-value"].apply(_sig_flag)
    print(coef_table.round(4).to_string())
    print("\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.10")

    print("\n--- Uji F (signifikansi model keseluruhan) ---")
    print(f"F-statistic : {model.fvalue:.4f}")
    print(f"Prob (F)    : {model.f_pvalue:.6f}  {_sig_flag(model.f_pvalue)}")

    print("\n--- Goodness of Fit ---")
    print(f"R-squared         : {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"AIC               : {model.aic:.2f}")
    print(f"BIC               : {model.bic:.2f}")
    print(f"N observations    : {int(model.nobs)}")
    print()
    return coef_table


def normality_report(residuals):
    """Uji normalitas residual: Shapiro-Wilk, Kolmogorov-Smirnov, Jarque-Bera."""
    print("=" * 70)
    print("3. UJI NORMALITAS RESIDUAL")
    print("=" * 70)

    # Shapiro-Wilk (paling reliable untuk n < ~2000, sensitif juga di n kecil-menengah)
    sw_stat, sw_p = stats.shapiro(residuals)
    print(f"Shapiro-Wilk       : W={sw_stat:.4f}, p-value={sw_p:.4f}  "
          f"{'-> Normal (gagal tolak H0)' if sw_p > 0.05 else '-> TIDAK normal (tolak H0)'}")

    # Kolmogorov-Smirnov (dibandingkan ke distribusi normal dgn mean/std residual)
    ks_stat, ks_p = stats.kstest(residuals, 'norm',
                                  args=(np.mean(residuals), np.std(residuals, ddof=1)))
    print(f"Kolmogorov-Smirnov : D={ks_stat:.4f}, p-value={ks_p:.4f}  "
          f"{'-> Normal (gagal tolak H0)' if ks_p > 0.05 else '-> TIDAK normal (tolak H0)'}")

    # Jarque-Bera
    jb_stat, jb_p, skew, kurtosis = jarque_bera(residuals)
    print(f"Jarque-Bera        : JB={jb_stat:.4f}, p-value={jb_p:.4f}  "
          f"{'-> Normal (gagal tolak H0)' if jb_p > 0.05 else '-> TIDAK normal (tolak H0)'}")
    print(f"  (Skewness={skew:.4f}, Kurtosis={kurtosis:.4f})")

    print("\nH0: residual berdistribusi normal | Ditolak jika p-value < 0.05")
    print()
    return {"shapiro": (sw_stat, sw_p), "ks": (ks_stat, ks_p), "jb": (jb_stat, jb_p)}


def homoskedasticity_report(model, X_with_const):
    """Uji homoskedastisitas: Breusch-Pagan dan Glejser."""
    print("=" * 70)
    print("4. UJI HOMOSKEDASTISITAS")
    print("=" * 70)

    # Breusch-Pagan
    bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(model.resid, X_with_const)
    print(f"Breusch-Pagan  : LM stat={bp_stat:.4f}, p-value={bp_p:.4f}  "
          f"{'-> Homoskedastis (gagal tolak H0)' if bp_p > 0.05 else '-> Heteroskedastis (tolak H0)'}")

    # Glejser: regresikan |residual| terhadap X
    abs_resid = np.abs(model.resid)
    glejser_model = sm.OLS(abs_resid, X_with_const).fit()
    print(f"Glejser        : F-stat={glejser_model.fvalue:.4f}, "
          f"p-value(F)={glejser_model.f_pvalue:.4f}  "
          f"{'-> Homoskedastis (gagal tolak H0)' if glejser_model.f_pvalue > 0.05 else '-> Heteroskedastis (tolak H0)'}")
    print("\n  Detail Glejser per variabel (p-value < 0.05 = variabel itu sumber heteroskedastisitas):")
    glejser_table = pd.DataFrame({
        "Coef": glejser_model.params,
        "p-value": glejser_model.pvalues,
        "Sig": glejser_model.pvalues.apply(_sig_flag),
    })
    print(glejser_table.round(4).to_string())

    print("\nH0: varians residual konstan (homoskedastis) | Ditolak jika p-value < 0.05")
    print()
    return {"breusch_pagan": (bp_stat, bp_p), "glejser_f": (glejser_model.fvalue, glejser_model.f_pvalue)}


def multicollinearity_report(X):
    """Uji multikolinearitas pakai VIF. X tanpa konstanta."""
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


def autocorrelation_report(model, X_with_const, nlags=4):
    """Uji autokorelasi: Durbin-Watson dan Breusch-Godfrey."""
    print("=" * 70)
    print("6. UJI AUTOKORELASI")
    print("=" * 70)

    dw = durbin_watson(model.resid)
    print(f"Durbin-Watson  : DW={dw:.4f}")
    if dw < 1.5:
        dw_note = "-> Indikasi autokorelasi POSITIF"
    elif dw > 2.5:
        dw_note = "-> Indikasi autokorelasi NEGATIF"
    else:
        dw_note = "-> Tidak ada indikasi autokorelasi (mendekati 2)"
    print(f"  {dw_note}  (Aturan kasar: ~2 = tidak ada autokorelasi, <1.5 atau >2.5 = ada indikasi)")

    # Breusch-Godfrey (lebih formal, bisa cek beberapa lag sekaligus)
    bg_stat, bg_p, bg_f, bg_fp = acorr_breusch_godfrey(model, nlags=nlags)
    print(f"\nBreusch-Godfrey (lag={nlags}): LM stat={bg_stat:.4f}, p-value={bg_p:.4f}  "
          f"{'-> Tidak ada autokorelasi (gagal tolak H0)' if bg_p > 0.05 else '-> ADA autokorelasi (tolak H0)'}")

    print("\nPenting: data harus tetap urut kronologis (jangan di-shuffle) sebelum uji ini.")
    print()
    return {"durbin_watson": dw, "breusch_godfrey": (bg_stat, bg_p)}


def outlier_influence_report(model, X_with_const, top_n=10):
    """Deteksi outlier & influential points: Cook's Distance, Leverage, Studentized Residual."""
    print("=" * 70)
    print("7. OUTLIER & INFLUENTIAL POINTS")
    print("=" * 70)

    influence = OLSInfluence(model)
    cooks_d = influence.cooks_distance[0]
    leverage = influence.hat_matrix_diag
    student_resid = influence.resid_studentized_external

    n = int(model.nobs)
    k = len(model.params)  # termasuk konstanta
    cooks_threshold = 4 / n
    leverage_threshold = 2 * k / n

    summary_df = pd.DataFrame({
        "Cooks_D": cooks_d,
        "Leverage": leverage,
        "Student_Resid": student_resid,
    })
    summary_df["Flag_Cooks"] = summary_df["Cooks_D"] > cooks_threshold
    summary_df["Flag_Leverage"] = summary_df["Leverage"] > leverage_threshold
    summary_df["Flag_Outlier"] = summary_df["Student_Resid"].abs() > 3

    n_flagged = summary_df[["Flag_Cooks", "Flag_Leverage", "Flag_Outlier"]].any(axis=1).sum()

    print(f"Threshold Cook's Distance : {cooks_threshold:.4f}  (rule of thumb: 4/n)")
    print(f"Threshold Leverage        : {leverage_threshold:.4f}  (rule of thumb: 2k/n)")
    print(f"Threshold Studentized Res : |t| > 3")
    print(f"\nTotal observasi terflag (minimal satu kriteria): {n_flagged} dari {n}")

    flagged = summary_df[summary_df[["Flag_Cooks", "Flag_Leverage", "Flag_Outlier"]].any(axis=1)]
    if len(flagged) > 0:
        print(f"\nTop {min(top_n, len(flagged))} observasi paling mencurigakan (urut by Cook's D):")
        print(flagged.sort_values("Cooks_D", ascending=False).head(top_n).round(4).to_string())
    else:
        print("\nTidak ada observasi yang terflag sebagai outlier/influential signifikan.")
    print()
    return summary_df


def full_diagnostic_report(X, y, feature_names=None, y_name="Occupancy", bg_lags=4, model_label=""):
    """
    Jalanin semua uji sekaligus dan return (model, dict_of_results).
    X: DataFrame variabel independen (TANPA konstanta, konstanta ditambah otomatis)
    y: Series/array variabel dependen
    """
    if feature_names is None:
        feature_names = list(X.columns)

    print("\n" + "#" * 70)
    print(f"#  FULL DIAGNOSTIC REPORT {'- ' + model_label if model_label else ''}")
    print("#" * 70 + "\n")

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    results = {}
    results["correlation"] = pearson_correlation_report(X, y, y_name=y_name)
    results["model_summary"] = model_summary_report(model, feature_names)
    results["normality"] = normality_report(model.resid)
    results["homoskedasticity"] = homoskedasticity_report(model, X_const)
    results["vif"] = multicollinearity_report(X)
    results["autocorrelation"] = autocorrelation_report(model, X_const, nlags=bg_lags)
    results["outliers"] = outlier_influence_report(model, X_const)

    print("=" * 70)
    print("RINGKASAN CEPAT (checklist asumsi klasik)")
    print("=" * 70)
    checklist = {
        "Normalitas residual (Shapiro-Wilk)": results["normality"]["shapiro"][1] > 0.05,
        "Homoskedastisitas (Breusch-Pagan)": results["homoskedasticity"]["breusch_pagan"][1] > 0.05,
        "Tidak ada autokorelasi (Breusch-Godfrey)": results["autocorrelation"]["breusch_godfrey"][1] > 0.05,
        "Tidak ada multikolinearitas (semua VIF<5)": (results["vif"]["VIF"] < 5).all(),
    }
    for k_, v_ in checklist.items():
        print(f"  [{'OK' if v_ else 'X '}] {k_}")
    print()

    return model, results


def _metrics(y_true, y_pred):
    """Hitung RMSE, MAE, MAPE, R2 dari y_true vs y_pred."""
    resid = y_true - y_pred
    rmse = np.sqrt(np.mean(resid ** 2))
    mae = np.mean(np.abs(resid))
    mape = np.mean(np.abs(resid / y_true)) * 100
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def _print_metrics_row(label, m):
    print(f"{label:10s} -> RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, "
          f"MAPE={m['mape']:.2f}%, R2={m['r2']:.4f}")


def validation_report(model, X_test, y_test, feature_names=None, label="Test Set",
                       clip_lower=0, clip_upper=100):
    """
    Evaluasi model di data validasi (out-of-sample): RMSE, MAE, MAPE, R2.
    Menampilkan DUA versi sekaligus: prediksi mentah (raw) dan prediksi
    yang di-clip ke rentang realistis (default 0-100%, karena Occupancy
    adalah persentase yang secara alami bounded, sementara OLS tidak
    membatasi output-nya).

    Raw dipakai untuk evaluasi statistik/akademis (mencerminkan kemampuan
    murni model linear). Clipped dipakai untuk pelaporan bisnis (angka
    yang masuk akal untuk dikomunikasikan ke manajemen).
    """
    print("=" * 70)
    print(f"VALIDASI MODEL - {label}")
    print("=" * 70)

    X_test_const = sm.add_constant(X_test, has_constant="add")
    # pastikan urutan kolom sama dengan model.params
    X_test_const = X_test_const[model.params.index]

    y_pred_raw = model.predict(X_test_const)
    y_pred_clipped = np.clip(y_pred_raw, clip_lower, clip_upper)

    m_raw = _metrics(y_test, y_pred_raw)
    m_clipped = _metrics(y_test, y_pred_clipped)

    print("--- Raw (prediksi mentah, untuk evaluasi statistik) ---")
    _print_metrics_row("Raw", m_raw)

    n_above = (y_pred_raw > clip_upper).sum()
    n_below = (y_pred_raw < clip_lower).sum()
    if n_above > 0 or n_below > 0:
        print(f"  Peringatan: {n_above} prediksi > {clip_upper}, {n_below} prediksi < {clip_lower} "
              f"(limitasi linear model pada y yang bounded).")

    print(f"\n--- Clipped ke [{clip_lower}, {clip_upper}] (untuk pelaporan bisnis) ---")
    _print_metrics_row("Clipped", m_clipped)

    print(f"\nSelisih R2 (clipped - raw): {m_clipped['r2'] - m_raw['r2']:+.4f}")
    print()

    return {
        "y_pred_raw": y_pred_raw, "y_pred_clipped": y_pred_clipped,
        "raw": m_raw, "clipped": m_clipped,
        # dipertahankan untuk kompatibilitas dengan kode lama yang baca key ini langsung
        "y_pred": y_pred_raw, "rmse": m_raw["rmse"], "mae": m_raw["mae"],
        "mape": m_raw["mape"], "r2_test": m_raw["r2"],
    }


def train_fit_report(model, X_train, y_train, label="Training Set",
                      clip_lower=0, clip_upper=100):
    """
    Evaluasi model di data training itu sendiri (in-sample fit): RMSE, MAE,
    MAPE, R2 - raw vs clipped, dengan struktur yang sama seperti
    validation_report(). Berguna untuk dibandingkan langsung dengan hasil
    validation_report() pada test set, supaya gap train-vs-valid (raw
    ataupun clipped) mudah dicek untuk indikasi overfitting.
    """
    print("=" * 70)
    print(f"FIT MODEL DI DATA TRAINING - {label}")
    print("=" * 70)

    X_train_const = sm.add_constant(X_train, has_constant="add")
    X_train_const = X_train_const[model.params.index]

    y_pred_raw = model.predict(X_train_const)
    y_pred_clipped = np.clip(y_pred_raw, clip_lower, clip_upper)

    m_raw = _metrics(y_train, y_pred_raw)
    m_clipped = _metrics(y_train, y_pred_clipped)

    print("--- Raw (prediksi mentah) ---")
    _print_metrics_row("Raw", m_raw)

    n_above = (y_pred_raw > clip_upper).sum()
    n_below = (y_pred_raw < clip_lower).sum()
    if n_above > 0 or n_below > 0:
        print(f"  Peringatan: {n_above} prediksi > {clip_upper}, {n_below} prediksi < {clip_lower}")

    print(f"\n--- Clipped ke [{clip_lower}, {clip_upper}] ---")
    _print_metrics_row("Clipped", m_clipped)

    print(f"\nSelisih R2 (clipped - raw): {m_clipped['r2'] - m_raw['r2']:+.4f}")
    print("\nCatatan: R2 raw di sini seharusnya identik dengan model.rsquared "
          "dari statsmodels (bagian 2 di full_diagnostic_report).")
    print()

    return {
        "y_pred_raw": y_pred_raw, "y_pred_clipped": y_pred_clipped,
        "raw": m_raw, "clipped": m_clipped,
    }


def train_test_comparison(train_result, test_result):
    """
    Ringkasan cepat perbandingan train vs test (raw & clipped sekaligus),
    dari output train_fit_report() dan validation_report().
    Gap positif besar pada R2 (train jauh > test) adalah indikasi overfitting.
    """
    print("=" * 70)
    print("RINGKASAN PERBANDINGAN TRAIN vs TEST (raw & clipped)")
    print("=" * 70)

    header = f"{'':10s}{'R2 Train':>12s}{'R2 Test':>12s}{'Gap (Train-Test)':>20s}"
    print(header)
    for kind in ["raw", "clipped"]:
        r2_train = train_result[kind]["r2"]
        r2_test = test_result[kind]["r2"]
        gap = r2_train - r2_test
        flag = "  <- indikasi overfitting" if gap > 0.05 else ""
        print(f"{kind:10s}{r2_train:12.4f}{r2_test:12.4f}{gap:+20.4f}{flag}")
    print()