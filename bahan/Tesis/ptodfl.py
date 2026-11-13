"""
Contoh mini PTO vs DFL: masalah newsvendor (kasus: berapa unit stok dipesan).
Hanya butuh numpy. Jalankan: python pto_vs_dfl_newsvendor.py
"""
import numpy as np

rng = np.random.default_rng(0)

# ---------- 1. DUNIA (data sintetis) ----------
# x = fitur (mis. indikator hari ramai), d = demand sebenarnya (baru ketahuan setelah memesan)
def make_data(n):
    x = rng.uniform(0, 1, n)
    d = 100 + 40 * x + rng.normal(0, 5 + 25 * x)   # makin ramai, makin tak pasti
    return x, d

x_tr, d_tr = make_data(3000)
x_te, d_te = make_data(20000)

# ---------- 2. KEPUTUSAN & BIAYA ----------
CU, CO = 9.0, 1.0                                   # biaya kurang stok vs kelebihan stok
def cost(q, d):                                     # biaya nyata setelah d terungkap
    return CU * np.maximum(d - q, 0) + CO * np.maximum(q - d, 0)

def solver(c_hat):
    # "Lapisan optimasi": min_q E[cost(q,d)] dengan d dianggap = c_hat.
    # Di toy ini jawabannya q = c_hat. Di kasus nyata, ini MILP/Gurobi (kotak hitam).
    return c_hat

# regret = biaya keputusan dari prediksi - biaya keputusan dari kebenaran (=0 di sini)
regret = lambda q, d: cost(q, d).mean()

# ---------- 3. PTO: latih model dengan MSE (langkah predict), lalu solver ----------
A = np.c_[np.ones_like(x_tr), x_tr]
w_pto = np.linalg.lstsq(A, d_tr, rcond=None)[0]
predict = lambda w, x: w[0] + w[1] * x

# ---------- 4. DFL: latih model dengan regret, gradien via perturbasi (zeroth-order) ----------
w = w_pto.copy()                                    # warm-up dari MSE (seperti di Guo et al.)
sigma, lr, K, epochs = 5.0, 0.05, 20, 600
for ep in range(epochs):
    c_hat = predict(w, x_tr)
    f0 = cost(solver(c_hat), d_tr)                  # loss tanpa gangguan
    g = np.zeros_like(c_hat)
    for _ in range(K):                              # rata-rata K sampel noise
        eps = rng.normal(size=c_hat.shape)
        f1 = cost(solver(c_hat + sigma * eps), d_tr)  # panggil solver pada prediksi yang diganggu
        g += (f1 - f0) / sigma * eps                  # Eq. 26 (estimator zeroth-order)
    g /= K
    # rantai ke parameter model (backprop biasa): dc_hat/dw = [1, x]
    grad_w = np.array([g.mean(), (g * x_tr).mean()])
    w -= lr * grad_w

# ---------- 5. EVALUASI di data uji ----------
def report(name, w):
    ch = predict(w, x_te)
    print(f"{name:<22} MAE={np.abs(ch - d_te).mean():6.2f}   regret(biaya)={regret(solver(ch), d_te):7.2f}   "
          f"koefisien=({w[0]:.1f}, {w[1]:.1f})")

print("--- hasil pada data uji ---")
report("PTO (latih MSE)", w_pto)
report("DFL (latih regret)", w)
# Pembanding teoretis: kuantil 90% = rasio kritis CU/(CU+CO)
w_q = np.array([100 + 1.2816 * 5, 40 + 1.2816 * 25])
report("Kuantil 90% (acuan)", w_q)