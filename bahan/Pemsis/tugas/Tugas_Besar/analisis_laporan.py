"""
Analisis lengkap untuk laporan HSM: fokus kapal kontainer, konteks Tanjung Priok.
Input (folder yang sama): hk_eta_ata_pairs.csv, hk_og_eta_snapshots.csv
Output: hasil_laporan.json + fig_*.png
"""
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
QS = [0.1, 0.3, 0.5, 0.7, 0.9]; V = np.arange(12, 18.0001, 0.05)
OUT = {}

# ---------- Data ----------
p = pd.read_csv("hk_eta_ata_pairs.csv", parse_dates=["_s"])
p["ATA"] = (p._s + pd.to_timedelta(p.lead_jam, unit="h")).dt.round("5min")
e = pd.read_csv("hk_og_eta_snapshots.csv").assign(_k=lambda d: d.call_sign.str.upper()).drop_duplicates("_k")
p = p.merge(e[["_k", "ship_type"]], on="_k", how="left")
p["cont"] = p.ship_type.str.contains("contain", case=False, na=False).astype(int)
p = p[p.est_jam >= 1].copy(); p["hz"] = pd.cut(p.est_jam, [0, 6, 12, 24, 48, 72])
d = p.sort_values("_s").groupby(["_k", "ATA", "hz"], observed=True).tail(1).dropna(subset=["hz"])
d = d.assign(y=np.maximum(d.lead_jam, 0.3)).sort_values("ATA").reset_index(drop=True)
n = len(d); a, b = int(.6*n), int(.8*n); tr, cal, te = d.iloc[:a], d.iloc[a:b], d.iloc[b:]
tec = te[te.cont == 1].reset_index(drop=True)
OUT["data"] = dict(pasangan_mentah=len(p), panggilan_total=int(d.groupby(["_k","ATA"]).ngroups),
    panggilan_kontainer=int(d[d.cont==1].groupby(["_k","ATA"]).ngroups), obs_total=n, obs_kontainer=int((d.cont==1).sum()),
    latih=len(tr), kalibrasi=len(cal), uji=len(te), uji_kontainer=len(tec),
    periode=[str(d.ATA.min().date()), str(d.ATA.max().date())])

# ---------- Statistik deskriptif galat (kontainer) ----------
dc = d[d.cont == 1].copy(); dc["galat"] = dc.lead_jam - dc.est_jam
g = dc.groupby("hz", observed=True).galat
desc = pd.DataFrame({"n": g.size(), "median": g.median(), "p10": g.quantile(.1), "p90": g.quantile(.9), "sd": g.std(), "MAE": g.apply(lambda x: x.abs().mean())})
OUT["deskriptif_kontainer"] = {str(k): {c: round(float(v), 2) for c, v in r.items()} for k, r in desc.iterrows()}
fig, ax = plt.subplots(figsize=(6.2, 3.6))
data = [dc[dc.hz == h].galat.values for h in desc.index]
ax.boxplot(data, tick_labels=[str(h).replace("(", "").replace("]", "").replace(", ", "–") + " j" for h in desc.index], showfliers=False)
ax.axhline(0, color="gray", lw=.8, ls="--"); ax.set_ylabel("Galat = realisasi − ETA (jam)"); ax.set_xlabel("Horizon ETA")
ax.set_title("Galat estimasi ETA kapal kontainer (HK)"); plt.tight_layout(); plt.savefig("fig_galat.png"); plt.close()

# ---------- Model prediksi ----------
F = ["est_jam", "cont"]
M = {q: GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=120, max_depth=2, learning_rate=0.05, random_state=0).fit(tr[F], tr.y) for q in QS}
pq = lambda X: np.sort(np.column_stack([M[q].predict(X[F]) for q in QS]), axis=1)
Qc = pq(cal); s = np.maximum.reduce([Qc[:,0]-cal.y.values, cal.y.values-Qc[:,-1], np.zeros(len(Qc))])
r = float(np.sort(s)[min(int(np.ceil((len(s)+1)*.8)), len(s))-1])
def prep(X):
    Q = pq(X); Qk = Q.copy(); Qk[:,0] -= r; Qk[:,-1] += r; return Q, Qk
Qt_all, Qk_all = prep(te); Qt, Qk = prep(tec)
def fe(X, Q, Qk_):
    y = X.y.values; return dict(n=len(X), MAE_ETA=float(np.mean(np.abs(y-X.est_jam))), MAE_model=float(np.mean(np.abs(y-Q[:,2]))),
        RMSE_ETA=float(np.sqrt(np.mean((y-X.est_jam)**2))), RMSE_model=float(np.sqrt(np.mean((y-Q[:,2])**2))),
        PICP80=float(np.mean((y>=Qk_[:,0])&(y<=Qk_[:,-1]))), MPIW80=float(np.mean(Qk_[:,-1]-Qk_[:,0])),
        PICP40=float(np.mean((y>=Q[:,1])&(y<=Q[:,3]))), MPIW40=float(np.mean(Q[:,3]-Q[:,1])),
        kuantil_empiris=[float(np.mean(y<=Q[:,j])) for j in range(5)], bias_ETA=float(np.mean(y-X.est_jam)), bias_model=float(np.mean(y-Q[:,2])))
calc = cal[cal.cont == 1]; Qcc = pq(calc)
adj = [float(np.quantile(calc.y.values - Qcc[:, j], QS[j])) for j in range(5)]
Q2 = np.maximum(np.sort(Qt + np.array(adj), axis=1), 0.3); OUT["rekalibrasi"] = dict(n_kalibrasi_kontainer=len(calc), penyesuaian=adj)
OUT["prediksi_kontainer"] = fe(tec, Qt, Qk); OUT["prediksi_kontainer_v2"] = fe(tec, Q2, Q2); OUT["prediksi_semua"] = fe(te, Qt_all, Qk_all); OUT["konformal_r"] = r
fig, ax = plt.subplots(figsize=(4.2, 3.8)); ax.plot([0,1],[0,1],"k--",lw=.8)
ax.plot(QS, OUT["prediksi_kontainer"]["kuantil_empiris"], "o-", label="Kontainer, model awal")
ax.plot(QS, OUT["prediksi_kontainer_v2"]["kuantil_empiris"], "^-", color="#c33", label="Kontainer, setelah rekalibrasi")
ax.set_xlabel("Level kuantil nominal τ"); ax.set_ylabel("Proporsi realisasi ≤ kuantil prediksi"); ax.legend(frameon=False); ax.set_title("Kalibrasi kuantil")
plt.tight_layout(); plt.savefig("fig_kalibrasi.png"); plt.close()

# ---------- Biaya & kebijakan ----------
def cost(v, D, y, q): T = D/v; return q["kappa"]*D*v**2 + q["cw"]*np.maximum(y-T,0) + q["cla"]*np.maximum(T-y,0) + q["c0"]*(T-y > q["delta"])
best = lambda D, ys, q: V[np.argmin(cost(V[:,None], D, ys[None,:], q).mean(axis=1))]
def scen(row, rg, N=400):
    u = rg.uniform(0,1,N); qs = np.r_[0, QS, 1]
    return np.interp(u, qs, np.r_[max(row[0]-(row[1]-row[0])*.5, .3), row, row[-1]+(row[-1]-row[-2])*.5])
ytr = tr.y.values
def run(q, X=tec, Qk_=Qk, shift=0.0, N=400, seed0=1000):
    res = {k: [] for k in ["RETD","P50","P90","E-SAA","P-SAA","Oracle"]}; sp = []
    for i in range(len(X)):
        rg = np.random.default_rng(seed0+i); D = 15.0*X.est_jam.values[i]; ya = X.y.values[i]; row = Qk_[i] + shift
        vs = {"RETD": best(D, np.array([X.est_jam.values[i]]), q), "P50": best(D, np.array([row[2]]), q), "P90": best(D, np.array([row[4]]), q),
              "E-SAA": best(D, rg.choice(ytr, N), q), "P-SAA": best(D, scen(row, rg, N), q), "Oracle": best(D, np.array([ya]), q)}
        for k, v in vs.items(): res[k].append(float(cost(v, D, ya, q)))
        sp.append([vs["RETD"], vs["P-SAA"]])
    out = {k: np.array(v) for k, v in res.items()}; out["_speeds"] = np.array(sp); return out
def ci(x, y, B=3000):
    idx = np.random.default_rng(7).integers(0, len(x), (B, len(x))); rr = 100*(1 - x[idx].mean(1)/y[idx].mean(1)); return [float(v) for v in np.percentile(rr, [2.5, 97.5])]
base = dict(kappa=0.2, cw=100, cla=1000, c0=10000, delta=2)
R = run(base)
OUT["biaya_dasar"] = {k: float(v.mean()) for k, v in R.items() if k[0] != "_"}
OUT["regret_dasar"] = {k: float((R[k]-R["Oracle"]).mean()) for k in ["RETD","P50","P90","E-SAA","P-SAA"]}
OUT["reduksi_PSAA"] = {ref: dict(pct=float(100*(1-R["P-SAA"].mean()/R[ref].mean())), ci=ci(R["P-SAA"], R[ref])) for ref in ["RETD","P50","P90","E-SAA"]}
OUT["share_lebih_murah_vs_RETD"] = float(np.mean(R["P-SAA"] < R["RETD"] - 1e-9)); OUT["share_sama_vs_RETD"] = float(np.mean(np.abs(R["P-SAA"]-R["RETD"]) <= 1e-9))
R2 = run(base, Qk_=Q2)
OUT["biaya_v2"] = {k: float(v.mean()) for k, v in R2.items() if k[0] != "_"}
OUT["reduksi_PSAA_v2"] = {ref: dict(pct=float(100*(1-R2["P-SAA"].mean()/R[ref].mean())), ci=ci(R2["P-SAA"], R[ref])) for ref in ["RETD","P50","P90","E-SAA"]}
OUT["regret_v2"] = float((R2["P-SAA"]-R2["Oracle"]).mean()); OUT["share_lebih_murah_v2"] = float(np.mean(R2["P-SAA"] < R["RETD"] - 1e-9))
OUT["kecepatan_v2"] = float(R2["_speeds"][:,1].mean())
OUT["kecepatan_rata"] = {"RETD": float(R["_speeds"][:,0].mean()), "P-SAA": float(R["_speeds"][:,1].mean())}
# semua kapal (robustness)
Ra = run(base, te, Qk_all)
OUT["robust_semua_kapal"] = {"n": len(te), "RETD": float(Ra["RETD"].mean()), "P-SAA": float(Ra["P-SAA"].mean()),
    "reduksi": float(100*(1-Ra["P-SAA"].mean()/Ra["RETD"].mean())), "ci": ci(Ra["P-SAA"], Ra["RETD"])}
# gambar biaya: selisih terhadap RETD (positif = lebih murah), CI bootstrap berpasangan
names = ["P50", "P90", "E-SAA", "P-SAA", "P-SAA rekal."]; arrs = [R["P50"], R["P90"], R["E-SAA"], R["P-SAA"], R2["P-SAA"]]
pt = [100*(1-a_.mean()/R["RETD"].mean()) for a_ in arrs]; cis = [ci(a_, R["RETD"]) for a_ in arrs]
OUT["vs_RETD"] = {n_: dict(pct=float(p_), ci=c_) for n_, p_, c_ in zip(names, pt, cis)}
fig, ax = plt.subplots(figsize=(6.2, 3.3)); yy = np.arange(len(names))[::-1]
for y_, p_, c_, n_ in zip(yy, pt, cis, names):
    ax.plot(c_, [y_, y_], "-", color="#35597a", lw=2); ax.plot(p_, y_, "o", color="#c33" if "SAA" in n_ else "#35597a")
ax.axvline(0, color="gray", ls="--", lw=.9); ax.set_yticks(yy); ax.set_yticklabels(names); ax.set_xlabel("Penghematan biaya terhadap RETD (%)  (kanan = lebih murah)")
ax.set_title("Perbandingan kebijakan, kapal kontainer (uji, n=%d)\ntitik = rata-rata, garis = CI 95%% bootstrap berpasangan" % len(tec), fontsize=9.5)
plt.tight_layout(); plt.savefig("fig_biaya.png"); plt.close()

# ---------- Verifikasi ----------
ver = {}
i0 = int(np.argmin(np.abs(tec.est_jam.values - 24))); D0 = 15*tec.est_jam.values[i0]; row0 = Qk[i0]
ver["kasus_uji"] = dict(est_jam=float(tec.est_jam.values[i0]), D=float(D0))
conv = {}
for N in [10, 50, 200, 1000, 5000]:
    vv = [best(D0, scen(row0, np.random.default_rng(s_), N), base) for s_ in range(40)]; conv[N] = [float(np.mean(vv)), float(np.std(vv))]
ver["konvergensi_SAA"] = conv
# uji batas: biaya telat sangat tinggi, y kecil -> v=vmax ; biaya tunggu sangat tinggi, y besar -> v=vmin
ver["uji_batas"] = dict(
  vmax_saat_cla_besar=float(best(300, np.array([5.0]), {**base, "cla": 1e7, "c0": 0})),
  vmin_saat_cw_besar=float(best(300, np.array([60.0]), {**base, "cw": 1e7, "c0": 0})),
  v_saat_kappa_nol_y_cocok=float(best(300, np.array([20.0]), {**base, "kappa": 0.0, "c0": 0})))   # T=y -> v=D/y=15
# resolusi grid
Vf = np.arange(12, 18.0001, 0.01); Vc = V
ys0 = scen(row0, np.random.default_rng(1), 2000)
vf = Vf[np.argmin(cost(Vf[:,None], D0, ys0[None,:], base).mean(axis=1))]; vc = Vc[np.argmin(cost(Vc[:,None], D0, ys0[None,:], base).mean(axis=1))]
ver["grid"] = dict(v_0p05=float(vc), v_0p01=float(vf), selisih_biaya=float(abs(cost(vc, D0, ys0, base).mean()-cost(vf, D0, ys0, base).mean())))
# monotonik fisik: naikkan kappa -> kecepatan optimal tidak naik
ver["monotonik_kappa"] = {str(k): float(best(D0, ys0, {**base, "kappa": k})) for k in [0.05, 0.1, 0.2, 0.4, 0.8]}
OUT["verifikasi"] = ver
fig, ax = plt.subplots(figsize=(5.2, 3.4)); Ns = list(conv); ax.plot(Ns, [conv[k][1] for k in Ns], "o-"); ax.set_xscale("log")
ax.set_xlabel("Jumlah skenario N"); ax.set_ylabel("Simpangan baku kecepatan optimal (knot)"); ax.set_title("Konvergensi SAA"); plt.tight_layout(); plt.savefig("fig_konvergensi.png"); plt.close()

# ---------- Sensitivitas ----------
rows = []
for nm, vals in [("c0",[0,10000,20000]), ("kappa",[0.1,0.2,0.4]), ("cla",[500,1000,2000]), ("cw",[50,100,200]), ("delta",[1,2,3,6])]:
    for v in vals:
        x = run({**base, nm: v}); x2 = run({**base, nm: v}, Qk_=Q2); rows.append(dict(PSAA2=float(x2["P-SAA"].mean()), reduksi2=float(100*(1-x2["P-SAA"].mean()/x["RETD"].mean())), ci2=ci(x2["P-SAA"], x["RETD"]), parameter=nm, nilai=v, RETD=float(x["RETD"].mean()), P50=float(x["P50"].mean()), PSAA=float(x["P-SAA"].mean()),
            reduksi=float(100*(1-x["P-SAA"].mean()/x["RETD"].mean())), ci=ci(x["P-SAA"], x["RETD"]), regret_RETD=float((x["RETD"]-x["Oracle"]).mean()), regret_PSAA=float((x["P-SAA"]-x["Oracle"]).mean())))
OUT["sensitivitas"] = rows
pd.DataFrame(rows).to_csv("sensitivitas_kontainer.csv", index=False)
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
for ax, nm, lab in zip(axs, ["c0","kappa","cla"], ["c0 (USD)", "κ (USD/nm·kn²)", "c_la (USD/jam)"]):
    sub = [r_ for r_ in rows if r_["parameter"] == nm]; xs = [r_["nilai"] for r_ in sub]
    ax.plot(xs, [r_["RETD"] for r_ in sub], "o-", color="#777", label="RETD"); ax.plot(xs, [r_["PSAA"] for r_ in sub], "s-", color="#c33", label="P-SAA")
    ax.set_xlabel(lab); ax.set_ylabel("Biaya rata-rata (USD)")
axs[0].legend(frameon=False); plt.tight_layout(); plt.savefig("fig_sensitivitas.png"); plt.close()

# ---------- Analisis error: bias prediksi + derau SAA ----------
bi = {}
for sh in [-3, -1.5, 0, 1.5, 3]:
    x = run(base, shift=sh); x2 = run(base, Qk_=Q2, shift=sh); bi[str(sh)] = dict(PSAA=float(x["P-SAA"].mean()), PSAA2=float(x2["P-SAA"].mean()), RETD=float(x["RETD"].mean()))
OUT["bias_injeksi"] = bi
fig, ax = plt.subplots(figsize=(5.4, 3.4)); xs = [float(k) for k in bi]
ax.plot(xs, [v["PSAA"] for v in bi.values()], "s-", color="#c33", label="P-SAA awal"); ax.plot(xs, [v["PSAA2"] for v in bi.values()], "^-", color="#27a", label="P-SAA rekalibrasi"); ax.axhline(bi["0"]["RETD"], color="#777", ls="--", label="RETD (referensi)")
ax.set_xlabel("Bias yang disuntikkan ke kuantil (jam)"); ax.set_ylabel("Biaya rata-rata (USD)"); ax.legend(frameon=False); plt.tight_layout(); plt.savefig("fig_bias.png"); plt.close()
seeds = [run(base, seed0=s0*10000)["P-SAA"].mean() for s0 in range(1, 9)]
OUT["derau_SAA"] = dict(rata=float(np.mean(seeds)), sd=float(np.std(seeds, ddof=1)))
json.dump(OUT, open("hasil_laporan.json", "w"), indent=1, ensure_ascii=False)
print("selesai")
