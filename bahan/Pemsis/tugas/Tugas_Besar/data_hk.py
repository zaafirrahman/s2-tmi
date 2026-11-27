"""
Galat jadwal kapal nyata dari Marine Department Hong Kong via arsip DATA.GOV.HK.
  python tarik_data_hongkong.py discover   -> cari URL resource (terutama ETD kapal non-konvensi)
  python tarik_data_hongkong.py run        -> unduh snapshot, pasangkan, hitung galat
Hasil: hk_eta_ata_pairs.csv (kapal samudra), hk_etd_atd_pairs.csv (kapal non-konvensi), ringkasan di layar.
CATATAN: nama tag XML belum saya lihat langsung; skrip mencetak kolom & mencarinya berdasar kata kunci.
Sumber data: Marine Department HK via DATA.GOV.HK (data dari nakhoda/operator, tanpa jaminan akurasi dari MD).
"""
import sys, json, time, requests, pandas as pd, xml.etree.ElementTree as ET

API = "https://app.data.gov.hk/v1/historical-archive"
DAYS, STEP = 14, 3                       # rentang hari & ambil tiap 3 snapshot (~1 jam; sumber update tiap 20 menit)
END = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)   # arsip tersedia sampai kemarin
START = END - pd.Timedelta(days=DAYS - 1)
f8 = lambda t: t.strftime("%Y%m%d")

# --- Resource kapal samudra (URL terverifikasi dari halaman dataset) ---
OG_ETA  = "https://www.mardep.gov.hk/e_files/en/opendata/RP04005i.XML"   # Vessels due to arrive
OG_ATA  = "https://www.mardep.gov.hk/e_files/en/opendata/RP05005i.XML"   # Arrived in last 36 hours
# --- Resource kapal non-konvensi: ISI dari hasil `discover` ("Due to depart" & "Departed") ---
NC_ETD  = ""
NC_ATD  = ""

def jget(path, **p):
    r = requests.get(f"{API}/{path}", params=p, timeout=60); r.raise_for_status(); return r.json()

def walk(o, out):
    if isinstance(o, dict):
        if "url" in o: out.append(o)
        for v in o.values(): walk(v, out)
    elif isinstance(o, list):
        for v in o: walk(v, out)

def discover():
    js = jget("list-files", start=f8(END), end=f8(END), provider="hk-md", max=200)
    items = []; walk(js, items)
    print(f"{len(items)} resource Marine Department. Cari yang berisi 'Non-Convention' + 'depart':\n")
    for it in items:
        print(it.get("url"), "|", {k: v for k, v in it.items() if k != "url" and isinstance(v, str)})
    print("\nSalin URL 'Due to depart' & 'Departed' kapal non-konvensi ke NC_ETD dan NC_ATD di skrip ini.")

def stamps(url):
    js = jget("list-file-versions", url=url, start=f8(START), end=f8(END))
    ts = js.get("timestamps") or next((v for v in js.values() if isinstance(v, list)), [])
    print(f"  {url.split('/')[-1]}: {len(ts)} versi arsip")
    return ts

def records(b):
    out = []
    for el in ET.fromstring(b).iter():
        kids = list(el)
        if kids and all(len(k) == 0 for k in kids):
            out.append({k.tag.strip().lower(): (k.text or "").strip() for k in kids})
    return out

def collect(url, label):
    rows = []
    for t in stamps(url)[::STEP]:
        r = requests.get(f"{API}/get-file", params={"url": url, "time": t}, timeout=60)
        if r.status_code == 200:
            try:
                for rec in records(r.content): rec["_snap"] = t; rows.append(rec)
            except ET.ParseError: pass
        time.sleep(0.1)
    df = pd.DataFrame(rows)
    df.to_csv(f"hk_{label}_snapshots.csv", index=False)
    print(f"  {label}: {len(df)} baris. Kolom: {list(df.columns)}")
    return df

def ptime(s):
    for f in ("%d/%m/%Y %H:%M", "%Y/%m/%d %H:%M", "%d-%b-%Y %H:%M", "%Y-%m-%d %H:%M:%S"):
        try: return pd.to_datetime(s, format=f)
        except Exception: pass
    return pd.NaT

def col(df, *alts):
    for kws in alts:
        for c in df.columns:
            if all(k in c for k in kws): return c
    raise KeyError(f"kolom {alts} tidak ada di {list(df.columns)}")

def pairs(est, act, key_est, key_act, est_t, act_t, out):
    """Pasangkan tiap snapshot 'estimasi' dengan realisasi pertama setelahnya (<=72 jam) untuk kapal yang sama."""
    est = est.assign(_k=est[key_est].str.upper(), _est=est[est_t].map(ptime), _s=pd.to_datetime(est["_snap"], format="%Y%m%d-%H%M", errors="coerce"))
    act = act.assign(_k=act[key_act].str.upper(), _act=act[act_t].map(ptime)).drop_duplicates(["_k", "_act"])
    m = est.merge(act[["_k", "_act"]], on="_k").dropna(subset=["_est", "_act", "_s"])
    m = m[(m._act >= m._s) & (m._act - m._s <= pd.Timedelta(hours=72))]
    m = m.sort_values("_act").drop_duplicates(["_k", "_s"])           # realisasi terdekat setelah snapshot
    m["lead_jam"] = (m._act - m._s).dt.total_seconds() / 3600           # RTD aktual
    m["est_jam"] = (m._est - m._s).dt.total_seconds() / 3600           # RTD menurut estimasi kapal
    m["galat_jam"] = m.lead_jam - m.est_jam                            # + = realisasi lebih lambat dari estimasi
    m = m[(m.est_jam > -1) & (m.est_jam < 72)]
    m[["_k", "_s", "est_jam", "lead_jam", "galat_jam"]].to_csv(out, index=False)
    print(f"\n{out}: {len(m)} pasangan, {m._k.nunique()} kapal")
    print(m.galat_jam.describe().round(2).to_string())
    print("MAE estimasi kapal (jam):", round(m.galat_jam.abs().mean(), 2),
          "| kuantil galat 10/50/90%:", m.galat_jam.quantile([.1, .5, .9]).round(2).tolist())
    return m

def run():
    print(f"Periode {START.date()} s.d. {END.date()}")
    print("\n[1] Kapal samudra: ETA -> ATA")
    e, a = collect(OG_ETA, "og_eta"), collect(OG_ATA, "og_ata")
    pairs(e, a, col(e, ("call",)), col(a, ("call",)), col(e, ("eta",), ("estimated",)),
          col(a, ("actual", "arrival"), ("arriv",)), "hk_eta_ata_pairs.csv")
    if NC_ETD and NC_ATD:
        print("\n[2] Kapal non-konvensi: ETD -> ATD")
        e, a = collect(NC_ETD, "nc_etd"), collect(NC_ATD, "nc_atd")
        pairs(e, a, col(e, ("vessel_name",)), col(a, ("vessel_name",)), col(e, ("est", "depart")),
              col(a, ("actual", "depart")), "hk_etd_atd_pairs.csv")
    else:
        print("\n[2] Dilewati: NC_ETD / NC_ATD belum diisi (jalankan `discover`).")

if __name__ == "__main__":
    discover() if (sys.argv[1:] or ["run"])[0] == "discover" else run()