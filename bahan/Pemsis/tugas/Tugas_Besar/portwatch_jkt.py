"""
Tarik data port call harian Tanjung Priok dari IMF PortWatch (terbuka, tanpa API key).
Versi diagnostik: memeriksa layanan, bentuk portid, lalu menarik data.
Output: portwatch_tanjung_priok_harian.csv, ..._bulanan.csv, portwatch_tanjung_priok.png
Jalankan:  pip install requests pandas matplotlib  ->  python tarik_portwatch_jakarta.py
Sumber: IMF PortWatch (estimasi dari data AIS UN Global Platform). Cantumkan atribusi di laporan.
"""
import requests, pandas as pd, matplotlib.pyplot as plt

ROOT = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG"
BASES = [f"{ROOT}/arcgis/rest/services", f"{ROOT}/ArcGIS/rest/services"]
SERVICES = ["Daily_Ports_Data", "Daily_Trade_Data"]
PORT_ID, PORT_NAME = "port514", "Tanjung Priok"

def call(url, **params):
    params.setdefault("f", "json")
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": {"message": str(e)}}

def feats(js):
    return [f["attributes"] for f in js.get("features", [])]

# ---- 1. Cari layanan yang bisa dibuka & lihat contoh baris ----
url, sample = None, []
for b in BASES:
    for s in SERVICES:
        u = f"{b}/{s}/FeatureServer/0/query"
        js = call(u, where="1=1", outFields="*", resultRecordCount=1)
        ok = "error" not in js and feats(js)
        print(f"[{'OK ' if ok else 'gagal'}] {b.split('/')[-3]}/{s}",
              "" if ok else f"-> {js.get('error', {}).get('message', 'kosong')}")
        if ok and url is None:
            url, sample = u, feats(js)
if url is None:
    raise SystemExit("Tidak ada layanan yang bisa dibuka. Kirim seluruh output ini ke asisten.")

print("\nKolom layanan:", list(sample[0].keys()))
print("Contoh baris :", sample[0])

# ---- 2. Cari bentuk filter portid yang menghasilkan data ----
cands = [f"portid='{PORT_ID}'", f"portid='{PORT_ID.upper()}'", f"portname='{PORT_NAME}'"]
where = None
for w in cands:
    js = call(url, where=w, returnCountOnly="true")
    n = js.get("count", 0)
    print(f"filter {w:30s} -> {n} baris")
    if n and where is None:
        where = w
if where is None:
    raise SystemExit("Semua filter kosong. Lihat 'Contoh baris' di atas untuk bentuk portid, lalu kirim ke asisten.")

# ---- 3. Tarik semua halaman ----
rows, offset = [], 0
while True:
    js = call(url, where=where, outFields="*", orderByFields="date",
              resultOffset=offset, resultRecordCount=1000)
    if "error" in js:
        raise SystemExit(f"Error saat paginasi: {js['error']}")
    got = feats(js)
    if not got:
        break
    rows += got; offset += 1000
df = pd.DataFrame(rows)

dcol = "date" if "date" in df.columns else [c for c in df.columns if "date" in c.lower()][0]
df[dcol] = pd.to_datetime(df[dcol], unit="ms") if pd.api.types.is_numeric_dtype(df[dcol]) else pd.to_datetime(df[dcol])
df = df.sort_values(dcol)
df.to_csv("portwatch_tanjung_priok_harian.csv", index=False)
print(f"\nRentang: {df[dcol].min().date()} s.d. {df[dcol].max().date()} | {len(df)} baris")

pc = [c for c in df.columns if c.lower().startswith("portcalls")]
print("Kolom port call:", pc)
main = "portcalls" if "portcalls" in pc else pc[0]
print("\nStatistik harian:\n", df[main].describe().round(2))

m = df.set_index(dcol)[pc].resample("MS").sum()
m.to_csv("portwatch_tanjung_priok_bulanan.csv")
ax = m[main].plot(figsize=(9, 4), title="Port call bulanan Tanjung Priok (IMF PortWatch)")
ax.set_ylabel("jumlah port call"); plt.tight_layout(); plt.savefig("portwatch_tanjung_priok.png", dpi=150)
print("Selesai.")