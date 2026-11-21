"""
build_dataset.py
Bangun dataset panel (Date, Ticker) untuk tugas besar TI5002:
  master.csv (tracker radar)  +  komponen SCORE & return ke depan dari yfinance.

Jalankan dari root repo (agar `us_market.engine.screen_score` bisa di-import):
    python build_dataset.py

Keluaran (folder OUT_DIR):
    panel_raw.csv        -> master + komponen + Up5/Ret5 (BELUM disaring)
    failed_tickers.csv   -> ticker gagal / data kurang, beserta alasannya
    validation.txt       -> cek kesesuaian hasil hitung ulang vs master
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from us_market.engine.screen_score import calculate_score
except ImportError:  # salinan identik dari screen_score.py (cadangan)
    def calculate_score(df):
        df = df.copy()
        df["SMA_V20"] = df["Volume"].rolling(20).mean()
        df["H325"] = df["High"].rolling(325).max()
        df["L325"] = df["Low"].rolling(325).min()
        df["P14"] = df["Close"] - df["Close"].shift(14)
        df["VOL_SURGE"] = df["Volume"] / df["SMA_V20"]
        df["RANGE_POS"] = (df["Close"] - df["L325"]) / (df["H325"] - df["L325"]).replace(0, np.nan)
        df["SCORE"] = df["P14"] * df["VOL_SURGE"] * df["RANGE_POS"]
        df = df.iloc[325:].copy()
        df.dropna(inplace=True)
        return df

# ───────────── KONFIGURASI ─────────────
MASTER_CSV = Path("master.csv")
OUT_DIR = Path("dataset_out")
START = "2023-11-01"      # ~3 tahun sebelum akhir data (sama seperti screener.py)
END = "2027-01-01"        # exclusive; longgar -> rerun nanti otomatis ikut data terbaru
HORIZON = 5               # hari bursa ke depan
BATCH = 20                # batch kecil = lebih sopan ke Yahoo
THREADS = 3               # jumlah thread unduh paralel (rendah = lebih aman)
SLEEP_BETWEEN = 3         # detik jeda antar batch
MAX_RETRY = 4             # percobaan ulang per batch jika terindikasi kena limit
BACKOFF = 30              # detik; menunggu BACKOFF * nomor_percobaan
MIN_ROWS = 325 + 60       # 325 warm-up + minimal ~60 titik SCORE agar threshold wajar
HORIZON_COLS = ["Open", "High", "Low", "Close", "Volume"]


# ───────────── INTI PERHITUNGAN (bisa diuji terpisah) ─────────────
def build_features(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """df: OHLCV harian satu ticker. Mengembalikan fitur per tanggal."""
    s = calculate_score(df)
    out = pd.DataFrame(index=s.index)
    out["Close_yf"] = s["Close"]
    out["SCORE_yf"] = s["SCORE"]
    out["Mom14"] = s["P14"] / (s["Close"] - s["P14"]) * 100      # persen
    out["Vol_Surge"] = s["VOL_SURGE"]
    out["Range_Pos"] = s["RANGE_POS"]
    out["ln_Price"] = np.log(s["Close"])
    # return ke depan dihitung pada deret yang sama (kontigu setelah warm-up)
    fwd = s["Close"].shift(-horizon) / s["Close"] - 1
    out["Ret5"] = fwd * 100
    out["Up5"] = np.where(fwd.isna(), np.nan, (fwd > 0).astype(float))
    return out


def _extract(data: pd.DataFrame, ticker: str):
    if isinstance(data.columns, pd.MultiIndex):
        if ticker not in data.columns.get_level_values(0):
            return None
        d = data[ticker]
    else:                       # batch berisi 1 ticker
        d = data
    d = d.dropna(how="all")
    if not set(HORIZON_COLS).issubset(d.columns):
        return None
    d = d[HORIZON_COLS].dropna()
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
    return d


def _n_with_data(data, batch):
    return sum(1 for t in batch if (lambda d: d is not None and not d.empty)(_extract(data, t)))


def process_batch(yf, batch, idx, cache):
    """Unduh + hitung fitur satu batch. Hasil sukses disimpan di cache (bisa dilanjutkan)."""
    cf, cfail = cache / f"feat_{idx:04d}.csv", cache / f"fail_{idx:04d}.csv"
    if cf.exists() and cfail.exists():
        print(f"  batch {idx}: pakai cache")
        f = pd.read_csv(cf, parse_dates=["Date"]) if cf.stat().st_size > 5 else pd.DataFrame()
        fl = pd.read_csv(cfail) if cfail.stat().st_size > 5 else pd.DataFrame(columns=["Ticker", "Alasan"])
        return f, fl

    data = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            data = yf.download(batch, start=START, end=END, group_by="ticker",
                               auto_adjust=True, threads=THREADS, progress=False)
        except Exception as e:
            print(f"  percobaan {attempt}: error unduh ({e})")
            data = None
        # kalau < separuh ticker punya data -> curiga kena rate limit/blokir
        if data is not None and not data.empty and _n_with_data(data, batch) >= 0.5 * len(batch):
            break
        data = None
        if attempt < MAX_RETRY:
            wait = BACKOFF * attempt
            print(f"  terindikasi kena limit; tunggu {wait} detik lalu coba lagi...")
            time.sleep(wait)

    if data is None:  # tetap gagal: JANGAN di-cache supaya run berikutnya mencoba ulang
        return pd.DataFrame(), pd.DataFrame([(t, "unduh gagal / kemungkinan diblokir") for t in batch],
                                            columns=["Ticker", "Alasan"])

    rows, fails = [], []
    for t in batch:
        try:
            d = _extract(data, t)
            if d is None or d.empty:
                fails.append((t, "tidak ada data"))
            elif len(d) < MIN_ROWS:
                fails.append((t, f"riwayat pendek ({len(d)} baris < {MIN_ROWS})"))
            else:
                f = build_features(d)
                f["Ticker"] = t
                rows.append(f.reset_index(names="Date"))
        except Exception as e:
            fails.append((t, f"error hitung: {e}"))
    feat = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    fl = pd.DataFrame(fails, columns=["Ticker", "Alasan"])
    feat.to_csv(cf, index=False)
    fl.to_csv(cfail, index=False)
    return feat, fl


def main():
    import yfinance as yf

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / "cache"
    cache.mkdir(exist_ok=True)
    master = pd.read_csv(MASTER_CSV, parse_dates=["Date"])
    tickers = sorted(master["Ticker"].unique())
    print(f"Ticker di master: {len(tickers)} | batch {BATCH} | thread {THREADS}")

    feats, fails = [], []
    for k, i in enumerate(range(0, len(tickers), BATCH)):
        batch = tickers[i:i + BATCH]
        print(f"[{i + 1}-{i + len(batch)}/{len(tickers)}]")
        f, fl = process_batch(yf, batch, k, cache)
        if len(f):
            feats.append(f)
        fails.append(fl)
        time.sleep(SLEEP_BETWEEN)

    failed = pd.concat(fails, ignore_index=True) if fails else pd.DataFrame(columns=["Ticker", "Alasan"])
    failed.to_csv(OUT_DIR / "failed_tickers.csv", index=False)
    n_block = failed["Alasan"].astype(str).str.contains("diblokir").sum()
    print(f"Berhasil: {len(tickers) - len(failed)} ticker | Gagal/dilewati: {len(failed)}")
    if n_block:
        print(f"PERINGATAN: {n_block} ticker gagal karena kemungkinan diblokir. "
              "Jalankan ulang skrip ini nanti / ganti jaringan (mis. hotspot); batch yang sudah sukses tidak diunduh ulang.")
    if not feats:
        raise SystemExit("Tidak ada data sama sekali. Cek koneksi / pasang ulang: pip install -U yfinance")

    feat = pd.concat(feats, ignore_index=True)
    panel = master.merge(feat, on=["Date", "Ticker"], how="left")
    panel["Trigger"] = np.where(panel["Distance_%"].isna(), np.nan, (panel["Distance_%"] > 0).astype(float))
    panel.to_csv(OUT_DIR / "panel_raw.csv", index=False)

    # validasi: apakah hitung ulang cocok dengan hasil scraping?
    ok = panel.dropna(subset=["SCORE_yf", "Radar_Score"])
    lines = [
        f"Baris master: {len(master)}",
        f"Baris yang berhasil digabung dengan komponen: {panel['SCORE_yf'].notna().sum()} "
        f"({panel['SCORE_yf'].notna().mean() * 100:.1f}%)",
        f"Spearman Radar_Score vs SCORE_yf: {ok['Radar_Score'].corr(ok['SCORE_yf'], method='spearman'):.4f}",
        f"Kesamaan tanda: {(np.sign(ok['Radar_Score']) == np.sign(ok['SCORE_yf'])).mean() * 100:.2f}%",
        f"Median |Price - Close_yf|/Price: "
        f"{((panel['Price'] - panel['Close_yf']).abs() / panel['Price']).median() * 100:.3f}%",
        f"Baris dengan Up5 tersedia: {panel['Up5'].notna().sum()}",
    ]
    (OUT_DIR / "validation.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
