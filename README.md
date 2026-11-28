# Repo S2 TMI

Vault belajar dan riset untuk **S2 Teknik dan Manajemen Industri, ITB**. Isinya bahan kuliah, latihan soal, alat bantu penelitian, dan sedikit hiburan, semuanya statis, ringan, dan bisa dibuka dari HP.

**Buka di:** <https://repo-tmi.pages.dev/>

Situsnya juga bisa dipasang sebagai PWA ("Add to Home Screen"), jadi terasa seperti aplikasi.

> Reinforcing loop: belajar → berlatih → eksekusi riset → mabar → belajar lagi.

## Isi

| Bagian | Isinya |
|---|---|
| **Bahan** (`bahan/`) | Materi dan tugas kuliah per mata kuliah. Ada viewer bawaan untuk `ipynb`, `py`, `pdf`, `docx`, `xlsx`, `csv`, dan `html`, dengan pencarian file dan link yang bisa dibagikan. PDF yang diproteksi password bisa dibuka lewat viewer (password diketik di browser dan tidak dikirim ke mana pun). |
| **Latihan** (`latihan/`) | Latihan pilihan ganda untuk persiapan kuis, UTS, dan UAS, per mata kuliah dan per materi. Tiap mata kuliah punya latihan per bab, cheat sheet (HTML dan PDF), dan paket simulasi UTS dari Easy sampai Expert. Mata kuliah Optim juga punya toolkit visualisasi metode numerik. |
| **Stack Riset** (`riset/`) | Alat bantu penelitian: katalog tesis terdahulu, protokol SLR, dan bengkel teknis penulisan tesis. Rincian di bawah. |
| **Mabar** (`mabar/`) | Mini games dengan leaderboard (estim, Pixelcode) dan tautan ke komunitas Discord. |

### Mata kuliah

| Kode | Singkatan | Nama |
|---|---|---|
| TI5001 | Optim | Optimisasi & Intelejensia Komputasional |
| TI5002 | Anstat | Analisis Data dan Statistik |
| TI5003 | Pemsis | Pemodelan dan Berpikir Sistem |
| TI5004 | Metlit | Metodologi Penelitian dan Literasi Digital |

### Stack Riset

- **Katalog Tesis**: tesis terdahulu yang sudah dipetakan menurut dosen, kelompok keahlian (KK), dan subjek, bersumber dari data perpustakaan ITB. Situsnya terpisah: <https://thesis-tmi.pages.dev>.
- **SLR Protocol**: panduan dan alat untuk menyusun systematic literature review secara terstruktur, mulai dari arah riset, taksonomi, sampai PRISMA. Aplikasinya ada di repo [`zaafirrahman/slr-protocol`](https://github.com/zaafirrahman/slr-protocol) (khusus Windows), halaman di sini berisi panduan instalasinya.
- **Bengkel Redaksi**: alat bantu teknis menulis tesis.
  - *Matrix Matcher*: membantu menentukan metode atau model yang cocok untuk penelitian.
  - *LaTeX Thesis Formatter*: template naskah tesis sesuai panduan dan format yang ditentukan.

## Struktur

```
.
├── index.html            # hub utama
├── assets/               # logo dan ikon (png, svg)
├── bahan/                # viewer materi dan tugas (isi file diletakkan di sini)
├── latihan/              # anstat, metlit, optim, pemsis
├── riset/                # index (Stack Riset), bengkel, matrix-matcher,
│                         # latex-formatter, slr-protocol
├── mabar/                # index (Mabar) dan mini games
├── icons/                # ikon PWA
├── manifest.json         # manifest PWA
├── sw.js                 # service worker
├── scripts/gen_manifest.py
└── .github/workflows/pages.yml
```

## Menambah bahan

1. Taruh file di dalam `bahan/`, boleh dalam subfolder (misalnya `bahan/Anstat/materi/`).
2. Commit dan push ke `main`.

Daftar file di viewer dibuat otomatis oleh `scripts/gen_manifest.py` setiap kali GitHub Actions berjalan, jadi tidak perlu mengedit apa pun secara manual. Ikon folder di sidebar bisa diatur lewat dictionary `FI` di `bahan/index.html`.

## Menjalankan lokal

Tidak ada langkah build. Cukup jalankan server statis dari root repo:

```bash
python scripts/gen_manifest.py   # membuat bahan/manifest.json
python -m http.server 8000
```

Lalu buka <http://localhost:8000>.

## Deploy

Setiap push ke `main` men-deploy seluruh repo ke GitHub Pages lewat workflow `Deploy materi`. Di pengaturan repo, set **Pages → Source** ke **GitHub Actions**.

## Catatan

Proyek ini inisiatif pribadi dan tidak berafiliasi resmi dengan ITB. Hak cipta materi kuliah tetap milik masing-masing dosen dan pengampu; gunakan untuk keperluan belajar.

---

Forged by [@Zaafirrahman](https://zaafirrahman.github.io)