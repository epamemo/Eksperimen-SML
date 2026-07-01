# Eksperimen SML — Epafraditus Memoriano (Kriteria 1)

Eksperimen manual + otomatisasi preprocessing untuk dataset **Telco Churn**
(klasifikasi biner: pelanggan churn / tidak).

## Isi
| File / Folder | Keterangan |
|---|---|
| `telco_churn_raw.csv` | Dataset mentah (2000 baris) |
| `Eksperimen_Epafraditus-Memoriano.ipynb` | Notebook: data loading, EDA, preprocessing (Basic) |
| `automate_Epafraditus-Memoriano.py` | Fungsi preprocessing otomatis, mengembalikan data siap latih (Skilled) |
| `telco_churn_preprocessing/` | Output: `train.csv`, `test.csv` (sudah di-impute, encode, scale) |
| `.github/workflows/preprocess.yml` | GitHub Actions: preprocessing otomatis + commit dataset terbaru (Advance) |
| `requirements.txt` | Dependensi |

## Menjalankan
```bash
pip install -r requirements.txt
jupyter notebook Eksperimen_Epafraditus-Memoriano.ipynb   # jalankan semua cell
# atau otomatisasi:
python "automate_Epafraditus-Memoriano.py" --input telco_churn_raw.csv --outdir telco_churn_preprocessing
```

## Level: Advance
- [x] Basic — data loading, EDA, preprocessing di notebook
- [x] Skilled — `automate_*.py` dengan fungsi preprocessing (struktur berbeda dari eksperimen)
- [x] Advance — GitHub Actions preprocessing otomatis, meng-commit dataset hasil proses
