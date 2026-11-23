# Sistem Rekomendasi Laboratorium KKSI - Backend REST API

Layanan REST API untuk **Sistem Rekomendasi Asisten Laboratorium Kelompok Keilmuan Sistem Informasi (KKSI)** - Departemen Sistem Informasi, Fakultas Teknologi Informasi, Universitas Andalas.

Backend ini mengimplementasikan **Two-Tier Hybrid Architecture**:
1. **Lapisan 1 (Rule-Based Gatekeeper):** Penyaring kelayakan administratif awal berdasarkan regulasi pendaftaran asisten (IPK minimum 3.00, semester aktif, dan syarat mata kuliah prasyarat).
2. **Lapisan 2 (Hybrid Recommender Engine):** Penggabungan komposit berbobot seimbang antara **Centroid Cosine Similarity** (geometri profil asisten riil) dan **XGBoost Classifier** (probabilitas kecocokan non-linear) dengan parameter $\alpha = 0.5$.

---

## 4 Laboratorium KKSI yang Didukung
- **RDBI:** Rekayasa Data dan Business Intelligence
- **SD:** System Development
- **SE:** System Enterprise
- **TKITI:** Tata Kelola dan Infrastruktur Teknologi Informasi

---

## Endpoint API

| Method | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/` | Informasi status layanan, arsitektur, dan daftar laboratorium |
| `GET` | `/health` | Health check endpoint untuk monitoring deployment (Render / Railway) |
| `POST` | `/predict` | Evaluasi dua lapis inferensi rekomendasi asisten |

### Contoh Request Payload `POST /predict`
```json
{
  "Semester Pendaftaran": "Semester 5",
  "IPK Kumulatif": "3.65",
  "Nilai DDP": "A",
  "Nilai SDA": "A",
  "Nilai PBO": "A-",
  "Nilai Pemrograman Web": "A",
  "Nilai Manajemen Proses Bisnis": "B+",
  "Nilai Infrastruktur TI": "B+",
  "Nilai Sistem Operasi": "B+",
  "Nilai Keamanan Informasi": "A-",
  "Nilai Basis Data": "A",
  "Nilai Perancangan Basis Data": "A",
  "Nilai Data Mining": "A",
  "Tools Database & SQL": 5,
  "Tools Visualisasi Data": 4,
  "Tools Python Analitik": 4,
  "Tools Pemrograman Web/App": 4,
  "Tools Desain UI/UX": 3,
  "Tools Pemodelan BPMN": 3,
  "Tools ERP & Bisnis": 2,
  "Tools Jaringan & OS": 3,
  "Minat Data & Analitik": 5,
  "Minat Coding & Pemrograman": 4,
  "Minat Proses Bisnis & ERP": 3,
  "Minat Server & Jaringan": 3
}
```

---

## Teknologi yang Digunakan
- **Bahasa:** Python 3.10+
- **Framework:** Flask & Flask-CORS
- **Machine Learning:** Scikit-Learn, XGBoost, NumPy, Pandas
- **Serialisasi:** Joblib
- **WSGI Production:** Gunicorn

---

## Cara Menjalankan secara Lokal

1. **Clone repositori:**
   ```bash
   git clone https://github.com/antigravity17/Sistem-Rekomendasi-KKSI-Backend-.git
   cd Sistem-Rekomendasi-KKSI-Backend-
   ```

2. **Buat & aktifkan virtual environment (opsional tetapi disarankan):**
   ```bash
   python -m venv venv
   # Di Windows:
   venv\Scripts\activate
   # Di Linux/macOS:
   source venv/bin/activate
   ```

3. **Pasang dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan server API:**
   ```bash
   python api.py
   ```
   Server akan berjalan secara lokal di `http://127.0.0.1:5000`.

---

## Deployment (Render / Railway / Cloud Run)
Berkas `Procfile` sudah disediakan untuk deployment otomatis:
```
web: gunicorn api:app
```
Port server akan otomatis mendeteksi variabel lingkungan `$PORT`.
