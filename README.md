# Smart Attendance

Smart Attendance adalah sistem presensi berbasis AI yang menggunakan deteksi dan pengenalan wajah untuk mencatat kehadiran secara otomatis melalui jaringan video dari kamera CCTV atau pemrosesan video rekaman.

## Tech Stack
Sistem ini dibangun menggunakan teknologi berikut:
- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **Database**: SQLite (dikelola melalui [SQLAlchemy](https://www.sqlalchemy.org/) ORM) menggunakan mode **WAL** untuk performa *high-concurrency*.
- **AI & Face Recognition**:
  - `InsightFace` (buffalo_sc model) untuk deteksi wajah (Face Detection) dan ekstraksi fitur (Face Embeddings).
  - `Scikit-Learn` (SVM - Support Vector Machine) untuk klasifikasi pengenalan wajah murni berdasarkan embedding.
- **Video Processing**: OpenCV
- **Templating Engine**: Jinja2

---

## Struktur Direktori

```text
Smart_Attendance/
├── app/
│   ├── ai/                # Modul AI: Face recognition & background monitor
│   ├── api/               # Rute API: CCTV, Detection Logs, Upload, dan WebSocket Stream
│   ├── config/            # File konfigurasi untuk CCTV (cctv_config.py)
│   ├── database/          # Konfigurasi koneksi Database (SQLAlchemy Engine)
│   ├── models/            # Model Database (ORM): DetectionLog
│   ├── repositories/      # Layer abstraksi database (CRUD queries)
│   ├── schemas/           # Pydantic Schemas untuk validasi input/output API
│   ├── static/            # File statis (CSS, JS, Images)
│   ├── storage/           # Tempat penyimpanan wajah & snapshot
│   ├── templates/         # Template HTML (Jinja2) untuk UI Web
│   ├── main.py            # Entry point aplikasi FastAPI
│   └── pages.py           # Rute untuk menyajikan template HTML
├── requirements.txt       # Daftar dependensi library Python
├── smart_attendance.db    # File Database SQLite yang otomatis dibuat
├── svm_face_classifier.pkl # Model SVM yang sudah dilatih untuk klasifikasi wajah
└── label_encoder.pkl      # Encoder label untuk mapping antara ID Prediksi dengan Nama
```

---

## Skema Database

Sistem ini menggunakan struktur data yang disederhanakan dan berjalan secara mandiri (standalone), mendeteksi nama langsung dari model *Pickle*. Hanya terdapat 1 tabel utama:

1. **`detection_logs` (Riwayat Deteksi Kehadiran)**
   Menyimpan riwayat kehadiran yang terdeteksi secara otomatis.
   - `id`: Primary Key
   - `cctv_id`: ID Kamera CCTV tempat wajah terdeteksi
   - `cctv_name`: Nama Kamera CCTV
   - `person_name`: Nama individu (dihasilkan langsung dari label encoder / SVM)
   - `first_seen`: Waktu pertama kali wajah terdeteksi dalam rentang waktu tertentu
   - `last_seen`: Waktu terakhir wajah terlihat dalam satu sesi deteksi
   - `confidence`: Tingkat kepercayaan/akurasi prediksi AI tertinggi
   - `status`: Status kehadiran ("present")
   - `snapshot_path`: Path gambar tangkapan layar penuh terbaik (confidence tertinggi) beserta bounding box
   - `crop_snapshot_path`: Path gambar potongan wajah yang terbaik (confidence tertinggi)

---

## Setup & Instalasi

### 1. Persyaratan (Prerequisites)
Pastikan Anda telah menginstal:
- Python 3.9 atau lebih tinggi

### 2. Instalasi Dependensi
Jalankan perintah berikut untuk menginstal semua library yang dibutuhkan:
```bash
pip install -r requirements.txt
```

### 3. Menjalankan Aplikasi Web
Jalankan server FastAPI menggunakan `uvicorn`:
```bash
python -m uvicorn app.main:app --reload
```

