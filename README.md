# Smart Attendance

Smart Attendance adalah sistem presensi berbasis AI yang menggunakan deteksi dan pengenalan wajah untuk mencatat kehadiran sivitas akademika secara otomatis melalui jaringan video dari kamera CCTV atau pemrosesan video rekaman.

## Tech Stack
Sistem ini dibangun menggunakan teknologi berikut:
- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **Database**: SQLite (dikelola melalui [SQLAlchemy](https://www.sqlalchemy.org/) ORM)
- **AI & Face Recognition**:
  - `InsightFace` (buffalo_s model) untuk deteksi wajah (Face Detection) dan ekstraksi fitur (Face Embeddings).
  - `Scikit-Learn` (SVM - Support Vector Machine) untuk klasifikasi pengenalan wajah murni berdasarkan embedding.
- **Video Processing**: OpenCV
- **Templating Engine**: Jinja2

---

## Struktur Direktori

```text
Smart_Attendance/
├── app/
│   ├── ai/                # Modul AI: Face recognition & background monitor
│   ├── api/               # Rute API: CCTV, Lectures, dan WebSocket Stream
│   ├── config/            # File konfigurasi untuk CCTV (cctv_config.py)
│   ├── database/          # Konfigurasi koneksi Database (SQLAlchemy Engine)
│   ├── models/            # Model Database (ORM): Lecture, DetectionLog
│   ├── repositories/      # Layer abstraksi database (CRUD queries)
│   ├── schemas/           # Pydantic Schemas untuk validasi input/output API
│   ├── static/            # File statis (CSS, JS, Images)
│   ├── storage/           # Tempat penyimpanan wajah & snapshot
│   ├── templates/         # Template HTML (Jinja2) untuk UI Web
│   ├── main.py            # Entry point aplikasi FastAPI
│   └── pages.py           # Rute untuk menyajikan template HTML
├── venv/                  # Python Virtual Environment
├── requirements.txt       # Daftar dependensi library Python
├── smart_attendance.db    # File Database SQLite yang otomatis dibuat
├── svm_face_classifier.pkl # Model SVM yang sudah dilatih untuk klasifikasi wajah
└── label_encoder.pkl      # Encoder label untuk mapping antara ID Prediksi dengan Nama
```

---

## Skema Database

Sistem ini menggunakan SQLite dengan 2 tabel utama yang disederhanakan:

1. **`lectures` (Data Dosen/Sivitas)**
   Menyimpan data individu yang terdaftar di sistem presensi.
   - `id`: Primary Key
   - `nis`: Nomor Identitas (Unik, berindeks)
   - `name`: Nama lengkap
   - `gender`: Jenis kelamin
   - `jabatan`: Jabatan akademik/fungsional
   - `program_studi`: Program studi
   - `jabatan_struktural`: Jabatan struktural
   - `images`: Nama file foto pendaftaran
   - `created_at`: Waktu registrasi

2. **`detection_logs` (Riwayat Deteksi Kehadiran)**
   Menyimpan riwayat kehadiran yang terdeteksi dari CCTV.
   - `id`: Primary Key
   - `cctv_id`: ID Kamera CCTV (berdasarkan ID di file konfigurasi)
   - `lecture_id`: Foreign Key ke tabel Lectures (Cascade delete)
   - `first_seen`: Waktu pertama kali wajah terdeteksi dalam rentang waktu tertentu
   - `last_seen`: Waktu terakhir wajah terlihat sebelum hilang dari frame
   - `confidence`: Tingkat kepercayaan prediksi AI
   - `status`: Status kehadiran (contoh: "present")
   - `snapshot_path`: Direktori gambar tangkapan layar wajah saat deteksi awal
   - `last_snapshot_path`: Direktori gambar tangkapan layar wajah saat deteksi akhir
   - `crop_snapshot_path`: Direktori gambar potongan khusus area wajah terdeteksi

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
uvicorn app.main:app --reload
```


