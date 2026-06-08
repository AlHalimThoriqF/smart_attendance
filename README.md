# Smart Attendance

Smart Attendance adalah sistem presensi (kehadiran) real-time berbasis AI yang menggunakan deteksi dan pengenalan wajah untuk mencatat kehadiran sivitas akademika secara otomatis melalui jaringan video (RTSP) dari kamera CCTV.

## Fitur Utama
- **Real-time Face Recognition**: Mendeteksi dan mengenali wajah dari kamera CCTV secara real-time.
- **Background Stream Processing**: Mendukung pemrosesan aliran video (RTSP stream) dari banyak kamera CCTV secara bersamaan di latar belakang.
- **Manajemen Kamera CCTV**: Menambahkan, mengedit, dan menghapus konfigurasi kamera CCTV beserta alamat RTSP-nya.
- **Manajemen Data Sivitas (Lectures)**: Registrasi data wajah mahasiswa/dosen/staf lengkap dengan informasi dasar (NIS, Nama, Jenis Kelamin).
- **Log Kehadiran (Detection Logs)**: Mencatat riwayat kehadiran secara rinci lengkap dengan tingkat kepercayaan (confidence) dari AI dan timestamp.
- **WebSocket Video Streaming**: Melihat langsung tangkapan kamera CCTV yang telah diproses oleh AI secara real-time dari antarmuka web.
- **Web Dashboard**: Antarmuka responsif berbasis HTML/CSS/JS (menggunakan Jinja2 templates) untuk memantau sistem.

## Tech Stack
Sistem ini dibangun menggunakan teknologi berikut:
- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **Database**: SQLite (dikelola melalui [SQLAlchemy](https://www.sqlalchemy.org/) ORM)
- **Database Migration**: [Alembic](https://alembic.sqlalchemy.org/)
- **AI & Face Recognition**:
  - `InsightFace` (buffalo_sc model) untuk deteksi wajah (Face Detection) dan ekstraksi fitur (Face Embeddings).
  - `Scikit-Learn` (SVM - Support Vector Machine) untuk klasifikasi wajah yang telah dilatih dengan data sivitas.
- **Video Processing**: OpenCV
- **Templating Engine**: Jinja2

---

## Struktur Direktori

```text
Smart_Attendance/
├── app/
│   ├── ai/                # Modul AI: Face recognition (InsightFace)& background monitor
│   ├── api/               # Rute API: Auth, CCTV, Lectures, dan WebSocket Stream
│   ├── core/              # Konfigurasi keamanan (JWT Token, Hash Password)
│   ├── database/          # Konfigurasi koneksi Database (SQLAlchemy Engine)
│   ├── models/            # Model Database (ORM): CCTV, User, Lecture, DetectionLog
│   ├── repositories/      # Layer abstraksi database (CRUD queries)
│   ├── schemas/           # Pydantic Schemas untuk validasi input/output API
│   ├── static/            # File statis (CSS, JS, Images)
│   ├── storage/           # Tempat penyimpanan file upload (foto pendaftaran)
│   ├── templates/         # Template HTML (Jinja2) untuk UI Web
│   ├── main.py            # Entry point aplikasi FastAPI
│   └── pages.py           # Rute untuk menyajikan template HTML
├── alembic/               # Konfigurasi dan file migrasi database
├── venv/                  # Python Virtual Environment
├── .env                   # Variabel lingkungan (Environment Variables)
├── alembic.ini            # Konfigurasi Alembic
├── requirements.txt       # Daftar dependensi library Python
├── reset_db.py            # Script utilitas untuk mereset dan menginisialisasi database
├── svm_face_classifier.pkl # Model SVM yang sudah dilatih (disimpan secara lokal)
└── label_encoder.pkl      # Encoder label untuk model SVM
```

---

## Skema Database

Sistem ini memiliki 4 tabel utama:

1. **`users` (User)**
   Menyimpan data pengguna administrator sistem (untuk keperluan autentikasi/login).
   - `id`: Primary Key
   - `username`: Nama pengguna (Unik)
   - `hashed_password`: Kata sandi yang telah di-hash.

2. **`cctv` (CCTV)**
   Menyimpan konfigurasi kamera pengawas.
   - `id`: Primary Key
   - `name`: Nama lokasi/kamera
   - `location`: Lokasi spesifik
   - `rtsp_url`: Alamat URL RTSP (atau index device kamera lokal seperti `0`)
   - `status`: Status keaktifan kamera
   - `created_at`: Waktu pembuatan

3. **`lectures` (Sivitas/Lecture)**
   Menyimpan data individu yang terdaftar di sistem presensi.
   - `id`: Primary Key
   - `nis`: Nomor Identitas (Unik, berindeks)
   - `name`: Nama lengkap
   - `gender`: Jenis kelamin
   - `images`: Nama file foto pendaftaran
   - `created_at`: Waktu registrasi

4. **`detection_logs` (Detection Log)**
   Menyimpan riwayat kehadiran yang terdeteksi dari CCTV.
   - `id`: Primary Key
   - `cctv_id`: Foreign Key ke tabel CCTV
   - `lecture_id`: Foreign Key ke tabel Lectures
   - `first_seen`: Waktu pertama kali wajah terdeteksi dalam rentang waktu tertentu
   - `last_seen`: Waktu terakhir wajah terlihat sebelum hilang dari frame
   - `confidence`: Tingkat kepercayaan AI saat mendeteksi
   - `status`: Status kehadiran (contoh: "hadir")

## Setup & Instalasi

### 1. Persyaratan (Prerequisites)
Pastikan Anda telah menginstal:
- Python 3.9 atau lebih tinggi
- (Opsional) C++ Build Tools untuk mengkompilasi InsightFace.

### 2. Instalasi Dependensi
Jalankan perintah berikut untuk menginstal semua library yang dibutuhkan di dalam virtual environment Anda:
```bash
pip install -r requirements.txt
```

### 3. Konfigurasi Variabel Lingkungan
Konfigurasi JWT dan secret terdapat pada `app/core/security.py` dan juga file `.env` di root direktori jika Anda menggunakannya.

### 4. Inisialisasi Database
Jika ini adalah pertama kali Anda menjalankan aplikasi, reset/inisialisasi database terlebih dahulu:
```bash
python reset_db.py
```
*Perintah ini akan membuat database SQLite, menjalankan migrasi Alembic, dan membuat akun default (username: `admin`, password: `admin123`).*

### 5. Menjalankan Aplikasi
Jalankan server FastAPI menggunakan `uvicorn`:
```bash
uvicorn app.main:app --reload
```
Aplikasi dapat diakses di `http://localhost:8000`. Dasbor utama otomatis merender template UI.
