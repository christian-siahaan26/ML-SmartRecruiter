# Layanan ML untuk Smart Recruiter

Ini adalah microservice Python (menggunakan Flask) yang berfungsi sebagai otak machine learning untuk proyek Smart Recruiter.

Layanan ini menyediakan beberapa API endpoint untuk melakukan evaluasi CV, prediksi kecocokan, dan memberikan rekomendasi pekerjaan.

## Cara Menjalankan

Pastikan Anda sudah menginstal Python 3.8+ di komputer Anda.

1.  **Buka Terminal**
    Buka terminal atau command prompt dan arahkan ke dalam folder `ml-service-python` ini.
    ```bash
    cd path/to/ml-service-python
    ```

2.  **Buat Virtual Environment (Sangat Direkomendasikan)**
    ```bash
    # Membuat environment
    python -m venv venv

    # Mengaktifkan environment
    # Windows
    venv\Scripts\activate
    # macOS / Linux
    source venv/bin/activate
    ```

3.  **Install Semua Library yang Dibutuhkan**
    Jalankan perintah berikut untuk menginstal semua library dari file `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan Server ML**
    ```bash
    python ml_service.py
    ```

    Jika berhasil, server akan berjalan di `http://127.0.0.1:5001`. Biarkan terminal ini tetap berjalan.

## Endpoint yang Tersedia

### 1. Rekomendasi Pekerjaan
- **URL:** `/recommend_jobs`
- **Method:** `POST`
- **Body (JSON):**
  ```json
  {
    "cv_text": "Teks lengkap dari CV pengguna..."
  }