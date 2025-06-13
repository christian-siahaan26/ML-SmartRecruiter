import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import os

# 1. Pastikan folder 'data' ada
os.makedirs('data', exist_ok=True)

# 2. URL file Job Description dari Google Drive yang Anda berikan
# URL ini mengarah ke file yang terlihat di screenshot terakhir Anda.
# Jika ada file lain untuk Job Description, ganti URL ini.
file_id_jd = '1EvumvgZxM8e7FyxYckezE8oPua1fI8p3'
url_jd = f'https://drive.google.com/uc?id={file_id_jd}'

print(f"Memuat dataset Job Description dari: {url_jd}")
try:
    df_jd = pd.read_csv(url_jd)
    print("Dataset Job Description berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat dataset Job Description: {e}")
    exit() # Hentikan eksekusi jika gagal memuat data

# 3. Buat kolom 'Job_ID'
# Jika tidak ada kolom ID unik di df_jd, kita akan menggunakan indeks baris sebagai ID.
# Berdasarkan screenshot Anda, kolom 'Job Id' (dengan spasi) juga tidak terlihat,
# jadi kita akan membuat 'Job_ID' dari indeks.
if 'Job_ID' not in df_jd.columns:
    df_jd['Job_ID'] = df_jd.index.values
    print("Kolom 'Job_ID' dibuat dari indeks DataFrame.")

# 4. Pilih kolom yang relevan untuk disimpan dalam df_jd_info.csv
# df_jd_info.csv akan digunakan oleh ml_service.py untuk mengambil detail pekerjaan.
# Pastikan 'Role' dan 'Job_ID' ada. Jika ada kolom 'Company' di data Anda, sertakan juga.
# Dalam screenshot Anda, 'Company' tidak terlihat, jadi saya tidak memasukkannya.
df_jd_info_to_save = df_jd[['Job_ID', 'Role']].copy()
# Anda bisa menambahkan kolom lain seperti 'Experience' jika ingin menampilkannya di front-end.
# df_jd_info_to_save = df_jd[['Job_ID', 'Role', 'Experience']].copy()


# 5. Gabungkan kolom teks untuk membuat embeddings Job Description
# Ini akan digunakan untuk membuat job_embeddings.pt
jd_cols_to_combine = ['Role', 'Job Description', 'skills', 'Experience']
# Pastikan semua kolom ini ada di df_jd asli Anda.
# Jika salah satu tidak ada, hapus dari daftar atau tangani secara sesuai.
# Contoh: if 'Job Description' in df_jd.columns: ...
df_jd['JD_combined'] = df_jd[jd_cols_to_combine].apply(
    lambda x: ' . '.join(x.dropna().astype(str)), axis=1
)
print("Kolom teks JD digabungkan.")

# 6. Muat model Sentence-BERT dan buat embeddings
print("Memuat model Sentence-BERT dan membuat embeddings JD...")
try:
    model_sbert = SentenceTransformer('all-MiniLM-L6-v2').to('cpu') # Pastikan model dimuat ke CPU
    jd_texts = df_jd['JD_combined'].tolist()
    jd_embeddings = model_sbert.encode(jd_texts, convert_to_tensor=True).to('cpu') # Pastikan embeddings di CPU
    print("Embeddings JD berhasil dibuat dan dipindahkan ke CPU.")
except Exception as e:
    print(f"Gagal membuat embeddings JD: {e}")
    exit()

# 7. Simpan embeddings dan informasi JD ke folder 'data'
try:
    torch.save(jd_embeddings, 'data/job_embeddings.pt')
    df_jd_info_to_save.to_csv('data/df_jd_info.csv', index=False)
    print("File 'job_embeddings.pt' dan 'df_jd_info.csv' berhasil disimpan di folder 'data'.")
except Exception as e:
    print(f"Gagal menyimpan file: {e}")

print("\nProses persiapan data selesai. Anda sekarang bisa menjalankan kembali 'ml_service.py'.")