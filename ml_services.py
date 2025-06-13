from flask import Flask, request, jsonify
import pandas as pd
# import joblib # Tidak lagi diperlukan untuk memuat model Keras
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
from nltk.corpus import stopwords
import nltk
import warnings # Untuk menangani warnings

# Tambahkan import TensorFlow dan Keras
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Menonaktifkan peringatan yang tidak krusial
warnings.filterwarnings("ignore")

# --- MEMUAT MODEL ML & DATASET JD SAAT APLIKASI FLASK DIMULAI ---
# Ini akan memastikan model dan data JD hanya dimuat sekali saat server dimulai,
# bukan setiap kali ada request.

# Memuat model SBERT
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model SBERT 'all-MiniLM-L6-v2' berhasil dimuat.")

# Memuat model klasifikasi (Diasumsikan ini adalah model Keras Sequential)
try:
    # Ganti joblib.load dengan tf.keras.models.load_model
    # Pastikan nama file model Anda adalah 'cv_jd_scorer_model.h5' atau '.keras'
    classifier_model = load_model('models/cv_jd_scorer_model.h5')
    print("Model Keras 'cv_jd_scorer_model.h5' berhasil dimuat.")
except FileNotFoundError:
    print("Error: 'cv_jd_scorer_model.h5' tidak ditemukan. Pastikan model sudah dilatih dan disimpan.")
    classifier_model = None # Set to None or handle error appropriately
except Exception as e:
    print(f"Error memuat model Keras: {e}")
    classifier_model = None

# Mengunduh resource NLTK jika belum ada (sesuai script asli)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
print("NLTK stopwords dan wordnet resource siap.")


# --- MEMUAT DATASET JD (df_jd_info.csv) ---
# Asumsi: Anda memiliki URL atau path lokal ke file df_jd_info.csv
# Berdasarkan source, URL yang digunakan adalah file_id2
file_id_jd = '1EvumvgZxM8e7FyxYckezE8oPua1fI8p3'
url_jd = f'https://drive.google.com/uc?id={file_id_jd}'

df_jd_ml_side = pd.DataFrame() # Inisialisasi kosong
jd_embeddings_ml_side = torch.tensor([]) # Inisialisasi tensor kosong

try:
    df_jd_ml_side = pd.read_csv(url_jd, sep=',')
    print("Dataset JD berhasil dimuat di sisi ML.")

    # Gabungkan kolom-kolom JD yang relevan seperti di script pelatihan awal
    # Kolom yang ada di df_jd_info.csv adalah 'Role', 'Experience', 'skills', 'Job Description'
    jd_cols_to_combine = ['Role', 'Experience', 'skills', 'Job Description']
    df_jd_ml_side['JD_combined'] = df_jd_ml_side[jd_cols_to_combine].apply(
        lambda x: ' . '.join(x.dropna().astype(str)), axis=1
    )
    # Tambahkan Job_ID jika belum ada (sesuai dengan kolom 'Job Id' di source Anda)
    # Asumsi: df_jd_info.csv memiliki kolom 'Job Id' atau Anda perlu membuat ID unik
    # Jika tidak ada 'Job Id' di df_jd_info.csv, gunakan df_jd_ml_side.index
    if 'Job Id' in df_jd_ml_side.columns:
        df_jd_ml_side['Job_ID'] = df_jd_ml_side['Job Id']
    else:
        df_jd_ml_side['Job_ID'] = df_jd_ml_side.index
    print("Kolom JD telah digabungkan dan Job_ID dibuat.")

    # Generate embeddings untuk semua JD saat startup
    jd_texts_for_embeddings = df_jd_ml_side['JD_combined'].tolist()
    # show_progress_bar=False karena ini di background server
    jd_embeddings_ml_side = sbert_model.encode(jd_texts_for_embeddings, convert_to_tensor=True, show_progress_bar=False)
    print("Embeddings semua JD berhasil dibuat di sisi ML.")

except Exception as e:
    print(f"Error memuat atau memproses dataset JD di sisi ML: {e}")
    # Pastikan variabel tetap didefinisikan meskipun ada error
    df_jd_ml_side = pd.DataFrame(columns=['Role', 'Experience', 'skills', 'Job Description', 'JD_combined', 'Job_ID'])
    jd_embeddings_ml_side = torch.tensor([])


# Fungsi pra-pemrosesan teks dari kode ML
stop_words_indonesian = set(stopwords.words('indonesian'))
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words_indonesian]
    return " ".join(filtered_words)

# Fungsi untuk membersihkan dan memisahkan skills
def clean_skills(text):
    if not isinstance(text, str) and not isinstance(text, list):
        return set()
    if isinstance(text, list): # Jika skills datang sebagai list
        text = ', '.join(text) # Gabungkan list menjadi string
    text = text.lower().replace('\n', ',')
    text = re.sub(r'\(.*?\)', '', text)
    skills_list = [skill.strip() for skill in text.split(',') if skill.strip()]
    return set(skills_list)


@app.route('/analyze_cv', methods=['POST'])
def analyze_cv():
    if classifier_model is None:
        return jsonify({"error": "Model ML tidak dimuat. Server mungkin gagal diinisialisasi."}), 500

    data = request.get_json() # Mendapatkan data JSON dari request

    if not data or 'data' not in data:
        return jsonify({"error": "Format JSON tidak valid atau kunci 'data' hilang"}), 400

    cv_data = data['data']

    # Convert cv_id to standard int if it's a NumPy type
    cv_id = int(cv_data.get('id')) if cv_data.get('id') is not None else None
    applied_job_role_from_fe = cv_data.get('appliedJob', '').strip() # Role job yang di-apply dari FE
    educations = cv_data.get('educations', '')
    technical_skills = cv_data.get('technicalSkills', [])
    professional_experiences = cv_data.get('profesionalExperiences', [])

    # --- Gabungkan Informasi CV menjadi satu teks untuk Embeddings ---
    cv_combined_text = ""
    if educations:
        cv_combined_text += f"Education: {educations}. "
    if technical_skills:
        cv_combined_text += f"Skills: {', '.join(technical_skills)}. "
    if professional_experiences:
        exp_texts = [f"{exp.get('role', '')} at {exp.get('company', '')}" for exp in professional_experiences]
        cv_combined_text += f"Experience: {'. '.join(exp_texts)}. "

    # --- AMBIL JOB DESCRIPTION DARI df_jd_ml_side BERDASARKAN appliedJob ---
    job_description_text_for_match = ""
    job_skills_for_match = ""
    job_id_for_match = None
    role_matched = "" # Untuk menyimpan role JD yang benar-benar cocok

    if df_jd_ml_side.empty:
        return jsonify({"error": "Dataset JD tidak dimuat di sisi ML. Tidak dapat melakukan pencocokan."}), 500

    # Strategi pencarian JD:
    # 1. Coba cari yang sama persis (case-insensitive)
    matching_jds = df_jd_ml_side[df_jd_ml_side['Role'].str.lower() == applied_job_role_from_fe.lower()]

    if matching_jds.empty:
        # 2. Jika tidak ada yang sama persis, coba cari yang mengandung substring
        matching_jds = df_jd_ml_side[df_jd_ml_side['Role'].str.contains(applied_job_role_from_fe, case=False, na=False)]

    if not matching_jds.empty:
        # Ambil JD pertama yang cocok
        selected_jd = matching_jds.iloc[0]
        job_description_text_for_match = selected_jd['JD_combined']
        job_skills_for_match = selected_jd['skills'] # Menggunakan kolom 'skills' asli dari JD
        # Convert job_id_for_match to standard int if it's a NumPy type
        job_id_for_match = int(selected_jd['Job_ID']) if selected_jd['Job_ID'] is not None else None
        role_matched = selected_jd['Role']
        print(f"JD ditemukan untuk '{applied_job_role_from_fe}': {role_matched}")
    else:
        # Jika tidak ditemukan JD, kirim error
        return jsonify({"error": f"Deskripsi Pekerjaan untuk '{applied_job_role_from_fe}' tidak ditemukan di database ML. Mohon periksa nama 'appliedJob'."}), 404

    # --- Lakukan Analisis ML ---

    # 1. Pra-pemrosesan teks CV dan JD
    cleaned_cv_text = preprocess_text(cv_combined_text)
    cleaned_jd_text = preprocess_text(job_description_text_for_match)

    # 2. Menghasilkan embeddings menggunakan SBERT
    cv_embedding = sbert_model.encode(cleaned_cv_text, convert_to_tensor=True)
    jd_embedding = sbert_model.encode(cleaned_jd_text, convert_to_tensor=True)

    # 3. Menghitung skor kecocokan (cosine similarity)
    similarity_score = util.cos_sim(cv_embedding, jd_embedding).item()

    # 4. Melakukan prediksi menggunakan model klasifikasi (Keras Sequential)
    feature_vector = (cv_embedding - jd_embedding).cpu().numpy().reshape(1, -1)
    
    # Keras Sequential model menggunakan 'predict' untuk mendapatkan output, yang biasanya probabilitas
    prediction_proba_raw = classifier_model.predict(feature_vector)[0]

    # Menentukan probabilitas dan label berdasarkan output model
    # Jika output adalah satu nilai (misal dari sigmoid untuk klasifikasi biner)
    if isinstance(prediction_proba_raw, (float, np.float32, np.float64)):
        prob_match = prediction_proba_raw
        prob_no_match = 1 - prob_match
        prediction_proba = {"no_match": prob_no_match, "match": prob_match}
        prediction_label = 1 if prob_match > 0.5 else 0 # Threshold 0.5 untuk menentukan label
    else: # Jika output adalah array (misal dari softmax untuk multi-kelas atau biner dengan dua output)
        prediction_proba = {"no_match": prediction_proba_raw[0], "match": prediction_proba_raw[1]} # Asumsi [prob_class0, prob_class1]
        prediction_label = np.argmax(prediction_proba_raw) # Label adalah indeks probabilitas tertinggi


    # 5. Analisis Keterampilan
    cv_skills_set = clean_skills(technical_skills)
    job_skills_set = clean_skills(job_skills_for_match) # Menggunakan skills dari JD yang diambil
    missing_skills = job_skills_set - cv_skills_set

    # 6. Rekomendasi pekerjaan lain
    job_recommendations = []
    # Pastikan embeddings JD sudah dimuat dan tidak kosong
    if jd_embeddings_ml_side is not None and len(jd_embeddings_ml_side) > 0:
        # Hitung skor kecocokan CV ini dengan semua JD yang ada di df_jd_ml_side
        all_jd_scores = util.cos_sim(cv_embedding, jd_embeddings_ml_side)[0]

        # Ambil top N rekomendasi (misalnya 10 untuk filtering)
        top_scores, top_indices = torch.topk(all_jd_scores, k=min(10, len(all_jd_scores)))

        temp_recommendations = []
        for score, jd_idx in zip(top_scores, top_indices):
            recommended_jd = df_jd_ml_side.iloc[jd_idx.item()]
            # Pastikan tidak merekomendasikan pekerjaan yang sama persis jika sudah di-apply
            recommended_jd_id = int(recommended_jd['Job_ID']) if recommended_jd['Job_ID'] is not None else None

            if recommended_jd_id != job_id_for_match and \
               recommended_jd['Role'].lower() != applied_job_role_from_fe.lower():
                temp_recommendations.append({
                    'role': recommended_jd['Role'],
                    'match_score': f"{score.item():.2%}",
                    'job_id': recommended_jd_id
                })
        # Ambil hanya top 5 setelah filtering
        job_recommendations = temp_recommendations[:5]


    # --- Mengirimkan Hasil Analisis ke BE ---
    results = {
        "cv_id": cv_id,
        "applied_job_role": applied_job_role_from_fe,
        "matched_job_details": { # Detail JD yang berhasil dicocokkan
            "role": role_matched,
            "job_id": job_id_for_match
        },
        "match_score_percentage": f"{similarity_score * 100:.2f}%",
        "classifier_prediction": "Match" if prediction_label == 1 else "No Match",
        "prediction_probabilities": prediction_proba, # Ini sudah dalam format dictionary
        "missing_skills": sorted(list(missing_skills)),
        "job_recommendations": job_recommendations
    }

    return jsonify(results), 200 # Mengirim hasil dalam bentuk JSON

if __name__ == '__main__':
    # Untuk deployment, ubah debug=False dan gunakan WSGI server seperti Gunicorn
    app.run(host='0.0.0.0', port=5000, debug=True)