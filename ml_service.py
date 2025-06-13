from flask import Flask, request, jsonify
import pandas as pd
import joblib # Untuk memuat model RandomForestClassifier
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
from nltk.corpus import stopwords
import nltk
import warnings # Untuk menangani warnings

app = Flask(__name__)

# Menonaktifkan peringatan yang tidak krusial
warnings.filterwarnings("ignore")

# --- MEMUAT MODEL ML & DATASET JD SAAT APLIKASI FLASK DIMULAI ---
# Ini akan memastikan model dan data JD hanya dimuat sekali saat server dimulai,
# bukan setiap kali ada request.

# Memuat model SBERT
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model SBERT 'all-MiniLM-L6-v2' berhasil dimuat.")

# Memuat model klasifikasi (RandomForestClassifier)
try:
    classifier_model = joblib.load('models/cv_jd_scorer_model.joblib')
    print("Model klasifikasi 'cv_jd_scorer_model.joblib' berhasil dimuat.")
except FileNotFoundError:
    print("Error: 'cv_jd_scorer_model.joblib' tidak ditemukan. Pastikan model sudah dilatih dan disimpan.")
    classifier_model = None # Set to None or handle error appropriately

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
        return jsonify({"error": "ML models not loaded. Server might have failed to initialize."}), 500

    # Mengambil data JSON langsung dari request.get_json()
    cv_data = request.get_json()

    # Validasi input JSON
    if not cv_data:
        return jsonify({"error": "Invalid JSON format or empty request body."}), 400
    
    # Cek field-field yang diperlukan
    required_fields = ['id', 'appliedJob', 'parseText', 'skills', 'experiences', 'educations']
    if not all(field in cv_data for field in required_fields):
        return jsonify({"error": f"Missing required fields. Expected: {', '.join(required_fields)}"}), 400


    # Extract data from the received JSON structure
    cv_id = int(cv_data.get('id')) if cv_data.get('id') is not None else None
    applied_job_role_from_fe = cv_data.get('appliedJob', '').strip()
    # Gunakan 'parseText' sebagai sumber utama teks CV
    cv_parse_text = cv_data.get('parseText', '') 
    # Skills datang sebagai list string, tidak perlu 'technicalSkills' lagi
    technical_skills = cv_data.get('skills', []) 
    # Experiences juga sudah dalam bentuk list, sesuaikan akses field
    professional_experiences = cv_data.get('experiences', []) 
    # Educations juga sudah dalam bentuk list
    educations = cv_data.get('educations', [])

    # --- Gabungkan Informasi CV menjadi satu teks untuk Embeddings ---
    # Lebih baik gunakan parseText jika ada, karena sudah hasil parsing lengkap
    cv_combined_text = cv_parse_text

    # Tambahkan informasi tambahan jika parseText mungkin tidak mencakup semua detail
    if not cv_combined_text: # Fallback jika parseText kosong
        if educations:
            # Contoh: "degree": "Not specified", "institution": "Bachelor of Science in Physics"
            edu_texts = [f"{edu.get('degree', '')} at {edu.get('institution', '')}" for edu in educations]
            cv_combined_text += f"Education: {'. '.join(edu_texts)}. "
        if technical_skills:
            cv_combined_text += f"Skills: {', '.join(technical_skills)}. "
        if professional_experiences:
            exp_texts = [f"{exp.get('jobTitle', '')} at {exp.get('company', '')}" for exp in professional_experiences] # Menggunakan jobTitle bukan role
            cv_combined_text += f"Experience: {'. '.join(exp_texts)}. "
    
    if not cv_combined_text.strip():
        return jsonify({"error": "CV text (parseText, skills, experiences, educations) is empty. Cannot analyze."}), 400


    # --- AMBIL JOB DESCRIPTION DARI df_jd_ml_side BERDASARKAN appliedJob ---
    job_description_text_for_match = ""
    job_skills_for_match = ""
    job_id_for_match = None
    role_matched = "" # Untuk menyimpan role JD yang benar-benar cocok

    if df_jd_ml_side.empty:
        return jsonify({"error": "JD dataset not loaded on ML side. Cannot perform matching."}), 500

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
        return jsonify({"error": f"Job Description for '{applied_job_role_from_fe}' not found in ML's database. Please check the 'appliedJob' name."}), 404

    # --- Lakukan Analisis ML ---

    # 1. Pra-pemrosesan teks CV dan JD
    cleaned_cv_text = preprocess_text(cv_combined_text)
    cleaned_jd_text = preprocess_text(job_description_text_for_match)

    # 2. Menghasilkan embeddings menggunakan SBERT
    cv_embedding = sbert_model.encode(cleaned_cv_text, convert_to_tensor=True)
    jd_embedding = sbert_model.encode(cleaned_jd_text, convert_to_tensor=True)

    # 3. Menghitung skor kecocokan (cosine similarity)
    similarity_score = util.cos_sim(cv_embedding, jd_embedding).item()

    # 4. Melakukan prediksi menggunakan model klasifikasi (RandomForestClassifier)
    # Penting: Pastikan 'classifier_model' Anda memang output probabilitas binary (0 atau 1)
    # Jika model Anda adalah RandomForestClassifier, .predict_proba() adalah yang Anda cari
    # Jika itu Keras/TensorFlow, pastikan output terakhir adalah sigmoid untuk binary atau softmax untuk multi-class.
    # Kode Anda di bawah tampaknya mengasumsikan model Keras/TF yang memprediksi probabilitas.

    feature_vector = (cv_embedding - jd_embedding).cpu().numpy().reshape(1, -1)
    
    # Check if classifier_model has predict_proba (for scikit-learn models)
    if hasattr(classifier_model, 'predict_proba'):
        raw_probs = classifier_model.predict_proba(feature_vector)[0]
        prediction_label = int(np.argmax(raw_probs)) # 0 for no match, 1 for match
        prediction_proba_dict = {
            "no_match": raw_probs[0].item(),
            "match": raw_probs[1].item()
        }
    elif hasattr(classifier_model, 'predict'): # For Keras/TensorFlow models
        raw_probs = classifier_model.predict(feature_vector)[0]
        # Keras/TF model might output a single value (sigmoid) or multiple (softmax)
        if len(raw_probs) == 1: # Binary classification with sigmoid output
            prediction_label = int(raw_probs[0] >= 0.5)
            prediction_proba_dict = {
                "no_match": (1 - raw_probs[0]).item(),
                "match": raw_probs[0].item()
            }
        else: # Multi-class classification with softmax output
            prediction_label = int(np.argmax(raw_probs))
            prediction_proba_dict = {
                "no_match": raw_probs[0].item(), # assuming index 0 is no match
                "match": raw_probs[1].item() # assuming index 1 is match
            }
    else:
        return jsonify({"error": "Classifier model does not have 'predict_proba' or 'predict' method."}), 500

    # 5. Analisis Keterampilan
    cv_skills_set = clean_skills(technical_skills)
    job_skills_set = clean_skills(job_skills_for_match) # Menggunakan skills dari JD yang diambil
    missing_skills = sorted(list(job_skills_set - cv_skills_set))

    # 6. Rekomendasi pekerjaan lain
    job_recommendations = []
    # Pastikan embeddings JD sudah dimuat dan tidak kosong
    if jd_embeddings_ml_side is not None and len(jd_embeddings_ml_side) > 0:
        # Hitung skor kecocokan CV ini dengan semua JD yang ada di df_jd_ml_side
        all_jd_scores = util.cos_sim(cv_embedding, jd_embeddings_ml_side)[0]

        # Ambil top N rekomendasi (misalnya 10)
        # Exclude JD yang sedang dianalisis
        top_scores, top_indices = torch.topk(all_jd_scores, k=min(20, len(all_jd_scores))) # Ambil lebih banyak untuk filter

        temp_recommendations = []
        for score, jd_idx in zip(top_scores, top_indices):
            recommended_jd = df_jd_ml_side.iloc[jd_idx.item()]
            recommended_jd_id = int(recommended_jd['Job_ID']) if recommended_jd['Job_ID'] is not None else None
            recommended_role = recommended_jd['Role']

            # Pastikan tidak merekomendasikan pekerjaan yang sama persis jika sudah di-apply
            # dan juga tidak merekomendasikan duplikat dalam daftar rekomendasi
            if recommended_jd_id != job_id_for_match and \
               recommended_role.lower() != applied_job_role_from_fe.lower() and \
               not any(rec['role'].lower() == recommended_role.lower() for rec in temp_recommendations):
                temp_recommendations.append({
                    'id': recommended_jd_id,
                    'role': recommended_role,
                    'matchScore': float(f"{score.item():.4f}") # Kirim sebagai float
                })
        # Ambil hanya top 5 setelah filtering
        job_recommendations = temp_recommendations[:5]


    # --- Mengirimkan Hasil Analisis ke BE ---
    # Sesuaikan output agar sesuai dengan format yang diharapkan oleh cv.controller.ts
    # cv.controller.ts updateCvAnalysisResults(newCv.id, aiResult.matchScore, aiResult.jobRecommendation, aiResult.fixCv)
    results = {
        "matchScore": float(f"{similarity_score:.4f}"), # float, bukan string persentase
        "jobRecommendation": job_recommendations,
        "fixCv": { # Ini adalah respons yang dibutuhkan oleh fixCv, sesuaikan dengan kebutuhan frontend Anda
            "missingSkills": missing_skills,
            "classifierPrediction": "Match" if prediction_label == 1 else "No Match",
            "predictionProbabilities": prediction_proba_dict
        },
        # Anda bisa menambahkan data lain untuk debugging/informasi jika diperlukan,
        # tapi pastikan tidak mengganggu apa yang diharapkan oleh updateCvAnalysisResults
        # "cv_id": cv_id, # Ini akan diupdate berdasarkan newCv.id dari backend
        # "applied_job_role": applied_job_role_from_fe,
        # "matched_job_details": {
        #     "role": role_matched,
        #     "job_id": job_id_for_match
        # }
    }

    return jsonify(results), 200 # Mengirim hasil dalam bentuk JSON

# Endpoint untuk health check (opsional, tapi disarankan)
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "ML service is running"}), 200

if __name__ == '__main__':
    # Untuk deployment, ubah debug=False dan gunakan WSGI server seperti Gunicorn
    app.run(host='0.0.0.0', port=5000, debug=True) # Perhatikan port di sini, controller menggunakan 5001