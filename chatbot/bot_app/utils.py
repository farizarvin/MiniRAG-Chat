import pickle
import os
from django.conf import settings

# --- FUNGSI LOAD MODEL (Agar path dinamis) ---
def load_pickle(folder_name, file_name):
    # Mencari path absolut ke folder 'model' di dalam 'bot_app'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'model', folder_name, file_name)
    
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: File {file_name} tidak ditemukan di {file_path}")
        return None

# --- 1. LOAD MODEL INTENT ---
print("Sedang memuat Model Intent...")
intent_vectorizer = load_pickle('model_intent_classification', 'vectorize.pkl')
intent_model = load_pickle('model_intent_classification', 'model_intent.pkl')

# --- 2. LOAD MODEL CLUSTERING ---
print("Sedang memuat Model Clustering...")
# Kita asumsikan cluster_label.pkl berisi Dictionary hasil pengelompokan {0: [docs], 1: [docs]}
cluster_data = load_pickle('model_clustering_dokumen', 'cluster_label.pkl')

# --- FUNGSI PREDIKSI (Dipanggil views.py) ---
def prediksi_niat_user(teks):
    if not intent_vectorizer or not intent_model:
        return "Error: Model Intent belum dimuat."
        
    # Proses teks user sama seperti saat training (Vectorize -> Predict)
    # Ubah teks ke format list karena vectorizer butuh iterable
    teks_vector = intent_vectorizer.transform([teks])
    prediksi = intent_model.predict(teks_vector)[0]
    return prediksi

def ambil_data_cluster():
    if not cluster_data:
        return {}
    return cluster_data