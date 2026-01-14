import os
import pickle
import numpy as np

# --- 1. KONFIGURASI PATH MODEL ---
# Mengambil path root project secara dinamis
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'bot_app', 'model', 'model_clustering_dokumen')

# --- 2. INISIALISASI VARIABEL GLOBAL ---
vectorizer = None
kmeans = None
data_dokumen = {}

# --- 3. FUNGSI LOAD MODEL (Dipakai saat Start & Reload) ---
def load_resources():
    """Helper function untuk memuat file .pkl dari disk"""
    global vectorizer, kmeans, data_dokumen
    try:
        path_vec = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
        path_kmeans = os.path.join(MODEL_DIR, 'kmeans_model.pkl')
        path_label = os.path.join(MODEL_DIR, 'cluster_label.pkl')

        # Cek keberadaan file
        if not (os.path.exists(path_vec) and os.path.exists(path_kmeans) and os.path.exists(path_label)):
            print("[WARNING] File model belum lengkap. Silakan lakukan Training dulu.")
            return False

        print("Loading Model Clustering dari Disk...")
        vectorizer = pickle.load(open(path_vec, 'rb'))
        kmeans = pickle.load(open(path_kmeans, 'rb'))
        data_dokumen = pickle.load(open(path_label, 'rb'))
        print("✅ Model Clustering Siap Digunakan!")
        return True

    except Exception as e:
        print(f"❌ Error Load Model: {e}")
        return False

# Jalankan load pertama kali saat server nyala
load_resources()

# --- 4. FUNGSI UTAMA: PENCARIAN ---
def cari_dokumen_relevan(teks_user):
    """
    Menerima teks user -> Memprediksi Cluster -> Mengembalikan Dokumen terkait.
    Return: (ID Cluster, List Dokumen)
    """
    # Cek apakah model sudah dimuat
    if not vectorizer or not kmeans:
        return None, ["Model belum tersedia. Silakan klik 'Latih Ulang Bot' di sidebar."]

    try:
        # A. Vectorize: Ubah teks jadi angka
        vec = vectorizer.transform([teks_user])
        
        # B. Predict: Tentukan masuk cluster mana
        cluster_id = kmeans.predict(vec)[0]
        
        # C. Retrieve: Ambil dokumen dari gudang data
        hasil_dokumen = data_dokumen.get(cluster_id, [])
        
        # Jika cluster kosong (jarang terjadi, tapi jaga-jaga)
        if not hasil_dokumen:
            return cluster_id, ["Belum ada dokumen di kategori ini."]
            
        return cluster_id, hasil_dokumen
    
    except Exception as e:
        return None, [f"Terjadi kesalahan prediksi: {str(e)}"]

# --- 5. FUNGSI KHUSUS: RELOAD (Untuk Tombol Training) ---
def reload_model_otomatis():
    """
    Fungsi ini dipanggil oleh views.py setelah proses training selesai.
    Tujuannya memperbarui variabel di RAM dengan file .pkl yang baru.
    """
    print("\n--- MEMUAT ULANG MODEL (RELOAD) ---")
    sukses = load_resources() # Panggil fungsi load yang sama
    if sukses:
        print("--- RELOAD SELESAI ---\n")
        return True
    else:
        print("--- RELOAD GAGAL ---\n")
        return False

# --- 6. BLOCK TESTING MANUAL (Hanya jalan jika file dieksekusi langsung) ---
if __name__ == "__main__":
    print("\n=== TEST MODUL CLUSTERING SEARCH ===")
    if vectorizer is None:
        print("Model belum ada. Pastikan sudah training.")
    else:
        while True:
            txt = input("\nMasukkan kata kunci (ketik 'exit' keluar): ")
            if txt.lower() == 'exit': break
            
            cid, docs = cari_dokumen_relevan(txt)
            print(f"Prediksi: Cluster {cid}")
            print("Dokumen Relevan:")
            for d in docs[:2]:
                print(f"- {d}")