import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier  # Sesuai notebook.ipynb
from sklearn.cluster import KMeans                  # Sesuai notebook..ipynb

# --- 1. KONFIGURASI FOLDER ---
# Script akan mencari folder bot_app/model/.. relatif dari file ini berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_INTENT = os.path.join(BASE_DIR, 'bot_app', 'model', 'model_intent_classification')
PATH_CLUSTER = os.path.join(BASE_DIR, 'bot_app', 'model', 'model_clustering_dokumen')

# Buat folder jika belum ada
os.makedirs(PATH_INTENT, exist_ok=True)
os.makedirs(PATH_CLUSTER, exist_ok=True)

print("=== MULAI GENERATE MODEL ULANG (SESUAI NOTEBOOK) ===")

# ==========================================
# 2. INTENT CLASSIFICATION (Sesuai Soal 1)
# ==========================================
print("\n[1] Melatih Model Intent (KNN)...")

# Dataset Dummy yang mencakup 7 Kategori dari notebook.ipynb
data_intent = [
    # ADMINISTRASI_KEUANGAN
    "kapan batas akhir pembayaran kuliah", "berapa biaya kuliah per semester", 
    "bagaimana cara bayar ukt", "dimana tempat bayar registrasi", "info tagihan uang gedung",
    
    # AKADEMIK_KRS
    "kapan jadwal pengisian krs", "krs saya error tidak bisa input", 
    "bagaimana cara batal tambah krs", "jadwal krs semester ganjil", "sks saya masih kurang",
    
    # INFORMASI_DOSEN
    "siapa dosen wali saya", "minta nomor wa dosen", 
    "jadwal bimbingan dosen", "dosen pengampu mata kuliah ini siapa", "cara menghubungi dosen",
    
    # BEASISWA
    "ada info beasiswa terbaru", "syarat beasiswa djarum", 
    "kapan pendaftaran beasiswa kpi", "beasiswa untuk mahasiswa kurang mampu", "beasiswa prestasi akademik",
    
    # PENDAFTARAN_PMB
    "cara daftar mahasiswa baru", "biaya pendaftaran pmb berapa", 
    "kapan gelombang 2 dibuka", "syarat masuk jurusan teknik", "alur pendaftaran mahasiswa baru",
    
    # FASILITAS_KAMPUS
    "jam buka perpustakaan", "wifi kampus lemot", 
    "dimana letak poliklinik", "cara pinjam ruangan aula", "parkiran motor sebelah mana",
    
    # TEKNIS_AKUN
    "lupa password akun siadin", "akun saya terkunci", 
    "cara reset password email mahasiswa", "tidak bisa login e-learning", "ganti password wifi"
]

labels_intent = [
    "ADMINISTRASI_KEUANGAN", "ADMINISTRASI_KEUANGAN", "ADMINISTRASI_KEUANGAN", "ADMINISTRASI_KEUANGAN", "ADMINISTRASI_KEUANGAN",
    "AKADEMIK_KRS", "AKADEMIK_KRS", "AKADEMIK_KRS", "AKADEMIK_KRS", "AKADEMIK_KRS",
    "INFORMASI_DOSEN", "INFORMASI_DOSEN", "INFORMASI_DOSEN", "INFORMASI_DOSEN", "INFORMASI_DOSEN",
    "BEASISWA", "BEASISWA", "BEASISWA", "BEASISWA", "BEASISWA",
    "PENDAFTARAN_PMB", "PENDAFTARAN_PMB", "PENDAFTARAN_PMB", "PENDAFTARAN_PMB", "PENDAFTARAN_PMB",
    "FASILITAS_KAMPUS", "FASILITAS_KAMPUS", "FASILITAS_KAMPUS", "FASILITAS_KAMPUS", "FASILITAS_KAMPUS",
    "TEKNIS_AKUN", "TEKNIS_AKUN", "TEKNIS_AKUN", "TEKNIS_AKUN", "TEKNIS_AKUN"
]

# a. Vectorizer
vectorizer_intent = TfidfVectorizer()
X_intent = vectorizer_intent.fit_transform(data_intent)

# b. Model KNN (n_neighbors=1 sesuai notebook)
model_intent = KNeighborsClassifier(n_neighbors=1)
model_intent.fit(X_intent, labels_intent)

# c. Simpan
print("    -> Menyimpan vectorize.pkl...")
with open(os.path.join(PATH_INTENT, 'vectorize.pkl'), 'wb') as f:
    pickle.dump(vectorizer_intent, f)

print("    -> Menyimpan model_intent.pkl...")
with open(os.path.join(PATH_INTENT, 'model_intent.pkl'), 'wb') as f:
    pickle.dump(model_intent, f)


# ==========================================
# 3. CLUSTERING DOKUMEN (Sesuai Soal 2)
# ==========================================
print("\n[2] Melatih Model Clustering (K-Means 7 Cluster)...")

# Dataset Dummy Dokumen (Untuk disebar ke 7 cluster)
dokumen_raw = [
    # Cluster: INFORMASI_DOSEN
    "Mahasiswa harus menghubungi dosen wali untuk validasi KRS.",
    "Jadwal konsultasi dengan dosen pembimbing tersedia di sistem.",
    "Kontak dosen pengampu dapat dilihat di menu akademik.",
    
    # Cluster: PENDAFTARAN_PMB
    "Pendaftaran mahasiswa baru gelombang 1 sudah dibuka.",
    "Syarat masuk jalur prestasi adalah rapor semester 1 sampai 5.",
    "Biaya pendaftaran PMB dapat ditransfer ke bank mitra.",
    
    # Cluster: ADMINISTRASI_KEUANGAN
    "Pembayaran UKT dapat dilakukan melalui Virtual Account.",
    "Batas akhir pembayaran uang gedung adalah sebelum UTS.",
    "Panduan pembayaran registrasi semester genap.",
    
    # Cluster: FASILITAS_KAMPUS
    "Perpustakaan buka hari Senin sampai Jumat pukul 08.00.",
    "Fasilitas laboratorium komputer tersedia di gedung D.",
    "Layanan kesehatan poliklinik gratis untuk mahasiswa.",
    
    # Cluster: TEKNIS_AKUN
    "Jika lupa password akun akademik, silakan hubungi BTI.",
    "Cara reset password email institusi mahasiswa.",
    "Akun terkunci karena salah input password 3 kali.",
    
    # Cluster: AKADEMIK_KRS
    "Pengisian KRS online wajib dilakukan sesuai jadwal.",
    "Mahasiswa yang terlambat input KRS harus lapor ke BAAK.",
    "Jadwal perubahan KRS (Revisi) minggu depan.",
    
    # Cluster: BEASISWA
    "Beasiswa Djarum Plus membuka pendaftaran untuk semester 4.",
    "Syarat beasiswa prestasi adalah IPK minimal 3.50.",
    "Info pencairan dana beasiswa KIP Kuliah.",
    
    # Tambahan acak biar model robust
    "Dosen wali wajib validasi sebelum cetak kartu ujian.",
    "Beasiswa Bank Indonesia untuk mahasiswa aktif organisasi."
]

# a. Vectorizer
vectorizer_cluster = TfidfVectorizer()
X_cluster = vectorizer_cluster.fit_transform(dokumen_raw)

# b. KMeans (7 Cluster sesuai notebook..ipynb)
kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
kmeans.fit(X_cluster)

# c. Buat Dictionary Cluster Label (ID -> List Dokumen)
# Ini penting agar di Web nanti muncul teksnya, bukan cuma angka/label.
cluster_label = {}
labels = kmeans.labels_

for i, label in enumerate(labels):
    if label not in cluster_label:
        cluster_label[label] = []
    # Simpan teks asli dokumen ke dalam list
    cluster_label[label].append(dokumen_raw[i])

# d. Simpan
print("    -> Menyimpan tfidf_vectorizer.pkl...")
with open(os.path.join(PATH_CLUSTER, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer_cluster, f)

print("    -> Menyimpan kmeans_model.pkl...")
with open(os.path.join(PATH_CLUSTER, 'kmeans_model.pkl'), 'wb') as f:
    pickle.dump(kmeans, f)

print("    -> Menyimpan cluster_label.pkl...")
with open(os.path.join(PATH_CLUSTER, 'cluster_label.pkl'), 'wb') as f:
    pickle.dump(cluster_label, f)

print("\n=== SUKSES! MODEL TELAH DIPERBARUI ===")
print("Sekarang jalankan: python manage.py runserver")