"""
Module untuk Feature Based Summarization (Soal UAS #3)
Meringkas dokumen menjadi 3 kalimat utama berdasarkan bobot kata kunci (TF-IDF)
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def ringkas_dokumen(dokumen_list, max_sentences=3, return_features=False):
    """
    Meringkas list dokumen menjadi N kalimat terpenting menggunakan Feature Based Summarization
    
    Args:
        dokumen_list: List of strings (dokumen-dokumen yang akan diringkas)
        max_sentences: Jumlah kalimat yang diinginkan (default: 3)
        return_features: Jika True, return tuple (ringkasan, top_features)
    
    Returns:
        String berisi ringkasan (3 kalimat teratas) atau tuple jika return_features=True
    """
    if not dokumen_list:
        result = "Tidak ada dokumen untuk diringkas."
        return (result, []) if return_features else result
    
    # Filter dokumen yang terlalu pendek atau kosong
    kalimat_list = [d.strip() for d in dokumen_list if len(d.strip()) > 15]
    
    if len(kalimat_list) == 0:
        result = "Dokumen terlalu pendek untuk diringkas."
        return (result, []) if return_features else result
    
    # Jika jumlah dokumen kurang dari atau sama dengan max_sentences, return semua
    if len(kalimat_list) <= max_sentences:
        hasil = ". ".join(kalimat_list)
        hasil = hasil + ("." if not hasil.endswith(('.', '!', '?')) else "")
        return (hasil, []) if return_features else hasil
    
    # 3. Hitung skor TF-IDF untuk setiap dokumen (FEATURE BASED)
    try:
        vectorizer = TfidfVectorizer(
            stop_words=None,        # Jangan filter stopwords untuk bahasa Indonesia
            max_features=100,       # Ambil 100 kata terpenting
            ngram_range=(1, 2),     # Unigram dan bigram
            token_pattern=r'(?u)\b\w+\b'  # Pattern untuk tokenisasi
        )
        
        tfidf_matrix = vectorizer.fit_transform(kalimat_list)
        feature_names = vectorizer.get_feature_names_out()
        
        # 4. Hitung skor untuk setiap kalimat (jumlah TF-IDF semua kata)
        sentence_scores = tfidf_matrix.sum(axis=1).A1  # Convert to 1D array
        
        # 5. Rank kalimat berdasarkan skor (tertinggi ke terendah)
        ranked_indices = np.argsort(sentence_scores)[::-1]
        
        # 6. Ambil top N kalimat (dalam urutan asli dokumen, bukan urutan rank)
        top_indices = sorted(ranked_indices[:max_sentences])
        
        # 7. Ekstraksi top features (kata kunci tertinggi di seluruh korpus)
        # Sum TF-IDF scores across all documents for each feature
        feature_scores = tfidf_matrix.sum(axis=0).A1
        top_feature_indices = np.argsort(feature_scores)[::-1][:10]
        top_features = [(feature_names[i], feature_scores[i]) for i in top_feature_indices]
        
        # 8. Susun ringkasan
        ringkasan_parts = [kalimat_list[i] for i in top_indices]
        ringkasan = ". ".join(ringkasan_parts)
        
        # Pastikan diakhiri dengan titik
        if not ringkasan.endswith(('.', '!', '?')):
            ringkasan += "."
        
        if return_features:
            return (ringkasan, top_features)
        return ringkasan
        
    except Exception as e:
        # Fallback jika TF-IDF error
        print(f"Warning: TF-IDF error - {e}")
        hasil = ". ".join(kalimat_list[:max_sentences])
        hasil = hasil + ("." if not hasil.endswith(('.', '!', '?')) else "")
        return (hasil, []) if return_features else hasil


def ringkas_teks_sederhana(teks, max_sentences=3):
    """
    Versi sederhana untuk meringkas teks tunggal
    (Untuk command "ringkas ...")
    """
    if not teks or len(teks.strip()) < 20:
        return "Teks terlalu pendek untuk diringkas."
    
    return ringkas_dokumen([teks], max_sentences)


# --- TESTING (Hanya jalan jika file dieksekusi langsung) ---
if __name__ == "__main__":
    # Test dengan contoh dokumen
    docs = [
        "Mahasiswa harus menghubungi dosen wali untuk validasi KRS. Jadwal konsultasi tersedia di sistem akademik.",
        "Pendaftaran mahasiswa baru gelombang 1 sudah dibuka. Syarat masuk jalur prestasi adalah rapor semester 1 sampai 5.",
        "Pembayaran UKT dapat dilakukan melalui Virtual Account. Batas akhir pembayaran adalah sebelum UTS.",
        "Perpustakaan buka hari Senin sampai Jumat pukul 08.00. Layanan kesehatan poliklinik gratis untuk mahasiswa."
    ]
    
    print("=== TEST SUMMARIZER ===")
    print("Dokumen asli:", len(docs), "dokumen")
    print("\nRingkasan (3 kalimat):")
    print(ringkas_dokumen(docs, 3))
