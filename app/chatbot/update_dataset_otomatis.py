import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

# 1. KONFIGURASI
# BASE_DIR saat ini .../sistek/chatbot
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Naik ke .../sistek/dataset/...
ROOT_DIR = os.path.dirname(BASE_DIR)
FILE_CSV = os.path.join(ROOT_DIR, 'dataset', 'clustering_dokumen', 'dataset.csv')

def update_dataset_dari_web():
    print("=== MULAI UPDATE DATASET DARI WEB ===")
    
    # 2. AMBIL DATA DARI WEB (SCRAPING)
    url = "https://dinus.ac.id/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        print(f"[1] Mengakses {url}...")
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        data_baru = []
        
        # Strategi 1: Cari link artikel berita (tag <a> dengan href artikel)
        article_links = soup.find_all('a', href=lambda x: x and 'dinus.ac.id' in x and '/202' in x)
        for link in article_links:
            teks = link.get_text(strip=True)
            if len(teks) > 15:  # Hanya ambil teks yang cukup panjang (bukan "Selengkapnya")
                data_baru.append(teks)
        
        # Strategi 2: Jika tidak ada, cari semua heading h2, h3, h4
        if not data_baru:
            headings = soup.find_all(['h2', 'h3', 'h4'])
            for h in headings:
                teks = h.get_text(strip=True)
                if len(teks) > 15:
                    data_baru.append(teks)
        
        # Strategi 3: Ambil semua paragraf panjang
        if not data_baru:
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                teks = p.get_text(strip=True)
                if len(teks) > 50:  # Paragraf minimal 50 karakter
                    data_baru.append(teks)
        
        # Hapus duplikat
        data_baru = list(dict.fromkeys(data_baru))
        
        print(f"    -> Ditemukan {len(data_baru)} item terbaru.")
        
        
    except Exception as e:
        print(f"[ERROR] Gagal scraping: {e}")
        return

    # 3. BACA CSV LAMA
    try:
        df = pd.read_csv(FILE_CSV)
        data_lama = df['dokumen'].tolist()
    except:
        data_lama = []
        print("    -> File CSV belum ada, akan dibuat baru.")

    # 4. FILTER DUPLIKAT & SIMPAN
    jumlah_tambah = 0
    with open(FILE_CSV, 'a', encoding='utf-8') as f:
        # Jika file kosong/baru, tambahkan header dulu
        if not data_lama:
            f.write("dokumen\n")
            
        for teks in data_baru:
            # Cek apakah kalimat ini sudah ada di CSV?
            if teks not in data_lama:
                # Tambahkan kutipan agar aman format CSV-nya
                f.write(f'"{teks}"\n')
                jumlah_tambah += 1
                print(f"    [+] Menambahkan: {teks[:50]}...")
    
    if jumlah_tambah > 0:
        print(f"\n[SUKSES] Berhasil menambahkan {jumlah_tambah} data baru ke dataset.csv!")
        print("Saran: Sekarang jalankan 'python latih_dari_csv.py' untuk update otak bot.")
    else:
        print("\n[INFO] Tidak ada data baru. Semua berita web sudah ada di CSV.")

if __name__ == "__main__":
    update_dataset_dari_web()