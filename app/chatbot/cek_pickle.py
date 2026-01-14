import pickle
import os

# Sesuaikan path ini dengan lokasi file pickle kamu di foto tadi
path_file = 'bot_app/model/model_clustering_dokumen/cluster_label.pkl'

print(f"--- MEMERIKSA FILE: {path_file} ---")

if os.path.exists(path_file):
    try:
        with open(path_file, 'rb') as f:
            data = pickle.load(f)
        
        # 1. Cek Tipe Datanya
        tipe_data = type(data)
        print(f"\n[1] TIPE DATA: {tipe_data}")
        
        # 2. Cek Isinya (Preview)
        print("\n[2] CONTOH ISI DATA:")
        if isinstance(data, dict):
            # Jika Dictionary, tampilkan keys dan sampel value
            print(f"Keys (Label Klaster): {list(data.keys())}")
            print("Contoh Data Klaster Pertama:")
            first_key = list(data.keys())[0]
            print(data[first_key][:3]) # Tampilkan 3 dokumen pertama
        elif isinstance(data, list) or "numpy" in str(tipe_data):
            # Jika List/Array, tampilkan 10 item pertama
            print(data[:10])
            print(f"Total Item: {len(data)}")
        else:
            print(data)
            
    except Exception as e:
        print(f"Error membaca pickle: {e}")
else:
    print("File tidak ditemukan! Cek path folder 'model' kamu lagi.")