import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Setup Path agar dinamis
# BASE_DIR saat ini adalah .../sistek/chatbot
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Navigate to project root: /Users/sistek/
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

# Updated path to new data structure
CSV_PATH = os.path.join(ROOT_DIR, 'data', 'ir_docs', 'dataset.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'bot_app', 'model', 'model_clustering_dokumen')

def latih_model_sekarang():
    """Fungsi utama untuk melatih ulang model K-Means"""
    try:
        if not os.path.exists(CSV_PATH):
            return False, "File dataset.csv tidak ditemukan."

        # 1. Baca CSV
        df = pd.read_csv(CSV_PATH)
        if 'dokumen' not in df.columns:
            return False, "Kolom 'dokumen' tidak ada di CSV."
            
        docs = df['dokumen'].astype(str).tolist()
        
        # 2. Vectorizing (TF-IDF)
        vec_cluster = TfidfVectorizer(stop_words='english') 
        X_cluster = vec_cluster.fit_transform(docs)
        
        # 3. K-Means (7 Cluster)
        kmeans = KMeans(n_clusters=7, random_state=42)
        kmeans.fit(X_cluster)
        
        # 4. Mapping Dokumen
        cluster_map = {}
        labels = kmeans.labels_
        for i, label in enumerate(labels):
            if label not in cluster_map: cluster_map[label] = []
            cluster_map[label].append(docs[i])
            
        # 5. Simpan Model (.pkl)
        os.makedirs(MODEL_DIR, exist_ok=True)
        pickle.dump(vec_cluster, open(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'), 'wb'))
        pickle.dump(kmeans, open(os.path.join(MODEL_DIR, 'kmeans_model.pkl'), 'wb'))
        pickle.dump(cluster_map, open(os.path.join(MODEL_DIR, 'cluster_label.pkl'), 'wb'))
        
        return True, f"Sukses! Melatih {len(docs)} dokumen ke dalam 7 Cluster."
        
    except Exception as e:
        return False, f"Error Training: {str(e)}"