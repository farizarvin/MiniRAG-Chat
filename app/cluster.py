"""
Clustering Demo - K-Means Clustering
Demonstrates K-Means clustering using trained model
"""

import sys
import os
import pickle
import pandas as pd

def demo_clustering():
    """Demonstrate K-Means clustering using trained model"""
    
    print("=" * 70)
    print("K-MEANS CLUSTERING DEMO")
    print("=" * 70)
    
    # Load trained clustering model
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              'data', 'model', 'model_clustering_kmeans.pkl')
    
    if not os.path.exists(model_path):
        print("⚠️  Model not found. Please train the model first.")
        return
    
    print(f"\n[1/3] Loading trained K-Means model...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    kmeans = model_data['model']
    vectorizer = model_data['vectorizer']
    
    print(f"✅ Model loaded successfully!")
    print(f"   • Number of clusters: {kmeans.n_clusters}")
    print(f"   • Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Load dataset to show cluster distribution
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                'data', 'ir_docs', 'dataset.csv')
    
    if os.path.exists(dataset_path):
        print(f"\n[2/3] Analyzing document distribution...")
        df = pd.read_csv(dataset_path)
        
        # Transform and predict
        X = vectorizer.transform(df['isi_berita'])
        labels = kmeans.predict(X)
        
        print("-" * 70)
        for cluster_id in range(kmeans.n_clusters):
            count = sum(labels == cluster_id)
            print(f"Cluster {cluster_id}: {count} documents")
            
            # Show sample documents
            cluster_docs = df[labels == cluster_id].head(2)
            for idx, row in cluster_docs.iterrows():
                preview = row['isi_berita'][:60] + "..." if len(row['isi_berita']) > 60 else row['isi_berita']
                print(f"   • {preview}")
            print()
    
        # Test clustering with new query
        print("=" * 70)
        print("[3/3] Testing with new query...")
        test_query = "informasi pendaftaran mahasiswa baru"
        print(f"Query: '{test_query}'")
        
        query_vec = vectorizer.transform([test_query])
        query_cluster = kmeans.predict(query_vec)[0]
        
        print(f"✅ Query belongs to Cluster {query_cluster}")
        print(f"\nOther documents in this cluster:")
        cluster_docs = df[labels == query_cluster].head(3)
        for idx, row in cluster_docs.iterrows():
            preview = row['isi_berita'][:60] + "..." if len(row['isi_berita']) > 60 else row['isi_berita']
            print(f"   • {preview}")
    
    print("\n" + "=" * 70)
    print("✅ K-Means Clustering Demo Complete!")
    print("=" * 70)

if __name__ == "__main__":
    demo_clustering()
