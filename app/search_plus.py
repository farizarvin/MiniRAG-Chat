"""
Search Plus Demo
Integrated search with K-NN classification, K-Means clustering, summarization, and sentiment analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knn_classifier import KNNClassifier
from src.kmeans_cluster import KMeansCluster
from src.summarizer import TextSummarizer
from src.sentiment import SentimentAnalyzer
from src.preprocess import preprocess_text

def demo_search_plus():
    """Demonstrate integrated search with all features"""
    
    print("=" * 70)
    print("SEARCH PLUS - INTEGRATED DEMO")
    print("K-NN + K-Means + Summarization + Sentiment Analysis")
    print("=" * 70)
    
    # Sample corpus
    corpus = [
        "Pendaftaran mahasiswa baru dibuka mulai bulan Juni. Calon mahasiswa dapat mendaftar secara online melalui website resmi universitas.",
        "Pembayaran UKT dapat dilakukan melalui bank BNI, BRI, atau Mandiri. Batas waktu pembayaran adalah akhir bulan setiap semester.",
        "Jadwal kuliah semester genap dimulai pada bulan Februari. Mahasiswa wajib mengisi KRS sebelum kuliah dimulai.",
        "Beasiswa prestasi tersedia untuk mahasiswa dengan IPK minimal 3.5. Informasi lengkap dapat dilihat di website kemahasiswaan."
    ]
    
    queries = [
        "cara daftar kuliah",
        "bayar ukt gimana",
        "jadwal kuliah kapan"
    ]
    
    print("\n" + "=" * 70)
    print("STEP 1: INTENT CLASSIFICATION (K-NN)")
    print("=" * 70)
    
    # Train classifier
    train_intents = ["cara daftar", "info bayar", "jadwal kuliah", "info beasiswa"]
    train_labels = ["PENDAFTARAN", "KEUANGAN", "AKADEMIK", "BEASISWA"]
    
    classifier = KNNClassifier(k=1)
    classifier.fit(train_intents, train_labels)
    
    for query in queries:
        intent = classifier.predict([query])[0]
        print(f"Query: '{query}' → Intent: {intent}")
    
    print("\n" + "=" * 70)
    print("STEP 2: DOCUMENT CLUSTERING (K-Means)")
    print("=" * 70)
    
    clusterer = KMeansCluster(n_clusters=2)
    clusterer.fit(corpus)
    
    print(f"Clustered {len(corpus)} documents into 2 groups")
    query_test = queries[0]
    cluster_id = clusterer.predict([query_test])[0]
    print(f"\nQuery '{query_test}' maps to Cluster {cluster_id}")
    
    print("\n" + "=" * 70)
    print("STEP 3: TEXT SUMMARIZATION")
    print("=" * 70)
    
    long_text = corpus[0]
    summarizer = TextSummarizer()
    summary = summarizer.summarize(long_text, num_sentences=1)
    
    print(f"\nOriginal ({len(long_text)} chars):")
    print(f"   {long_text}")
    print(f"\nSummary ({len(summary)} chars):")
    print(f"   {summary}")
    
    print("\n" + "=" * 70)
    print("STEP 4: SENTIMENT ANALYSIS")
    print("=" * 70)
    
    test_sentiments = [
        "Pelayanan kampus sangat baik dan membantu!",
        "Websitenya sulit diakses, sangat mengecewakan.",
        "Informasi cukup jelas, tidak ada masalah."
    ]
    
    analyzer = SentimentAnalyzer()
    
    for text in test_sentiments:
        result = analyzer.analyze(text)
        print(f"\nText: {text}")
        print(f"   → Sentiment: {result['sentiment']} ({result['confidence']:.1%} confidence)")
    
    print("\n" + "=" * 70)
    print("✅ SEARCH PLUS DEMO COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    demo_search_plus()
