# Sistem Temu Kembali Informasi (STKI)
## A11.2020.12708 - Muhammad Fariz Arvin Pratama

Proyek ini mengimplementasikan sistem temu kembali informasi dengan fitur K-NN classification, K-Means clustering, text summarization, dan sentiment analysis menggunakan IndoBERT.

## 📂 Struktur Folder

```
stki-uas-A11.2020.12708/
├── data/
│   ├── ir_docs/              # Dokumen untuk clustering (166 docs)
│   └── sentiment/            # Dataset sentiment (12,260 samples)  
│
├── src/                      # Source code utama
│   ├── preprocess.py         # Text preprocessing
│   ├── vectorize.py          # TF-IDF vectorization
│   ├── knn_classifier.py     # K-NN classification
│   ├── kmeans_cluster.py     # K-Means clustering
│   ├── feature_selection.py  # Chi-square, Mutual Info
│   ├── summarizer.py         # Extractive summarization
│   ├── sentiment.py          # Sentiment analysis (IndoBERT)
│   └── eval_metrics.py       # Evaluation metrics
│
├── app/                      # Demo applications
│   ├── classify.py           # K-NN classification demo
│   ├── cluster.py            # K-Means clustering demo
│   └── search_plus.py        # Integrated search demo
│
├── chatbot/                  # Full Django web implementation
│   ├── bot_app/
│   │   ├── views.py          # API endpoints
│   │   ├── templates/        # Web UI
│   │   └── model/            # Trained models (.pkl)
│   └── manage.py
│
├── model/                    # Trained model files
│   ├── model_intent_knn.pkl         # K-NN (93% accuracy)
│   ├── model_clustering_kmeans.pkl  # K-Means (k=7)
│   └── sentimen_analisis.pkl        # IndoBERT (93.9% accuracy)
│
├── sentimen_analisis/        # Sentiment training scripts
│   ├── train_sentiment_indobert.py
│   └── predict_sentiment.py
│
├── notebooks/
│   └── UAS_STKI_A11.2020.12708.ipynb  # Jupyter demo
│
├── reports/
│   ├── readme.md             # Dokumentasi lengkap
│   └── alur.md               # Flow diagram
│
└── requirements.txt
```

## 🎯 Fitur yang Diimplementasikan

### 1. K-NN Classification (SOAL 02)
- File: `src/knn_classifier.py`, `app/classify.py`
- Intent classification dengan K-NN (k=3)
- Accuracy: 93%
- Demo: `python app/classify.py`

### 2. K-Means Clustering (SOAL 03)  
- File: `src/kmeans_cluster.py`, `app/cluster.py`
- Document clustering untuk search
- Clusters: 7 topik akademik
- Demo: `python app/cluster.py`

### 3. Summarization + Feature Selection (SOAL 04)
- File: `src/summarizer.py`, `src/feature_selection.py`
- Extractive summarization dengan sentence scoring
- Feature selection: Chi-square, Mutual Information
- TF-IDF vectorization: `src/vectorize.py`

### 4. Sentiment Analysis (SOAL 05)
- File: `src/sentiment.py`
- Model: IndoBERT (state-of-the-art)
- Dataset: 12,260 samples (3 classes)
- Accuracy: 93.9%
- Melebihi baseline (lexicon + ML)

### 5. Integrated Search Plus
- File: `app/search_plus.py`
- Kombinasi semua fitur dalam satu demo
- Demo: `python app/search_plus.py`

## 🚀 Cara Menjalankan

### Option 1: Demo Scripts (Standalone)
```bash
# K-NN Classification Demo
python app/classify.py

# K-Means Clustering Demo
python app/cluster.py

# Integrated Search Demo
python app/search_plus.py
```

### Option 2: Web Application (Full Features)
```bash
cd chatbot
python manage.py runserver
```
Buka: `http://127.0.0.1:8000`

**Fitur Web:**
- ✅ Chatbot dengan K-NN intent classification
- ✅ Document search dengan K-Means clustering
- ✅ Auto-summarization (toggle ON/OFF)
- ✅ Auto-sentiment analysis (IndoBERT, sidebar widget)
- ✅ Web scraping berita
- ✅ Polite response templates

### Option 3: Jupyter Notebook
```bash
jupyter notebook notebooks/UAS_STKI_A11.2020.12708.ipynb
```

## 📊 Evaluasi Model

| Model | Accuracy | Dataset | Method |
|-------|----------|---------|--------|
| K-NN Intent | 93% | 300+ samples | K-NN (k=3) |
| K-Means Cluster | N/A | 166 documents | K-Means (k=7) |
| IndoBERT Sentiment | 93.9% | 12,260 samples | Fine-tuned BERT |

**Evaluasi tersedia di**: `src/eval_metrics.py`
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix
- Classification Report (macro/weighted)

## 🛠️ Dependencies

```
Django==5.2.10
scikit-learn==1.7.2
transformers==4.57.5
torch==2.9.1
pandas
numpy
```

Install: `pip install -r requirements.txt`

## 📝 Struktur Data

### Dataset IR (Clustering)
- Location: `data/ir_docs/dataset.csv`
- Format: CSV dengan kolom `isi_berita` dan `label`
- Size: 166 documents, 7 clusters
- Labels: ADMINISTRASI_KEUANGAN, AKADEMIK_KRS, INFORMASI_DOSEN, dll.

### Dataset Sentiment
- Location: `data/sentiment/dataset/`
- Files: `train_preprocess_ori.tsv`, `valid_preprocess.tsv`
- Format: TSV dengan kolom `text` dan `sentiment`
- Size: 11,000 train + 1,260 validation
- Classes: positive, neutral, negative

## 🎓 Author

**Muhammad Fariz Arvin Pratama**  
NIM: A11.2020.12708  
Universitas Dian Nuswantoro  
Mata Kuliah: Sistem Temu Kembali Informasi (STKI)