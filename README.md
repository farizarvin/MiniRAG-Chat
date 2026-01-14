# STKI UAS - Sistem Temu Kembali Informasi
## A11.2020.12708 - Muhammad Fariz Arvin Pratama

Implementasi lengkap STKI dengan K-NN, K-Means, Feature Selection, Summarization, dan Sentiment Analysis menggunakan IndoBERT.

## 📂 Struktur Folder Final

```
sistek/
├── data/                         # 📊 Datasets & Models
│   ├── ir_docs/                 # IR documents (166 docs)
│   ├── sentiment/               # Sentiment data (12K samples)
│   ├── intent/                  # Intent data
│   └── model/                   # 🎯 Trained models (.pkl)
│
├── src/                          # 💻 Source Code
│   ├── preprocess.py
│   ├── vectorize.py
│   ├── knn_classifier.py
│   ├── kmeans_cluster.py
│   ├── feature_selection.py
│   ├── summarizer.py
│   ├── sentiment.py
│   └── eval_metrics.py
│
├── app/                          # 🎮 Applications
│   ├── classify.py              # K-NN demo
│   ├── cluster.py               # K-Means demo
│   ├── search_plus.py           # Integrated demo
│   └── chatbot/                 # 🌐 Django web app
│       └── manage.py
│
├── notebooks/                    # 📓 Jupyter Notebooks
│   ├── UAS_STKI_A11.2020.12708.ipynb
│   ├── Intent_Classification_KNN.ipynb
│   └── Clustering_KMeans.ipynb
│
├── sentimen_analisis/            # 🧠 Training scripts
│   ├── train_sentiment_indobert.py
│   └── predict_sentiment.py
│
├── reports/                      # 📝 Documentation
│   ├── readme.md
│   └── alur.md
│
└── requirements.txt
```

## ✅ Semua Requirement TERPENUHI

### SOAL 02 - K-NN (20%) ✅
- Dataset: Intent classification
- Model: 93% accuracy
- Files: `src/knn_classifier.py`, `app/classify.py`

### SOAL 03 - K-Means (15%) ✅
- Dataset: 166 documents, 7 clusters
- Files: `src/kmeans_cluster.py`, `app/cluster.py`

### SOAL 04 - Feature + Summary (25%) ✅
- Chi-square, Mutual Info, TF-IDF
- Extractive summarization
- Files: `src/feature_selection.py`, `src/summarizer.py`

### SOAL 05 - Sentiment (25%) ✅✅
- Dataset: 12,260 samples (>> requirement)
- Model: IndoBERT 93.9% (>> baseline)
- File: `src/sentiment.py`

## 🚀 Quick Start

### Demo Scripts
```bash
python app/classify.py
python app/cluster.py
python app/search_plus.py
```

### Web App
```bash
cd app/chatbot
python manage.py runserver
```

### Jupyter
```bash
jupyter notebook notebooks/
```

## 📊 Performance

| Model | Accuracy | Dataset |
|-------|----------|---------|
| K-NN | 93% | 300+ samples |
| K-Means | - | 166 docs |
| IndoBERT | 93.9% | 12,260 samples |

**Status: 95% Complete** - Tinggal laporan PDF
