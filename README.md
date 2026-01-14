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

## 📥 Download Trained Models

> **Important:** Model files (.pkl) are excluded from Git due to large file sizes (total ~500MB).

### Download from Google Drive

📦 **[Download All Models (ZIP - ~500MB)](https://drive.google.com/file/d/1czJavDZDeW1E8LFJ7mpJWgSlYMZSFaC8/view?usp=sharing)**

### Required Model Files

After downloading, extract and place files in these locations:

#### 1. Sentiment Analysis Model (482MB)
```
data/model/sentimen_analisis.pkl
```
- IndoBERT fine-tuned model
- 93.9% accuracy on Indonesian sentiment
- Required for sentiment analysis features

#### 2. K-NN Intent Classification
```
data/model/model_intent_classification/
├── model_intent.pkl (24KB)
└── vectorize.pkl (12KB)
```

#### 3. K-Means Clustering
```
data/model/model_clustering_dokumen/
├── kmeans_model.pkl (8KB)
├── tfidf_vectorizer.pkl (16KB)
└── cluster_label.pkl (4KB)
```

#### 4. Chatbot Models (copy from data/model/)
```
app/chatbot/bot_app/model/
├── model_sentiment_analisis/sentimen_analisis.pkl (475MB)
├── model_intent_classification/
│   ├── model_intent.pkl
│   └── vectorize.pkl
└── model_clustering_dokumen/
    ├── kmeans_model.pkl
    ├── tfidf_vectorizer.pkl  
    └── cluster_label.pkl
```

### Quick Setup Script

```bash
# After downloading models.zip to Downloads/
cd /Users/arvin/joki/sistek
unzip ~/Downloads/models.zip -d .
# Models will be extracted to correct locations
```

### Alternative: Train Models Yourself

If you prefer to train from scratch:

```bash
# K-NN Intent Classification
python src/knn_classifier.py

# K-Means Clustering
python src/kmeans_cluster.py

# IndoBERT Sentiment (requires GPU, ~2 hours)
python notebooks/Sentiment_Analysis_Training.py
```

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
