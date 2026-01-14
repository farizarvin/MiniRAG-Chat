# Data Directory Structure

This directory contains all datasets used in the STKI project.

## 📂 Subdirectories

### 1. `ir_docs/`
Information Retrieval documents for clustering.

**Files:**
- `dataset.csv` - 166 documents for K-Means clustering
  - Columns: `isi_berita`, `label`
  - 7 clusters: ADMINISTRASI_KEUANGAN, AKADEMIK_KRS, INFORMASI_DOSEN, BEASISWA, PENDAFTARAN_PMB, FASILITAS_KAMPUS, TEKNIS_AKUN
- `stopwords_custom.json` - Custom stopwords for preprocessing

**Usage:** K-Means clustering (SOAL 03), document search

---

### 2. `sentiment/`
Sentiment analysis dataset.

**Files:**
- `dataset/train_preprocess_ori.tsv` - 11,000 training samples
- `dataset/valid_preprocess.tsv` - 1,260 validation samples
- Format: TSV with columns `text` and `sentiment`
- Classes: positive, neutral, negative

**Usage:** IndoBERT sentiment analysis (SOAL 05)

---

### 3. `intent/`
Intent classification dataset.

**Files:**
- `intents.json` - Intent patterns and responses
  - 5 categories: PENDAFTARAN, KEUANGAN, AKADEMIK, BEASISWA, FASILITAS
- `dataset.csv` - Intent classification dataset

**Usage:** K-NN intent classification (SOAL 02)

---

## 📊 Dataset Summary

| Dataset | Type | Size | Classes | Purpose |
|---------|------|------|---------|---------|
| IR Docs | CSV | 166 docs | 7 | Clustering + Search |
| Sentiment | TSV | 12,260 samples | 3 | Sentiment Analysis |
| Intent | JSON/CSV | Variable | 5 | Intent Classification |

---

## 🔄 Preprocessing

All datasets are preprocessed using:
- Lowercase conversion
- Tokenization
- Stopword removal
- Stemming (Sastrawi for Indonesian)

See `src/preprocess.py` for preprocessing implementation.
