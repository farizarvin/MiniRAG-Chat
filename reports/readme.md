stki-uts-<nim>-<nama>/
├─ data/
│  ├─ docs/              # Dokumen mentah (.txt)
│  └─ processed/         # Hasil preprocessing (.txt)
│
├─ src/
│  ├─ preprocess.py      # Soal 02: preprocessing pipeline
│  ├─ boolean_ir.py      # Soal 03: Boolean Retrieval
│  ├─ vsm_ir.py          # Soal 04: Vector Space Model
│  ├─ eval.py            # Soal 04 & 05: evaluasi (precision, recall, MAP)
│  └─ search_engine.py   # Soal 05: orchestrator CLI
│
├─ app/
│  └─ chatbot.py         # Soal 05: RAG mini (template-based generator)
│
├─ notebooks/
│  └─ UTS_STKI_<nim>.ipynb  # Eksperimen & dokumentasi
│
├─ reports/
│  ├─ laporan.pdf
│  └─ readme.md          # (file ini)
│
└─ requirements.txt

# Instalasi Awal
pip install -r requirements.txt

# Preprocessing (Soal 02) - Membersihkan, tokenisasi, stopword removal, dan stemming korpus.
python -m src.preprocess

# Boolean Retrieval (Soal 03)
## Membangun inverted index dan menjalankan query Boolean (AND, OR, NOT).
python -m src.boolean_ir

# Vector Space Model (Soal 04)
## Membangun TF–IDF matriks, menghitung cosine similarity, dan ranking top-k hasil.
python -m src.vsm_ir

# Tes VSM normal, top-3
python -m src.search_engine --model vsm --query "apa itu machine learning" --k 3 --scheme normal

# Tes VSM sublinear
python -m src.search_engine --model vsm --query "apa itu machine learning" --k 3 --scheme sublinear

# Tes BM25
python -m src.search_engine --model vsm --query "apa itu machine learning" --k 3 --scheme bm25

# Tes Boolean
python -m src.search_engine --model boolean --query "python AND data" --k 5

# Eval memakai goldset file
python -m src.search_engine --model vsm --scheme bm25 --k 3 --eval data/goldsets.json --query "apa itu machine learning"

# Run chatbot (Soal 05)
python -m app.chatbot

# Run web UI
- cd app/ragsite
- python manage.py runserver