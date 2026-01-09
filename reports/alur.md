## 2. Garis Besar Arsitektur Search Engine Klasik

Arsitektur search engine klasik pada proyek ini mengikuti alur umum sistem temu balik informasi (Information Retrieval System). Berdasarkan struktur folder dan program yang diimplementasikan (`preprocess.py`, `boolean_ir.py`, `vsm_ir.py`, dan `search_engine.py`), pipeline sistem dapat dijelaskan sebagai berikut:

### Alur Proses:
1. **Input Query**
   - Pengguna memasukkan teks query melalui CLI (misalnya: `"python AND data"` atau `"machine learning"`).

2. **Preprocessing**
   - Dilakukan oleh modul `preprocess.py` yang melakukan pembersihan dokumen (tokenisasi, normalisasi, stopword removal, dsb).
   - Hasil preprocessing disimpan dalam folder `data/processed/` untuk digunakan oleh model retrieval.

3. **Retrieval**
   - Dua pendekatan utama disediakan:
     - **Boolean IR (`boolean_ir.py`)**: menggunakan inverted index dan operasi logika (AND, OR, NOT).
     - **Vector Space Model (`vsm_ir.py`)**: menggunakan representasi TF-IDF (dan BM25) untuk menghitung cosine similarity antara query dan dokumen.

4. **Ranking**
   - Pada model VSM, dokumen diberi skor kesamaan (`cosine similarity` atau `BM25`) dan diurutkan dari skor tertinggi ke terendah.
   - Top-k dokumen terbaik diambil (biasanya k=3 atau 5).

5. **Presentation**
   - `search_engine.py` menampilkan hasil dalam format:
     ```
     - nama_dokumen (score=0.xxx)
       top terms: [term1, term2, ...]
     ```
   - Hasil dapat dievaluasi dengan metrik **Precision@k**, **MAP@k**, dan **nDCG@k** berdasarkan *gold set*.

### Ilustrasi Arsitektur: Query → Preprocess → Retrieve (Boolean/VSM) → Rank (Cosine/BM25) → Present Result

---

## 3. Sketsa Arsitektur RAG (Retrieval-Augmented Generation)

RAG (Retrieval-Augmented Generation) pada proyek ini diimplementasikan secara sederhana melalui `app/chatbot.py`. Pendekatan ini memadukan proses *retrieval klasik* dengan *template-based text generation*.

### Arsitektur Umum:
1. **Retrieval Tahap Awal**
   - Sistem menggunakan model retrieval klasik (`boolean_ir` atau `vsm_ir`) untuk mengambil *top-k* dokumen yang paling relevan terhadap query pengguna.

2. **Context Injection**
   - Hasil retrieval (teks dari top-k dokumen) dikumpulkan dan disiapkan sebagai konteks tambahan untuk proses generasi.

3. **Template-based Generator**
   - Alih-alih menggunakan LLM penuh, generator di sini menggunakan pendekatan sederhana berbasis template.
   - Contoh:
     ```python
     response = f"Berdasarkan dokumen terkait ({', '.join(top_docs)}), topik '{query}' membahas ..."
     ```

4. **Output**
   - Sistem mengembalikan jawaban terstruktur dengan menyebutkan dokumen sumber dan ringkasan konteksnya.

### Skema RAG dalam Proyek Ini:
User Query
↓
Retrieval (Boolean/VSM) src/vsm_ir.py
↓
Ambil top-k dokumen relevan src/search_engine.py
↓
Template-based Generator RAG app/chatbot.py
↓
Teks jawaban dengan konteks dari dokumen
