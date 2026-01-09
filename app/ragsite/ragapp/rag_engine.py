import os, sys, re, heapq
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.append(ROOT_DIR)

from src.vsm_ir import load_processed, build_term_stats, tfidf_matrix, search

# =========================================================
# Fungsi bantu
# =========================================================
def extract_sentences(text):
    """Memecah teks menjadi kalimat yang cukup panjang untuk diambil ringkasannya."""
    text = re.sub(r'(?<=[a-z0-9])\n(?=[A-Z0-9])', '. ', text)
    sents = re.split(r'(?<=[.!?])\s+|\n+', text)
    sents = [s.strip() for s in sents if len(s.strip()) > 20]
    return sents

def score_sentence_by_query(sent, q_tokens):
    toks = sent.lower().split()
    return sum(1 for t in toks if t in q_tokens)

# =========================================================
# Inti: RAG sederhana berbasis VSM/TF-IDF/BM25
# =========================================================
def rag_answer(query, k=3, scheme="normal"):
    """Menjawab query berbasis pencarian VSM/BM25 sederhana."""
    docs, names = load_processed()
    vocab, t2i, TF, DF, lens, avg_len = build_term_stats(docs)
    TFIDF_normal, idf = tfidf_matrix(TF, DF, len(docs), sublinear=False)
    TFIDF_sub, _ = tfidf_matrix(TF, DF, len(docs), sublinear=True)

    # Pilih skema retrieval
    if scheme == "bm25":
        results = search(query, TF=TF, DF=DF, lens=lens, avg_len=avg_len,
                         t2i=t2i, names=names, docs=docs, top_k=k, scheme="bm25")
    else:
        TFIDF = TFIDF_sub if scheme == "sublinear" else TFIDF_normal
        results = search(query, TFIDF=TFIDF, t2i=t2i, idf=idf,
                         names=names, docs=docs, top_k=k, scheme=scheme)

    # Template jawaban
    q_tokens = set(query.replace("AND","").replace("OR","").replace("NOT","").lower().split())
    summary_lines = []
    summary_lines.append("Berdasarkan dokumen teratas berikut:")
    for doc, score, idx in results:
        summary_lines.append(f"- {doc} (score={score:.3f})")

    summary_lines.append("\nRingkasan (kalimat relevan):")
    picked = 0
    for doc, score, idx in results:
        sents = extract_sentences(docs[idx])
        scored = [(score_sentence_by_query(s, q_tokens), s) for s in sents]
        scored = sorted(scored, key=lambda x: x[0], reverse=True)
        for sc, s in scored[:2]:
            if sc > 0:
                summary_lines.append(f"- {s}")
                picked += 1
            if picked >= 5:
                break
        if picked >= 5:
            break

    if picked == 0:
        summary_lines.append("Maaf, tidak ditemukan kalimat yang relevan pada konteks teratas.")

    sources = [r[0] for r in results]
    answer_text = "\n".join(summary_lines)
    citation_text = "\n".join([f"- {r[0]} (score={r[1]:.3f})" for r in results])
    return answer_text, citation_text

# =========================================================
# Kompatibilitas fungsi lama
# =========================================================
def handle_query(question: str, top_k: int = 3, scheme="normal"):
    """Fungsi utama yang dipanggil front-end."""
    answer, citations = rag_answer(question, k=top_k, scheme=scheme)
    return answer, citations

# =========================================================
# Wrapper untuk views.py
# =========================================================
def rag_pipeline(question: str, k: int = 3, scheme: str = "normal"):
    """
    Wrapper agar kompatibel dengan Django views.
    Mengembalikan (answer: str, citations: list[dict]) sesuai format front-end.
    """
    answer, citations_text = handle_query(question, top_k=k, scheme=scheme)

    # Ubah citations_text (string multiline) → list of dict untuk front-end
    citations = []
    for line in citations_text.splitlines():
        if line.strip().startswith("-"):
            try:
                parts = line.strip("- ").rsplit(" (score=", 1)
                title = parts[0].strip()
                score = float(parts[1].replace(")", "")) if len(parts) > 1 else 0.0
                citations.append({
                    "title": title,
                    "source": title,  # bisa disesuaikan kalau ingin beda
                    "score": score,
                })
            except Exception:
                continue

    return answer, citations


# =========================================================
# CLI cepat untuk pengujian manual
# =========================================================
if __name__ == "__main__":
    q = input("Tanya: ")
    ans, cites = handle_query(q, top_k=3)
    print("\n🧩 Jawaban:\n", ans)
    print("\n📚 Sumber:\n", cites)
