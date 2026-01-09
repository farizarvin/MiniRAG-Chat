# app/chatbot.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vsm_ir import load_processed, build_term_stats, tfidf_matrix, search
import re, heapq

def extract_sentences(text):
    # Ganti newline jadi titik jika di antara huruf/angka
    text = re.sub(r'(?<=[a-z0-9])\n(?=[A-Z0-9])', '. ', text)
    # Pecah berdasarkan tanda baca dan newline
    sents = re.split(r'(?<=[.!?])\s+|\n+', text)
    # Bersihkan dan buang kalimat pendek
    sents = [s.strip() for s in sents if len(s.strip()) > 20]
    return sents

def score_sentence_by_query(sent, q_tokens):
    toks = sent.lower().split()
    return sum(1 for t in toks if t in q_tokens)

def rag_answer(query, k=3, scheme="normal"):
    docs, names = load_processed()
    vocab, t2i, TF, DF, lens, avg_len = build_term_stats(docs)
    TFIDF_normal, idf = tfidf_matrix(TF, DF, len(docs), sublinear=False)
    TFIDF_sub, _ = tfidf_matrix(TF, DF, len(docs), sublinear=True)

    if scheme == "bm25":
        results = search(query, TF=TF, DF=DF, lens=lens, avg_len=avg_len, t2i=t2i, names=names, docs=docs, top_k=k, scheme="bm25")
    else:
        TFIDF = TFIDF_sub if scheme=="sublinear" else TFIDF_normal
        results = search(query, TFIDF=TFIDF, t2i=t2i, idf=idf, names=names, docs=docs, top_k=k, scheme=scheme)

    # build template jawaban
    q_tokens = set(query.replace("AND","").replace("OR","").replace("NOT","").lower().split())
    lines = []
    lines.append("Berdasarkan dokumen teratas berikut:")
    for doc, score, idx in results:
        lines.append(f"- {doc} (score={score:.3f})")
    lines.append("\nRingkasan (kalimat relevan):")
    # pilih 3 sentences
    picked = 0
    for doc, score, idx in results:
        sents = extract_sentences(docs[idx])
        # score sentences
        scored = [(score_sentence_by_query(s, q_tokens), s) for s in sents]
        scored = sorted(scored, key=lambda x: x[0], reverse=True)
        for sc, s in scored[:2]:  # up to 2 sents each doc
            if sc>0:
                lines.append(f"- {s}")
                picked += 1
            if picked >= 5:
                break
        if picked >= 5:
            break
    if picked == 0:
        lines.append("Maaf, tidak ditemukan kalimat yang relevan pada konteks teratas.")
    lines.append("\nSumber: " + ", ".join([r[0] for r in results]))
    return "\n".join(lines)

# quick CLI
if __name__ == "__main__":
    q = input("Tanya: ")
    print(rag_answer(q, k=3, scheme="normal"))
