# src/vsm_ir.py
import os, math, numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix, diags
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"

def load_processed():
    files = sorted([p for p in PROC_DIR.iterdir() if p.suffix == ".txt"])
    docs = [p.read_text(encoding="utf-8", errors="ignore") for p in files]
    names = [p.name for p in files]
    return docs, names

def build_term_stats(docs):
    vocab = sorted({tok for doc in docs for tok in doc.split()})
    t2i = {t:i for i,t in enumerate(vocab)}
    N = len(docs)
    rows, cols, data = [], [], []
    DF = np.zeros(len(vocab), dtype=int)
    lens = np.zeros(N, dtype=int)

    for j, doc in enumerate(docs):
        tf = {}
        toks = doc.split()
        lens[j] = len(toks)
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        for t,c in tf.items():
            i = t2i[t]
            rows.append(j); cols.append(i); data.append(c)
            DF[i] += 1

    TF = csr_matrix((data,(rows,cols)), shape=(len(docs), len(vocab)), dtype=float)
    avg_len = float(lens.mean()) if len(lens)>0 else 0.0
    return vocab, t2i, TF, DF, lens, avg_len

def compute_idf(DF, N, smooth=True):
    if smooth:
        return np.log(1 + (N / (1 + DF)))
    else:
        return np.log(N / (DF + 1e-9))

def tfidf_matrix(TF, DF, N, sublinear=False, smooth=True):
    TF = TF.copy()
    if sublinear:
        TF.data = np.log1p(TF.data)
    idf = compute_idf(DF, N, smooth=smooth)
    D = diags(idf, 0)
    TFIDF = TF.dot(D)
    TFIDF = normalize(TFIDF, norm='l2', axis=1)
    return TFIDF, idf

def bm25_scores(query, t2i, TF, DF, lens, avg_len, k1=1.5, b=0.75):
    N = TF.shape[0]
    idf = np.log(1 + (N - DF + 0.5) / (DF + 0.5))
    q_freq = {}
    for t in query:
        if t in t2i:
            q_freq[t] = q_freq.get(t, 0) + 1
    scores = np.zeros(N, dtype=float)
    for t, qf in q_freq.items():
        i = t2i.get(t)
        if i is None: continue
        col = TF[:, i].toarray().reshape(-1)
        denom = col + k1 * (1 - b + b * (lens / avg_len))
        contrib = (col * (k1 + 1)) / (denom + 1e-9)
        scores += idf[i] * contrib
    return scores

def query_to_vector(query, t2i, idf, sublinear=False):
    qvec = np.zeros(len(t2i), dtype=float)
    for t in query.split():
        if t in t2i:
            qvec[t2i[t]] += 1.0
    if sublinear:
        qvec = np.log1p(qvec)
    qvec = qvec * idf
    norm = np.linalg.norm(qvec)
    if norm > 0:
        qvec = qvec / norm
    return qvec

def search(query, TFIDF=None, t2i=None, idf=None, names=None, docs=None, TF=None, DF=None, lens=None, avg_len=None, top_k=3, scheme="normal"):
    q_raw = query.replace("AND","").replace("OR","").replace("NOT","").replace("(","").replace(")","")
    q_clean = " ".join(q_raw.split())
    q_tokens = q_clean.split()

    if scheme == "bm25":
        sims = bm25_scores(q_tokens, t2i, TF, DF, lens, avg_len)
        idxs = np.argsort(sims)[::-1][:top_k]
        return [(names[i], float(sims[i]), i) for i in idxs]
    else:
        qv = query_to_vector(q_clean, t2i, idf, sublinear=(scheme=="sublinear"))
        sims = TFIDF.dot(qv)
        sims = np.array(sims).reshape(-1)
        idxs = np.argsort(sims)[::-1][:top_k]
        return [(names[i], float(sims[i]), i) for i in idxs]

# Evaluasi
def precision_at_k(relevant, retrieved, k):
    retrieved_k = retrieved[:k]
    hit = sum(1 for doc in retrieved_k if doc in relevant)
    return hit / k if k > 0 else 0.0

def recall_at_k(relevant, retrieved, k):
    retrieved_k = retrieved[:k]
    hit = sum(1 for doc in retrieved_k if doc in relevant)
    return hit / len(relevant) if len(relevant) > 0 else 0.0

def average_precision(relevant, retrieved, k):
    retrieved_k = retrieved[:k]
    hits, sum_prec = 0, 0.0
    for i, doc in enumerate(retrieved_k, 1):
        if doc in relevant:
            hits += 1
            sum_prec += hits / i
    return sum_prec / len(relevant) if len(relevant) > 0 else 0.0

def evaluate_vsm_results(TFIDF, t2i, idf, docs, names, goldsets, k=3):
    print("\n=== Evaluasi VSM terhadap Gold Set ===")
    results = []
    for query, gold_docs in goldsets.items():
        retrieved = [name for name, _, _ in search(query, TFIDF, t2i, idf, names, top_k=k)]
        p = precision_at_k(gold_docs, retrieved, k)
        r = recall_at_k(gold_docs, retrieved, k)
        ap = average_precision(gold_docs, retrieved, k)
        results.append((query, p, r, ap))
        print(f"\nQuery: {query}")
        print(f"Gold: {gold_docs}")
        print(f"Retrieved: {retrieved}")
        print(f"Precision@{k}: {p:.3f}, Recall@{k}: {r:.3f}, MAP@{k}: {ap:.3f}")
    mean_p = np.mean([r[1] for r in results])
    mean_r = np.mean([r[2] for r in results])
    mean_map = np.mean([r[3] for r in results])
    print("\n=== Rata-rata ===")
    print(f"Mean Precision@{k}: {mean_p:.3f}")
    print(f"Mean Recall@{k}: {mean_r:.3f}")
    print(f"Mean MAP@{k}: {mean_map:.3f}")
    return {"precision": mean_p, "recall": mean_r, "map": mean_map}


# Test
if __name__ == "__main__":
    docs, names = load_processed()
    vocab, t2i, TF, DF, lens, avg_len = build_term_stats(docs)
    TFIDF_normal, idf = tfidf_matrix(TF, DF, len(docs), sublinear=False)
    TFIDF_sub, _ = tfidf_matrix(TF, DF, len(docs), sublinear=True)

    goldsets = {
        "python AND data": ["deep-learning-with-python.txt","struktur-data.txt","python-for-data-analysis.txt"],
        "AI OR machine": ["introduction-to-machine-learning.txt","hands-on-machine-learning.txt","artificial-intelligence-modern-approach.txt"],
        "NOT AI": ["struktur-data.txt"],
        "Current News": ["news-udi-multiple.txt"]
    }

    print("\n=== Hasil Pencarian: Normal TF-IDF ===")
    for q in goldsets.keys():
        res = search(q, TFIDF=TFIDF_normal, t2i=t2i, idf=idf, names=names, docs=docs, top_k=3, scheme="normal")
        print(q, res)

    print("\n=== Evaluasi Skema Normal ===")
    metrics_normal = evaluate_vsm_results(TFIDF_normal, t2i, idf, docs, names, goldsets, k=3)

    print("\n=== Evaluasi Skema Sublinear ===")
    metrics_sub = evaluate_vsm_results(TFIDF_sub, t2i, idf, docs, names, goldsets, k=3)

    print("\n=== Evaluasi Skema BM25 ===")
    results_bm25 = []
    for q in goldsets.keys():
        res = search(q, TF=TF, DF=DF, lens=lens, avg_len=avg_len, t2i=t2i, names=names, docs=docs, top_k=3, scheme="bm25")
        print(q, res)