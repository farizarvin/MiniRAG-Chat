# src/boolean_ir.py
import re, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"

# load data
def load_processed():
    files = sorted([p for p in PROC_DIR.iterdir() if p.suffix == ".txt" and "log" not in p.name])
    docs = [p.read_text(encoding="utf-8") for p in files]
    filenames = [p.name for p in files]
    return docs, filenames

# vocab dan index
def build_vocabulary(docs):
    vocab = sorted({tok for doc in docs for tok in doc.split()})
    t2i = {t: i for i, t in enumerate(vocab)}
    return vocab, t2i

def build_incidence_matrix(docs, t2i):
    rows, cols, data = [], [], []
    for j, doc in enumerate(docs):
        unique = set(doc.split())
        for term in unique:
            i = t2i[term]
            rows.append(i)
            cols.append(j)
            data.append(1)
    mat = csr_matrix((data, (rows, cols)), shape=(len(t2i), len(docs)), dtype=int)
    return mat

def build_inverted_index(docs):
    inv = defaultdict(set)
    for j, doc in enumerate(docs):
        for t in set(doc.split()):
            inv[t].add(j)
    return dict(inv)

# boolean query & evaluate
def tokenize_query(q):
    # Tangani tanda kurung dan operator
    q = q.replace("(", " ( ").replace(")", " ) ")
    toks = q.split()
    toks = [t.lower() for t in toks]
    return toks

def evaluate_boolean(tokens, inv, n_docs):
    """Evaluasi query dengan precedence: NOT > AND > OR"""
    ALL = set(range(n_docs))

    def eval_term(tok):
        return inv.get(tok, set())

    def apply_not(terms):
        new_terms = []
        skip = False
        for i, t in enumerate(terms):
            if skip:
                skip = False
                continue
            if t == "not" and i + 1 < len(terms):
                new_terms.append(ALL - eval_term(terms[i + 1]))
                skip = True
            elif t not in ("and", "or", "not"):
                new_terms.append(eval_term(t) if isinstance(t, str) else t)
            else:
                new_terms.append(t)
        return new_terms

    # precedence step 1: NOT
    tokens = apply_not(tokens)

    # precedence step 2: AND
    i = 0
    while i < len(tokens):
        if tokens[i] == "and":
            left = tokens[i - 1]
            right = tokens[i + 1]
            new_set = left & right
            tokens = tokens[:i - 1] + [new_set] + tokens[i + 2:]
            i -= 1
        else:
            i += 1

    # precedence step 3: OR
    result = set()
    for t in tokens:
        if isinstance(t, set):
            result |= t
    return result

def boolean_query(q, inv, n_docs):
    tokens = tokenize_query(q)
    # handle rekursif parentheses
    def eval_parentheses(tokens):
        while "(" in tokens:
            close = tokens.index(")")
            open_ = max(i for i in range(close) if tokens[i] == "(")
            sub = tokens[open_ + 1:close]
            subres = evaluate_boolean(sub, inv, n_docs)
            tokens = tokens[:open_] + [subres] + tokens[close + 1:]
        return evaluate_boolean(tokens, inv, n_docs)
    return eval_parentheses(tokens)

# function explain
def explain(q, inv):
    docs, files = load_processed()
    n = len(docs)
    result = boolean_query(q, inv, n)

    explanation = {
        "query": q,
        "n_docs": n,
        "returned": [files[i] for i in sorted(result)],
        "terms": {}
    }

    toks = re.findall(r"\w+", q.lower())
    for t in toks:
        if t not in ("and", "or", "not"):
            explanation["terms"][t] = [files[i] for i in sorted(inv.get(t, set()))]

    return result, explanation

# evaluasi mini truth set
def evaluate_boolean_goldset(inv, files, goldsets=None):
    """
    Evaluasi hasil Boolean Retrieval terhadap gold relevant docs (mini truth set).
    
    Parameter:
    - inv: inverted index (dict term -> set(doc_index))
    - files: list nama file dokumen hasil preprocessing
    - goldsets: dict {query: [list of relevant filenames]}
    
    Jika goldsets=None, maka otomatis mencoba membaca dari data/goldsets.json
    """
    print("\n=== Evaluasi Mini Truth Set ===")

    # --- Load goldsets (jika tidak diberikan) ---
    if goldsets is None:
        gold_path = ROOT / "data" / "goldsets.json"
        if gold_path.exists():
            with open(gold_path, "r", encoding="utf-8") as f:
                goldsets = json.load(f)
            print(f"[INFO] Loaded gold sets from {gold_path}")
        else:
            print(f"[WARN] File {gold_path} tidak ditemukan dan parameter goldsets=None.")
            return None

    def precision_recall(predicted, gold):
        predicted, gold = set(predicted), set(gold)
        tp = len(predicted & gold)
        precision = tp / len(predicted) if predicted else 0
        recall = tp / len(gold) if gold else 0
        return precision, recall

    total_p, total_r, results = 0, 0, []

    for q, gold_docs in goldsets.items():
        res, exp = explain(q, inv)
        predicted_docs = [files[i] for i in sorted(res)]
        p, r = precision_recall(predicted_docs, gold_docs)

        print(f"\nQuery: {q}")
        print(f"Predicted: {predicted_docs}")
        print(f"Gold: {gold_docs}")
        print(f"Precision = {p:.2f}, Recall = {r:.2f}")

        results.append({
            "Query": q,
            "Predicted": predicted_docs,
            "Gold": gold_docs,
            "Precision": round(p, 2),
            "Recall": round(r, 2)
        })
        total_p += p
        total_r += r

    if results:
        avg_p, avg_r = total_p / len(results), total_r / len(results)
        print("\nRata-rata Precision = {:.2f}, Recall = {:.2f}".format(avg_p, avg_r))
    else:
        print("\n[Tidak ada query untuk dievaluasi]")

    return results

# Test
if __name__ == "__main__":
    docs, files = load_processed()
    vocab, t2i = build_vocabulary(docs)
    inv = build_inverted_index(docs)

    print("\n=== Uji Query Manual ===")
    tests = [
        "python AND data",
        "AI OR machine",
        "NOT AI",
        "deep AND (learning OR network)"
    ]

    for q in tests:
        res, exp = explain(q, inv)

    goldsets = {
        "python AND data": ["deep-learning-with-python.txt","deep-learning-with-python.txt","struktur-data.txt", "python-for-data-analysis.txt"],
        "AI OR machine": ["introduction-to-machine-learning.txt","hands-on-machine-learning.txt","artificial-intelligence-modern-approach.txt"],
        "NOT AI": ["struktur-data.txt"]
    }

    evaluate_boolean_goldset(inv, files, goldsets)
