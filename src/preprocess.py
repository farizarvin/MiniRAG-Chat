# src/preprocess.py

import re
from collections import Counter
from pathlib import Path
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib.pyplot as plt

# --- Direktori utama ---
ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "docs"
PROC_DIR = ROOT / "data" / "processed"
LOG_FILE = PROC_DIR / "preprocess_log.txt"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# --- Stopwords sederhana (bisa ditambah) ---
STOPWORDS = set("""
yang dan di ke dari untuk pada dengan adalah ini itu oleh sebagai juga 
akan tidak dapat telah atau lebih karena para mereka kita kamu saya anda 
sebuah suatu bila saja setiap serta namun masih maka
""".split())

stemmer = StemmerFactory().create_stemmer()

# ------------------------------------------------------------
# 🔧 Tahapan Preprocessing
# ------------------------------------------------------------
def clean(text: str) -> str:
    """Case folding, normalisasi angka & tanda baca"""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # hapus tanda baca
    text = re.sub(r"\d+", " ", text)          # hapus angka
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str):
    """Memecah teks menjadi token sederhana"""
    return text.split()

def remove_stopwords(tokens):
    """Menghapus stopwords umum"""
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def stem(tokens):
    """Melakukan stemming menggunakan Sastrawi"""
    return [stemmer.stem(t) for t in tokens]

def preprocess_text(text):
    """Pipeline preprocessing"""
    text = clean(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    return tokens

# ------------------------------------------------------------
# 🚀 Eksekusi Preprocessing ke Semua Dokumen
# ------------------------------------------------------------
def run_all(plot=True):
    stats = []
    log_lines = []

    files = sorted([p for p in DOCS_DIR.iterdir() if p.suffix == ".txt"])
    for p in files:
        raw = p.read_text(encoding="utf-8", errors="ignore")
        toks = preprocess_text(raw)
        out = PROC_DIR / p.name
        out.write_text(" ".join(toks), encoding="utf-8")

        stats.append((p.name, len(toks)))
        log_lines.append(f"{p.name}: {len(toks)} tokens")

        print(f"[preprocess] {p.name} -> {len(toks)} tokens -> {out}")

    # Simpan log proses
    LOG_FILE.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\n✅ Log tersimpan di: {LOG_FILE}")

    # --- Opsional: Plot distribusi panjang dokumen ---
    if plot and stats:
        plt.figure(figsize=(8, 4))
        names = [s[0] for s in stats]
        vals = [s[1] for s in stats]
        plt.bar(names, vals)
        plt.xticks(rotation=30, ha="right")
        plt.title("Distribusi Panjang Dokumen (Jumlah Token)")
        plt.tight_layout()
        plt.savefig(PROC_DIR / "doc_length_distribution.png")
        print(f"[plot] saved to {PROC_DIR / 'doc_length_distribution.png'}")

    # --- Tampilkan top token per dokumen ---
    for name, _ in stats:
        toks = (PROC_DIR / name).read_text(encoding="utf-8").split()
        freq = Counter(toks)
        print(f"\nTop 10 tokens untuk {name}:")
        for t, c in freq.most_common(10):
            print(f"  {t:15} {c}")

def main():
    """Wrapper agar bisa dipanggil dari notebook"""
    run_all()

if __name__ == "__main__":
    run_all()
