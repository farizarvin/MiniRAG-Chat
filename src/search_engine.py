# src/search_engine.py
import argparse, json
from pathlib import Path
from src.boolean_ir import load_processed as load_boolean_processed, build_inverted_index, explain
from src.vsm_ir import (
    load_processed as load_vsm_processed,
    build_term_stats,
    tfidf_matrix,
    search as vsm_search,
)
from src.eval import precision_recall_f1, mean_average_precision, ndcg_at_k

ROOT = Path(__file__).resolve().parents[1]


def top_terms_for_doc(doc_text, top_n=5):
    toks = doc_text.split()
    freq = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in items[:top_n]]


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["boolean", "vsm"], default="vsm")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--scheme", choices=["normal", "sublinear", "bm25"], default="normal")
    parser.add_argument("--eval", type=str, help="path to goldsets.json (optional)")
    args = parser.parse_args()

    if args.model == "boolean":
        docs, files = load_boolean_processed()
        inv = build_inverted_index(docs)
        result_docs, explanation = explain(args.query, inv)

        # === Seragamkan format output dengan VSM ===
        print("Boolean results:")
        if not result_docs:
            print("(tidak ada dokumen yang cocok)")
        else:
            for idx in sorted(result_docs):
                if idx < len(files):
                    docname = files[idx]
                    print(f"- {docname} (score=1.0000)")  # Boolean → skor 1.0
                    print("  top terms:", top_terms_for_doc(docs[idx], top_n=6))

        print("\nExplain terms:", explanation.get("terms", []))

    else:
        docs, names = load_vsm_processed()
        vocab, t2i, TF, DF, lens, avg_len = build_term_stats(docs)
        TFIDF_normal, idf = tfidf_matrix(TF, DF, len(docs), sublinear=False)
        TFIDF_sub, _ = tfidf_matrix(TF, DF, len(docs), sublinear=True)

        if args.scheme == "normal":
            TFIDF = TFIDF_normal
            results = vsm_search(
                args.query,
                TFIDF=TFIDF,
                t2i=t2i,
                idf=idf,
                names=names,
                docs=docs,
                top_k=args.k,
                scheme="normal",
            )
        elif args.scheme == "sublinear":
            TFIDF = TFIDF_sub
            results = vsm_search(
                args.query,
                TFIDF=TFIDF,
                t2i=t2i,
                idf=idf,
                names=names,
                docs=docs,
                top_k=args.k,
                scheme="sublinear",
            )
        else:  # bm25
            results = vsm_search(
                args.query,
                TF=TF,
                DF=DF,
                lens=lens,
                avg_len=avg_len,
                t2i=t2i,
                names=names,
                docs=docs,
                top_k=args.k,
                scheme="bm25",
            )

        print("VSM results:")
        for doc, score, idx in results:
            print(f"- {doc} (score={score:.4f})")
            print("  top terms:", top_terms_for_doc(docs[idx], top_n=6))

    # === Optional evaluation ===
    if args.eval:
        gold_path = Path(args.eval)
        if gold_path.exists():
            goldsets = json.loads(gold_path.read_text(encoding="utf-8"))

            if args.model == "vsm":
                print("\nEvaluating VSM scheme:", args.scheme)
                retrieved_lists = []
                gold_lists = []

                for q, gold in goldsets.items():
                    if args.scheme == "bm25":
                        res = vsm_search(
                            q,
                            TF=TF,
                            DF=DF,
                            lens=lens,
                            avg_len=avg_len,
                            t2i=t2i,
                            names=names,
                            docs=docs,
                            top_k=args.k,
                            scheme="bm25",
                        )
                    else:
                        TFIDF_use = TFIDF_sub if args.scheme == "sublinear" else TFIDF_normal
                        res = vsm_search(
                            q,
                            TFIDF=TFIDF_use,
                            t2i=t2i,
                            idf=idf,
                            names=names,
                            docs=docs,
                            top_k=args.k,
                            scheme=args.scheme,
                        )

                    retrieved_lists.append([r[0] for r in res])
                    gold_lists.append(gold)

                # === Compute evaluation metrics ===
                map_val = mean_average_precision(gold_lists, retrieved_lists, args.k)
                ndcg_val = sum(
                    ndcg_at_k(r, g, args.k) for r, g in zip(retrieved_lists, gold_lists)
                ) / len(gold_lists)
                print(f"\nMAP@{args.k} = {map_val:.4f}, nDCG@{args.k} = {ndcg_val:.4f}")

            else:
                print("[Eval] Boolean eval not implemented here — gunakan notebook khusus.")
        else:
            print("[WARN] goldsets file not found:", gold_path)


if __name__ == "__main__":
    cli()