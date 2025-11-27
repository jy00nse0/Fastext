# -*- coding: utf-8 -*-
"""
Analogy_Evaluator (Extended)

Original skeleton derived from svobik's evaluator, refactored to:
 - Load plain text fastText-style vectors (.vec) without gensim dependency
 - Optionally merge SISG-estimated OOV vectors from .npz
 - Evaluate word analogy datasets (Google-style or generated from pairs CSVs)
 - Auto-generate analogy data from Czech pairs (word similarity CSV) if requested
 - Provide per-category and overall accuracy with top-K support
 - Language summary lines for EN/CS: single OOV and total accuracy per language

Usage Examples:
  1) Evaluate a pre-made analogy file (Google EN set):
     python Analogy_Evaluator.py ./fastext/result/model_en.vec --analogy_file ./fastext/data/questions-words.txt --topk 1 --lang en

  2) Generate Czech analogies from pairs CSVs and evaluate in-memory:
     python Analogy_Evaluator.py ./fastext/result/model_cs.vec --pairs_dir ./external/cz_pairs --topk 1 --max_analogies_per_file 5000 --lang cs

  3) Include SISG OOV vectors:
     python Analogy_Evaluator.py ./fastext/result/model_cs.vec --pairs_dir ./external/cz_pairs --sisg --oov_npz ./fastext/oov/cs_oov.npz --lang cs

  4) Dump generated analogies to file while evaluating:
     python Analogy_Evaluator.py ./fastext/result/model_cs.vec --pairs_dir ./external/cz_pairs --dump_generated ./fastext/data/cs_analogies_generated.txt --lang cs

Notes:
 - Generating analogies from similarity pairs (rg65, ws353, etc.) is heuristic and may not be semantically robust.
 - Use --max_analogies_per_file to cap explosion (N pairs -> N*(N-1) analogies).
"""

import os
import sys
import argparse
import numpy as np
from typing import List, Tuple, Dict, Iterable

# -------------------------------
# Loading Vectors
# -------------------------------

def load_text_vectors(vector_path: str, lowercase: bool = True) -> Dict[str, np.ndarray]:
    """
    Load word vectors from fastText/GloVe-like text file.
    Handles header lines of form: <vocab_size> <dim> or absence thereof.
    """
    vectors = {}
    print(f"[INFO] Loading vectors: {vector_path}")
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Vector file not found: {vector_path}")

    try_encodings = ["utf-8", "latin-1"]
    last_error = None

    for enc in try_encodings:
        try:
            with open(vector_path, "r", encoding=enc) as f:
                first = f.readline().strip().split()
                if len(first) == 2 and first[0].isdigit() and first[1].isdigit():
                    # Header present
                    dim = int(first[1])
                else:
                    # No header
                    f.seek(0)
                    dim = None

                for line in f:
                    parts = line.rstrip().split()
                    if dim is None:
                        dim = len(parts) - 1
                    if len(parts) != dim + 1:
                        continue
                    w = parts[0]
                    if lowercase:
                        w = w.lower()
                    try:
                        vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    except ValueError:
                        continue
                    if vec.shape[0] != dim:
                        continue
                    vectors[w] = vec
            print(f"[INFO] Loaded {len(vectors)} vectors (dim={dim}).")
            return vectors
        except UnicodeDecodeError as e:
            last_error = e
            print(f"[WARN] Failed decoding with {enc}, trying next...")

    raise last_error if last_error else RuntimeError("Failed to load vectors with attempted encodings.")

def load_oov_vectors(npz_path: str, lowercase: bool = True) -> Dict[str, np.ndarray]:
    """
    Load OOV vectors saved in .npz (word -> vector).
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"OOV npz file not found: {npz_path}")
    print(f"[INFO] Loading OOV vectors: {npz_path}")
    data = np.load(npz_path)
    oov = {}
    for key in data.keys():
        w = key.lower() if lowercase else key
        oov[w] = data[key].astype(np.float32)
    print(f"[INFO] Loaded {len(oov)} OOV vectors.")
    return oov

def merge_oov(vectors: Dict[str, np.ndarray], oov: Dict[str, np.ndarray]) -> int:
    """
    Merge OOV vectors into existing dictionary without overwriting existing words.
    Returns number of added entries. Verifies dimensional consistency.
    """
    if not vectors:
        raise ValueError("Base vectors are empty; cannot merge OOV.")
    base_dim = len(next(iter(vectors.values())))
    added = 0
    mismatch = 0
    for w, v in oov.items():
        if len(v) != base_dim:
            mismatch += 1
            continue
        if w not in vectors:
            vectors[w] = v
            added += 1
    if mismatch:
        print(f"[WARN] Skipped {mismatch} OOV vectors due to dimension mismatch (expected {base_dim}).")
    print(f"[INFO] Added {added} new OOV vectors. Total vocab size: {len(vectors)}")
    return added

# -------------------------------
# Analogy Generation from pairs CSV
# -------------------------------

def read_pairs_csv(file_path: str, lowercase: bool = True) -> List[Tuple[str, str]]:
    """
    Read first two columns of a CSV/TSV file (similarity pair files).
    Auto-detect delimiter (, or tab). Ignores header if present.
    Returns list of (word1, word2).
    """
    pairs = []
    if not os.path.exists(file_path):
        print(f"[WARN] Pairs file not found: {file_path}")
        return pairs

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # heuristic: split by tab first, then comma, else whitespace
            if "\t" in line:
                parts = line.split("\t")
            elif "," in line:
                parts = line.split(",")
            else:
                parts = line.split()
            if len(parts) < 2:
                continue
            w1, w2 = parts[0].strip(), parts[1].strip()
            if lowercase:
                w1, w2 = w1.lower(), w2.lower()
            # skip header guesses (if numeric score at parts[2], fine; we ignore anyway)
            if w1.startswith("#"):
                continue
            pairs.append((w1, w2))
    return pairs

def generate_analogies_from_pairs(pairs: List[Tuple[str, str]],
                                  category_name: str,
                                  max_analogies: int = None) -> List[Tuple[str, str, str, str, str]]:
    """
    Create analogy quadruples from pair list:
       For distinct pairs (a,b) and (c,d): produce a b c d
    Returns list of 5-tuples: (category_header, a, b, c, d)
    :category_header is a line like ': category_name'
    """
    analogies = []
    if not pairs:
        return analogies
    header = f": {category_name}"
    count = 0
    for i, (a, b) in enumerate(pairs):
        for j, (c, d) in enumerate(pairs):
            if i == j:
                continue
            analogies.append((header, a, b, c, d))
            count += 1
            if max_analogies is not None and count >= max_analogies:
                break
        if max_analogies is not None and count >= max_analogies:
            break
    return analogies

def build_analogy_dataset_from_pairs_dir(pairs_dir: str,
                                         lowercase: bool,
                                         max_analogies_per_file: int = None) -> List[Tuple[str, str, str, str, str]]:
    """
    Iterate over CSV/TSV files in pairs_dir to construct a combined analogy dataset.
    Each file contributes its own category.
    """
    all_analogies = []
    if not os.path.isdir(pairs_dir):
        raise NotADirectoryError(f"Pairs directory not found: {pairs_dir}")
    files = [f for f in os.listdir(pairs_dir) if f.lower().endswith((".csv", ".tsv"))]
    if not files:
        print(f"[WARN] No CSV/TSV files found in {pairs_dir}")
        return all_analogies
    for f in files:
        path = os.path.join(pairs_dir, f)
        pairs = read_pairs_csv(path, lowercase=lowercase)
        category = os.path.splitext(f)[0].replace("-", "_")
        analogies = generate_analogies_from_pairs(pairs, category, max_analogies=max_analogies_per_file)
        print(f"[INFO] File {f}: pairs={len(pairs)} -> analogies generated={len(analogies)}")
        all_analogies.extend(analogies)
    return all_analogies

# -------------------------------
# Analogy Evaluation
# -------------------------------

def prepare_matrix(vectors: Dict[str, np.ndarray]) -> Tuple[List[str], Dict[str, int], np.ndarray]:
    vocab_list = list(vectors.keys())
    vocab_map = {w: i for i, w in enumerate(vocab_list)}
    mat = np.vstack([vectors[w] for w in vocab_list]).astype(np.float32)
    # normalize
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / (norms + 1e-10)
    return vocab_list, vocab_map, mat

def analogy_query(vec_a, vec_b, vec_c):
    return vec_b - vec_a + vec_c


def evaluate_analogy_dataset(vectors: Dict[str, np.ndarray],
                             analogy_lines: Iterable[Tuple[str, str, str, str, str]],
                             top_k: int = 1,
                             show_progress_every: int = 5000):
    """
    analogy_lines: iterable of (header_or_category, a, b, c, d)
    header_or_category lines start with ':' (category header).
    We assume preprocessed list where each analogy tuple includes its header context.
    """
    vocab_list, vocab_map, matrix = prepare_matrix(vectors)

    # Stats
    current_category = None
    cat_total = 0
    cat_correct = 0
    cat_skipped = 0

    category_results = []
    total_questions = 0
    total_correct = 0
    total_skipped = 0

    processed = 0

    for entry in analogy_lines:
        header, a, b, c, d = entry  # header starts with ': category'
        # Category boundary detection
        if header.startswith(":"):
            if current_category is None:
                current_category = header
            elif header != current_category:
                # finalize previous category
                if cat_total > 0:
                    acc = (cat_correct / cat_total) * 100.0
                else:
                    acc = 0.0
                category_results.append({
                    "category": current_category[2:].strip(),
                    "questions": cat_total,
                    "correct": cat_correct,
                    "skipped": cat_skipped,
                    "accuracy": acc
                })
                current_category = header
                cat_total = cat_correct = cat_skipped = 0

        # Evaluate analogy
        # a b c d
        total_questions += 1
        cat_total += 1
        # OOV skip
        if (a not in vocab_map) or (b not in vocab_map) or (c not in vocab_map) or (d not in vocab_map):
            total_skipped += 1
            cat_skipped += 1
            continue

        va = matrix[vocab_map[a]]
        vb = matrix[vocab_map[b]]
        vc = matrix[vocab_map[c]]
        target = analogy_query(va, vb, vc)
        norm = np.linalg.norm(target)
        if norm == 0:
            total_skipped += 1
            cat_skipped += 1
            continue
        target = target / norm

        scores = np.dot(matrix, target)
        input_indices = {vocab_map[a], vocab_map[b], vocab_map[c]}

        k_plus = min(top_k + 3, len(scores))
        best = np.argpartition(scores, -k_plus)[-k_plus:]
        ordered = best[np.argsort(scores[best])[::-1]]

        found = False
        picked = 0
        for idx in ordered:
            if idx in input_indices:
                continue
            predicted = vocab_list[idx]
            if predicted == d:
                found = True
                break
            picked += 1
            if picked >= top_k:
                break

        if found:
            total_correct += 1
            cat_correct += 1

        processed += 1
        if show_progress_every and processed % show_progress_every == 0:
            print(f"[INFO] Processed {processed} analogies...")

    # finalize last category
    if current_category is not None:
        if cat_total > 0:
            acc = (cat_correct / cat_total) * 100.0
        else:
            acc = 0.0
        category_results.append({
            "category": current_category[2:].strip(),
            "questions": cat_total,
            "correct": cat_correct,
            "skipped": cat_skipped,
            "accuracy": acc
        })

    overall_acc = (total_correct / (total_questions - total_skipped)) * 100.0 if (total_questions - total_skipped) > 0 else 0.0

    return {
        "total_questions": total_questions,
        "total_correct": total_correct,
        "total_skipped": total_skipped,
        "overall_accuracy": overall_acc,
        "categories": category_results
    }

# -------------------------------
# Utility: Load analogy file
# -------------------------------

def load_analogy_file(path: str, lowercase: bool = True) -> List[Tuple[str, str, str, str, str]]:
    """
    Load a Google-style analogy file:
      Lines starting with ':' are category headers
      Other lines: a b c d
    Return unified list of tuples (header, a, b, c, d)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Analogy file not found: {path}")
    lines_out = []
    current_header = ": default"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            if raw.startswith(":"):
                current_header = raw
                continue
            parts = raw.split()
            if len(parts) != 4:
                continue
            a, b, c, d = parts
            if lowercase:
                a, b, c, d = a.lower(), b.lower(), c.lower(), d.lower()
            lines_out.append((current_header, a, b, c, d))
    return lines_out

# -------------------------------
# Dump generated analogy data
# -------------------------------

def dump_analogies(analogies: List[Tuple[str, str, str, str, str]], out_path: str):
    """
    Write generated analogy dataset to file.
    """
    print(f"[INFO] Dumping {len(analogies)} analogy lines to {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        current_header = None
        for header, a, b, c, d in analogies:
            if header != current_header:
                f.write(f"{header}\n")
                current_header = header
            f.write(f"{a} {b} {c} {d}\n")

# -------------------------------
# Main
# -------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Extended Analogy Evaluator with fastText-style vector & SISG OOV support and language summaries."
    )
    ap.add_argument("vector_file", help="Path to base word vector file (.vec or text format)")
    group_input = ap.add_mutually_exclusive_group(required=True)
    group_input.add_argument("--analogy_file", help="Existing analogy file (Google format)")
    group_input.add_argument("--pairs_dir", help="Directory with similarity pair CSV/TSV files to auto-generate analogies")

    ap.add_argument("--topk", type=int, default=1, help="Top-K prediction threshold (default=1)")
    ap.add_argument("--lowercase", action="store_true", default=True, help="Lowercase all tokens (default True)")
    ap.add_argument("--no-lowercase", dest="lowercase", action="store_false", help="Disable lowercasing")
    ap.add_argument("--sisg", action="store_true", help="Enable SISG mode to merge OOV vectors")
    ap.add_argument("--oov_npz", type=str, help="Path to OOV npz file (required if --sisg)")
    ap.add_argument("--max_analogies_per_file", type=int, default=None,
                    help="Limit number of generated analogies per pairs file (to avoid explosion)")
    ap.add_argument("--dump_generated", type=str,
                    help="If provided, dump the generated analogy dataset to this file")
    ap.add_argument("--progress_every", type=int, default=5000, help="Progress print frequency")
    ap.add_argument("--lang", type=str, choices=["en", "cs"], help="Language label for summary lines (en|cs)")
    return ap.parse_args()

def main():
    args = parse_args()

    # Load base vectors
    vectors = load_text_vectors(args.vector_file, lowercase=args.lowercase)

    if args.sisg:
        if not args.oov_npz:
            raise ValueError("--oov_npz is required when --sisg is enabled.")
        print("[INFO] SISG mode enabled.")
        oov_vecs = load_oov_vectors(args.oov_npz, lowercase=args.lowercase)
        merge_oov(vectors, oov_vecs)

    # Prepare analogy data
    if args.analogy_file:
        analogy_data = load_analogy_file(args.analogy_file, lowercase=args.lowercase)
        print(f"[INFO] Loaded analogy file with {len(analogy_data)} lines.")
    else:
        analogy_data = build_analogy_dataset_from_pairs_dir(
            args.pairs_dir,
            lowercase=args.lowercase,
            max_analogies_per_file=args.max_analogies_per_file
        )
        print(f"[INFO] Generated {len(analogy_data)} analogy entries from pairs directory.")
        if args.dump_generated:
            dump_analogies(analogy_data, args.dump_generated)

    if not analogy_data:
        print("[ERROR] No analogy data available to evaluate.")
        return

    # Evaluate
    print("[INFO] Starting analogy evaluation...")
    result = evaluate_analogy_dataset(
        vectors,
        analogy_data,
        top_k=args.topk,
        show_progress_every=args.progress_every
    )

    # Report
    print("\n================ Evaluation Summary ================")
    print(f"Total Questions: {result['total_questions']}")
    print(f"Total Skipped (OOV or invalid): {result['total_skipped']}")
    print(f"Total Correct: {result['total_correct']}")
    print(f"Overall Accuracy (Top-{args.topk}): {result['overall_accuracy']:.2f}%")

    # Single OOV and accuracy lines per language (if --lang provided)
    if args.lang:
        oov_rate = (result['total_skipped'] / result['total_questions'] * 100.0) if result['total_questions'] else 0.0
        print("---------------------------------------------------")
        print(f"[{args.lang.upper()}] TOTAL OOV: {result['total_skipped']}/{result['total_questions']} ({oov_rate:.2f}%)")
        print(f"[{args.lang.upper()}] TOTAL ACCURACY (Top-{args.topk}): {result['overall_accuracy']:.2f}%")

    print("---------------------------------------------------")
    print("Per-Category Results:")
    for cat in result["categories"]:
        print(f"  [{cat['category']}] "
              f"Q={cat['questions']} "
              f"Correct={cat['correct']} "
              f"Skipped={cat['skipped']} "
              f"Acc={cat['accuracy']:.2f}%")


if __name__ == "__main__":
    main()