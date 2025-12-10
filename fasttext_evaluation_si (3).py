# -*- coding: utf-8 -*-
"""FastText_Evaluation
 Spearman correlation evaluation
"""

import numpy as np
import sys
import os
import argparse
from scipy.stats import spearmanr

def load_oov_vectors(oovvector_file):
    """Load OOV word vectors from oovvector_file file.
    
    Args:
        npz_path: Path to oovvector_file file containing OOV word vectors
        
    Returns:
        dict: Dictionary mapping words to their vectors (np.ndarray)
    """
    oov_vectors = {}
    print(f"Loading OOV vectors from {oovvector_file}...")
    with np.load(oovvector_file) as data:
        for word in data.keys():
            oov_vectors[word] = data[word].astype(np.float32)
    print(f"Loaded {len(oov_vectors)} OOV vectors.")
    return oov_vectors

def load_text_vectors(vector_path):
    """FastText 텍스트 포맷 벡터 로드 (사용자 제공 함수)"""
    vectors = {}
    print(f"Loading vectors from {vector_path}...")
    try:
        with open(vector_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split()
            # 헤더가 없는 경우 처리 (GloVe 등의 포맷일 수 있음)
            if len(header) > 2:
                # 헤더가 없고 바로 데이터인 경우 파일을 다시 처음으로
                f.seek(0)
                vocab_size = None
                dim = None
            else:
                vocab_size, dim = int(header[0]), int(header[1])

            for line in f:
                parts = line.rstrip().split()
                # 헤더가 없었다면 첫 라인에서 dim 결정
                if dim is None:
                    dim = len(parts) - 1

                if len(parts) != dim + 1:
                    continue
                word = parts[0]
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                vectors[word] = vec
    except UnicodeDecodeError:
        print(f"UTF-8 decoding failed for {vector_path}. Trying 'latin-1'...")
        try:
            with open(vector_path, "r", encoding="latin-1") as f:
                header = f.readline().strip().split()
                if len(header) > 2:
                    f.seek(0)
                    dim = None
                else:
                    vocab_size, dim = int(header[0]), int(header[1])

                for line in f:
                    parts = line.rstrip().split()
                    if dim is None: dim = len(parts) - 1
                    if len(parts) != dim + 1: continue
                    word = parts[0]
                    vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    vectors[word] = vec
        except UnicodeDecodeError as e:
            print(f"Both UTF-8 and latin-1 decoding failed for {vector_path}.")
            raise e

    print(f"Loaded {len(vectors)} vectors.")
    return vectors

def cosine_similarity(v1, v2):
    """두 벡터 간의 코사인 유사도 계산"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)
'''
def evaluate_similarity(vectors, data_path):
    """
    단어 유사도(Word Similarity) 평가
    데이터 포맷: word1 word2 human_score (공백, 탭, 또는 콤마 구분 지원)
    평가 지표: Spearman Rank Correlation
    """
    if not os.path.exists(data_path):
        print(f"[Skip] File not found: {data_path}")
        return

    manual_scores = []
    vector_scores = []
    missed = 0

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or len(line.strip()) == 0:
                continue

            # 구분자 처리 로직 개선: 탭 -> 콤마 -> 공백 순서로 시도
            line = line.strip()
            if '\t' in line:
                parts = line.split('\t')
            elif ',' in line:
                parts = line.split(',')
            else:
                parts = line.split()

            if len(parts) < 3:
                continue

            # CSV 등에서 공백이 포함된 경우 제거 (예: " word1 ", "word2", "0.5")
            w1 = parts[0].strip().lower()
            w2 = parts[1].strip().lower()
            try:
                score = float(parts[2].strip())
            except ValueError:
                continue # 점수가 숫자가 아니면 스킵

            if w1 in vectors and w2 in vectors:
                manual_scores.append(score)
                sim = cosine_similarity(vectors[w1], vectors[w2])
                vector_scores.append(sim)
            else:
                missed += 1

    if not manual_scores:
        print(f"No valid pairs found in {data_path}")
        return

    correlation, _ = spearmanr(manual_scores, vector_scores)
    print(f"Dataset: {os.path.basename(data_path)}")
    print(f" - Found pairs: {len(manual_scores)}")
    print(f" - Missed pairs: {missed}")
    print(f" - Spearman Correlation: {correlation:.4f}")
    return correlation
'''

def evaluate_similarity(vectors, data_path):
    """
    단어 유사도(Word Similarity) 평가 (nan 값 포함 쌍은 무시)
    """
    import math

    if not os.path.exists(data_path):
        print(f"[Skip] File not found: {data_path}")
        return

    manual_scores = []
    vector_scores = []
    word_pairs = []

    skipped_nan = 0
    skipped_nan_list = []
    missed = 0

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#') or len(line.strip()) == 0:
                continue

            line = line.strip()
            if '\t' in line:
                parts = line.split('\t')
            elif ',' in line:
                parts = line.split(',')
            else:
                parts = line.split()

            if len(parts) < 3:
                continue

            w1 = parts[0].strip().lower()
            w2 = parts[1].strip().lower()
            try:
                score = float(parts[2].strip())
            except ValueError:
                continue

            if w1 in vectors and w2 in vectors:
                sim = cosine_similarity(vectors[w1], vectors[w2])
                # nan 혹은 inf 점수 쌍은 버림
                if any([
                    math.isnan(score), math.isnan(sim),
                    math.isinf(score), math.isinf(sim)
                ]):
                    skipped_nan += 1
                    skipped_nan_list.append(f"{w1}\t{w2}")
                    continue
                manual_scores.append(score)
                vector_scores.append(sim)
                word_pairs.append((w1, w2))
            else:
                missed += 1

    if not manual_scores:
        print(f"No valid pairs found in {data_path}")
        return

    correlation, _ = spearmanr(manual_scores, vector_scores)
    print(f"Dataset: {os.path.basename(data_path)}")
    print(f" - Found pairs: {len(manual_scores)}")
    print(f" - Missed pairs: {missed}")
    print(f" - Spearman Correlation: {correlation:.4f}")

    print(f" - Skipped (NaN/Inf pairs): {skipped_nan}")
    if skipped_nan > 0:
        print(f" - Skipped word pairs (showing up to 10):")
        for pair in skipped_nan_list[:10]:
            print(f"    {pair}")
    return correlation
    
def evaluate_analogy(vectors, data_path, top_k=1):
    """
    단어 유추(Word Analogy) 평가 (A:B :: C:?)
    데이터 포맷: word1 word2 word3 word4 (A B C D -> A:B :: C:D)
    평가 지표: Accuracy
    """
    if not os.path.exists(data_path):
        print(f"[Skip] File not found: {data_path}")
        return

    # 빠른 계산을 위해 모든 벡터 정규화 및 행렬화
    print("Normalizing vectors for fast analogy search...")
    vocab_list = list(vectors.keys())
    vocab_map = {w: i for i, w in enumerate(vocab_list)}
    matrix = np.array([vectors[w] for w in vocab_list])

    # L2 정규화 ( ||v|| = 1 )
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / (norm + 1e-10) # 0으로 나누기 방지

    correct = 0
    total = 0
    skipped = 0

    print(f"Evaluating Analogy on {os.path.basename(data_path)}...")

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 진행상황 표시를 위한 설정
    total_lines = len(lines)

    for idx, line in enumerate(lines):
        if line.startswith(':') or line.startswith('#') or not line.strip():
            continue # 섹션 헤더나 주석 스킵

        parts = line.lower().split()
        if len(parts) != 4:
            continue

        a, b, c, expected = parts

        # 단어가 하나라도 없으면 스킵 (OOV 처리)
        if a not in vocab_map or b not in vocab_map or c not in vocab_map or expected not in vocab_map:
            skipped += 1
            continue

        total += 1

        # 벡터 연산: target = b - a + c
        # 정규화된 벡터 공간에서는 3CosMul 등이 더 좋으나, 논문 표준은 3CosAdd (b - a + c)
        vec_a = matrix[vocab_map[a]]
        vec_b = matrix[vocab_map[b]]
        vec_c = matrix[vocab_map[c]]

        target_vec = vec_b - vec_a + vec_c

        # 정규화하여 코사인 유사도 검색 준비
        target_vec = target_vec / np.linalg.norm(target_vec)

        # 행렬 곱으로 코사인 유사도 전체 계산 (Score = Matrix dot Target)
        scores = np.dot(matrix, target_vec)

        # 자기 자신(input words)은 결과에서 제외하는 것이 일반적
        input_indices = {vocab_map[a], vocab_map[b], vocab_map[c]}

        # Top-K 찾기 (자신 제외)
        # argpartition으로 상위 몇 개만 빠르게 추출 후 정렬
        best_indices = np.argpartition(scores, -(top_k + 3))[-(top_k + 3):]
        sorted_indices = best_indices[np.argsort(scores[best_indices])[::-1]]

        found = False
        count = 0
        for i in sorted_indices:
            if i in input_indices:
                continue

            predicted_word = vocab_list[i]
            if predicted_word == expected:
                correct += 1
                found = True

            count += 1
            if count >= top_k:
                break

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{total_lines}...")

    acc = correct / total if total > 0 else 0.0
    print(f"Dataset: {os.path.basename(data_path)}")
    print(f" - Total valid questions: {total}")
    print(f" - Skipped (OOV): {skipped}")
    print(f" - Accuracy: {acc * 100:.2f}%")
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='FastText Word Vector Evaluation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example:
  python fasttext_evaluation.py vectors.vec ./data/
  python fasttext_evaluation.py vectors.vec ./data/ --sisg --oov_npz oov_vectors.npz
        '''
    )
    parser.add_argument('vector_file', help='Path to the word vector file')
    parser.add_argument('data_target', help='Path to evaluation data file or directory')
    parser.add_argument('--sisg', action='store_true',
                        help='Enable SISG mode to include OOV word vectors in evaluation')
    parser.add_argument('--oovvector_file', type=str, default=None,
                        help='Path to oovvector_file file containing OOV word vectors (required when --sisg is enabled)')
    
    args = parser.parse_args()

    # sisg 옵션이 활성화된 경우 oov_npz 파일 경로가 필요함
    if args.sisg and not args.oovvector_file:
        parser.error("--oovvector_file is required when --sisg is enabled")

    vector_file = args.vector_file
    data_target = args.data_target

    # 1. 벡터 로드
    vectors = load_text_vectors(vector_file)

    # 2. sisg 옵션이 활성화된 경우 OOV 벡터 추가
    if args.sisg:
        print(f"\n--- SISG Mode Enabled ---")
        oov_vectors = load_oov_vectors(args.oovvector_file)
        # OOV 벡터를 기존 벡터에 추가 (기존에 있는 단어는 덮어쓰지 않음)
        added_count = 0
        for word, vec in oov_vectors.items():
            if word not in vectors:
                vectors[word] = vec
                added_count += 1
        print(f"Added {added_count} OOV vectors to vocabulary.")
        print(f"Total vocabulary size: {len(vectors)}")

    # 평가할 파일 목록 수집
    files_to_eval = []
    if os.path.isdir(data_target):
        for f in os.listdir(data_target):
            files_to_eval.append(os.path.join(data_target, f))
    else:
        files_to_eval.append(data_target)

    # 3. 평가 실행
    print("\n--- Starting Evaluation ---")
    for file_path in files_to_eval:
        filename = os.path.basename(file_path).lower()

        # 파일 이름이나 내용으로 태스크 추정 (간단한 로직)
        # 보통 WS353, rw, men 등은 유사도, questions-words는 유추
        is_analogy = False

        # 파일 확장자가 .csv이면 보통 유사도 파일 (WS353_RO.csv 등)
        if filename.endswith('.csv'):
             is_analogy = False
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                # 유추 데이터는 보통 4단어 (A B C D)
                # 유사도 데이터는 보통 3컬럼 (A B Score)
                # 콤마, 탭, 공백 모두 고려하여 분리 시도
                if '\t' in first_line: parts = first_line.split('\t')
                elif ',' in first_line: parts = first_line.split(',')
                else: parts = first_line.split()

                # 유추(Analogy)는 4개 필드 (A B C D)
                if len(parts) == 4 and not parts[0].replace('.','',1).isdigit():
                    is_analogy = True
                elif filename.startswith('questions-words'):
                    is_analogy = True

        if is_analogy:
            print(f"\n[Task: Word Analogy] {filename}")
            evaluate_analogy(vectors, file_path)
        else:
            print(f"\n[Task: Word Similarity] {filename}")
            evaluate_similarity(vectors, file_path)
