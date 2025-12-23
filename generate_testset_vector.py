#!/usr/bin/env python3
"""
Generate word vectors for test words from a saved FastText model.

This script:
1. Loads a model file saved by SaveModel in Fastext_20251116_Version1.cpp
2. Reads test words from CSV files
3. Splits test words into subwords using the same method as Fastext_20251116_Version1.cpp
4. Computes word vectors by averaging subword vectors from the model
5. Saves the resulting test word vectors to a file

Usage:
    python generate_testset_vector.py --model model.bin --test-dir test/ --output test_vectors.vec
    python generate_testset_vector.py --model model.bin --test-csv test/words.csv --output test_vectors.vec
"""

import argparse
import os
import struct
import numpy as np
from typing import List, Dict, Tuple, Optional


def fastext_hash(word: bytes) -> int:
    """
    FNV-1a hash function (same as FastTextHash in C++ code).
    Uses signed int8 for XOR operation to match C++ behavior.
    """
    h = 2166136261
    for byte in word:
        # Simulate signed int8 behavior for XOR
        signed_byte = byte if byte < 128 else byte - 256
        h = h ^ (signed_byte & 0xFFFFFFFF)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def get_word_hash(word: bytes, bucket_size: int) -> int:
    """Get subword bucket index (same as GetWordHash in C++ code)."""
    return fastext_hash(word) % bucket_size


def compute_subwords(word: str, maxn: int, bucket_size: int, vocab_size: int) -> List[int]:
    """
    Compute subword indices for a word (same as computeSubwords in C++ code).
    
    The word should already have < and > markers around it.
    Returns list of subword bucket indices (offset by vocab_size).
    """
    subwords = []
    word_bytes = word.encode('utf-8')
    buflen = len(word_bytes)
    
    if buflen == 0:
        return subwords
    
    i = 0
    while i < buflen:
        # Skip UTF-8 continuation bytes
        if (word_bytes[i] & 0xC0) == 0x80:
            i += 1
            continue
        
        for n in range(1, maxn + 1):
            ngram_bytes = bytearray()
            ngram_len = 0
            jj = i
            
            while jj < buflen and ngram_len < n:
                ngram_bytes.append(word_bytes[jj])
                jj += 1
                ngram_len += 1
                # Include UTF-8 continuation bytes
                while jj < buflen and (word_bytes[jj] & 0xC0) == 0x80:
                    ngram_bytes.append(word_bytes[jj])
                    jj += 1
            
            # Only add if not at word boundary (same condition as C++ code)
            # (n >= 1 && n <= 6) && !(i == 0 || jj == buflen)
            if 1 <= n <= 6 and not (i == 0 or jj == buflen):
                h = get_word_hash(bytes(ngram_bytes), bucket_size)
                idx = vocab_size + h
                subwords.append(idx)
        
        i += 1
    
    return subwords


def load_model(model_path: str) -> Tuple[Dict, np.ndarray, int, int, int]:
    """
    Load model saved by SaveModel in Fastext_20251116_Version1.cpp.
    
    Returns:
        vocab: Dictionary mapping word -> (count, type, subwords)
        syn0: Word vectors array (vocab_size + bucket_size, layer1_size)
        layer1_size: Vector dimension
        bucket_size: Subword bucket size
        maxn: Maximum n-gram length
    """
    print(f"Loading model from {model_path}...")
    
    with open(model_path, 'rb') as f:
        # 1. Read hyperparameters
        binary = struct.unpack('i', f.read(4))[0]
        debug_mode = struct.unpack('i', f.read(4))[0]
        window = struct.unpack('i', f.read(4))[0]
        min_count = struct.unpack('i', f.read(4))[0]
        num_threads = struct.unpack('i', f.read(4))[0]
        min_reduce = struct.unpack('i', f.read(4))[0]
        label_count = struct.unpack('i', f.read(4))[0]
        minn = struct.unpack('i', f.read(4))[0]
        maxn = struct.unpack('i', f.read(4))[0]
        model_name = struct.unpack('i', f.read(4))[0]
        loss_name = struct.unpack('i', f.read(4))[0]
        seed = struct.unpack('i', f.read(4))[0]
        vocab_max_size = struct.unpack('q', f.read(8))[0]
        layer1_size = struct.unpack('q', f.read(8))[0]
        bucket_size = struct.unpack('q', f.read(8))[0]
        token_count = struct.unpack('q', f.read(8))[0]
        train_words = struct.unpack('q', f.read(8))[0]
        word_count_actual = struct.unpack('q', f.read(8))[0]
        iter_count = struct.unpack('q', f.read(8))[0]
        file_size = struct.unpack('q', f.read(8))[0]
        classes = struct.unpack('q', f.read(8))[0]
        starting_alpha = struct.unpack('f', f.read(4))[0]
        alpha = struct.unpack('f', f.read(4))[0]
        sample = struct.unpack('f', f.read(4))[0]
        hs = struct.unpack('i', f.read(4))[0]
        negative = struct.unpack('i', f.read(4))[0]
        normalize_gradient = struct.unpack('?', f.read(1))[0]
        
        print(f"  Model parameters:")
        print(f"    layer1_size (vector dim): {layer1_size}")
        print(f"    bucket_size: {bucket_size}")
        print(f"    minn: {minn}, maxn: {maxn}")
        print(f"    window: {window}")
        print(f"    negative: {negative}")
        
        # 2. Read vocabulary
        vocab = {}
        vocab_size = 0
        
        # Read vocab entries until we hit the syn0 data
        # We need to know vocab_size first, but it's not explicitly saved
        # Let's read vocab entries until we can infer from the file structure
        
        # Actually, we need to read all vocab entries first
        # The vocab is stored without explicit count, so we read until
        # we reach the expected position for syn0
        
        # Calculate expected syn0 position (we need vocab_size)
        # Since vocab_size is not stored explicitly, we read vocab entries
        # and count them
        
        vocab_list = []
        while True:
            try:
                # Read word length
                len_data = f.read(4)
                if len(len_data) < 4:
                    break
                word_len = struct.unpack('i', len_data)[0]
                
                # Sanity check - if word_len is too large, we've hit syn0 data
                if word_len <= 0 or word_len > 10000:
                    # Seek back and break
                    f.seek(-4, 1)
                    break
                
                # Read word (including null terminator)
                word_bytes = f.read(word_len)
                if len(word_bytes) < word_len:
                    f.seek(-4 - len(word_bytes), 1)
                    break
                word = word_bytes[:-1].decode('utf-8', errors='replace')  # Remove null terminator
                
                # Read count
                cn_data = f.read(8)
                if len(cn_data) < 8:
                    f.seek(-4 - word_len - len(cn_data), 1)
                    break
                cn = struct.unpack('q', cn_data)[0]
                
                # Read type
                type_data = f.read(4)
                if len(type_data) < 4:
                    f.seek(-4 - word_len - 8 - len(type_data), 1)
                    break
                word_type = struct.unpack('i', type_data)[0]
                
                # Read subwords
                subword_size_data = f.read(4)
                if len(subword_size_data) < 4:
                    f.seek(-4 - word_len - 8 - 4 - len(subword_size_data), 1)
                    break
                subword_size = struct.unpack('i', subword_size_data)[0]
                
                if subword_size < 0 or subword_size > 100000:
                    f.seek(-4 - word_len - 8 - 4 - 4, 1)
                    break
                
                subwords_data = f.read(4 * subword_size)
                if len(subwords_data) < 4 * subword_size:
                    f.seek(-4 - word_len - 8 - 4 - 4 - len(subwords_data), 1)
                    break
                subwords = list(struct.unpack(f'{subword_size}i', subwords_data)) if subword_size > 0 else []
                
                vocab[word] = {'cn': cn, 'type': word_type, 'subwords': subwords}
                vocab_list.append(word)
                
            except struct.error:
                break
        
        vocab_size = len(vocab_list)
        print(f"  Vocabulary size: {vocab_size}")
        
        # 3. Read syn0 vectors
        total_vectors = vocab_size + bucket_size
        syn0_size = total_vectors * layer1_size
        
        print(f"  Reading {total_vectors} vectors ({vocab_size} vocab + {bucket_size} buckets)...")
        syn0_data = f.read(4 * syn0_size)
        if len(syn0_data) < 4 * syn0_size:
            raise ValueError(f"Expected {4 * syn0_size} bytes for syn0, got {len(syn0_data)}")
        
        syn0 = np.frombuffer(syn0_data, dtype=np.float32).reshape(total_vectors, layer1_size)
        
        print(f"  Model loaded successfully!")
        
    return vocab, syn0, int(layer1_size), int(bucket_size), int(maxn)


def read_test_words_from_csv(csv_path: str) -> List[str]:
    """
    Read test words from a CSV file.
    
    Assumes the CSV has word pairs or words in the first columns.
    Extracts unique words from all columns.
    """
    words = set()
    
    print(f"Reading test words from {csv_path}...")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split by comma and extract words
            parts = line.split(',')
            for part in parts:
                part = part.strip()
                # Skip empty parts, numbers, and common non-word entries
                if part and not part.replace('.', '').replace('-', '').isdigit():
                    # Skip single character alphabetic parts that are likely labels (n, v, a)
                    if len(part) > 1:
                        words.add(part)
    
    word_list = sorted(list(words))
    print(f"  Found {len(word_list)} unique words")
    
    return word_list


def read_test_words_from_directory(dir_path: str) -> List[str]:
    """Read test words from all CSV files in a directory."""
    words = set()
    
    print(f"Reading test words from directory {dir_path}...")
    
    for filename in os.listdir(dir_path):
        if filename.endswith('.csv'):
            csv_path = os.path.join(dir_path, filename)
            words.update(read_test_words_from_csv(csv_path))
    
    word_list = sorted(list(words))
    print(f"  Total unique words from directory: {len(word_list)}")
    
    return word_list


def compute_word_vector(word: str, vocab: Dict, syn0: np.ndarray, 
                        layer1_size: int, bucket_size: int, maxn: int) -> Optional[np.ndarray]:
    """
    Compute word vector for a given word.
    
    If word is in vocab, use its stored subwords.
    Otherwise, compute subwords on the fly.
    
    Returns averaged subword vectors (same as SaveVectors in C++ code).
    """
    vocab_size = len(vocab)
    
    if word in vocab:
        # Use stored subwords
        subwords = vocab[word]['subwords']
    else:
        # Compute subwords for OOV word
        # Add < and > markers (same as initNgrams in C++ code)
        if word != '</s>':
            word_with_markers = f'<{word}>'
        else:
            word_with_markers = word
        
        # For OOV words, we only have subword vectors (no word ID)
        subwords = compute_subwords(word_with_markers, maxn, bucket_size, vocab_size)
    
    if not subwords:
        return None
    
    # Compute average of subword vectors (same as SaveVectors)
    vec = np.zeros(layer1_size, dtype=np.float32)
    for subword_id in subwords:
        if 0 <= subword_id < len(syn0):
            vec += syn0[subword_id]
    
    vec /= len(subwords)
    
    return vec


def save_vectors(words: List[str], vectors: Dict[str, np.ndarray], 
                 output_path: str, layer1_size: int):
    """
    Save word vectors in FastText .vec format.
    
    Format:
    First line: <vocab_size> <vector_dim>
    Following lines: <word> <vec[0]> <vec[1]> ... <vec[dim-1]>
    """
    print(f"Saving vectors to {output_path}...")
    
    # Filter out words without vectors
    valid_words = [w for w in words if w in vectors and vectors[w] is not None]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"{len(valid_words)} {layer1_size}\n")
        
        for word in valid_words:
            vec = vectors[word]
            vec_str = ' '.join(f'{x:.6f}' for x in vec)
            f.write(f"{word} {vec_str}\n")
    
    print(f"  Saved {len(valid_words)} word vectors")


def main():
    parser = argparse.ArgumentParser(
        description="Generate word vectors for test words from a saved FastText model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate vectors for words in a test directory
  python generate_testset_vector.py --model model.bin --test-dir test/ --output test_vectors.vec
  
  # Generate vectors for words in a specific CSV file
  python generate_testset_vector.py --model model.bin --test-csv test/words.csv --output test_vectors.vec
        """
    )
    
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model file saved by SaveModel in Fastext_20251116_Version1.cpp")
    parser.add_argument("--test-dir", type=str,
                        help="Directory containing CSV files with test words")
    parser.add_argument("--test-csv", type=str,
                        help="Path to a single CSV file with test words")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for word vectors (.vec format)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress information")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.test_dir and not args.test_csv:
        print("Error: Either --test-dir or --test-csv must be specified")
        parser.print_help()
        return 1
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    # Load model
    vocab, syn0, layer1_size, bucket_size, maxn = load_model(args.model)
    
    # Read test words
    if args.test_dir:
        if not os.path.isdir(args.test_dir):
            print(f"Error: Test directory not found: {args.test_dir}")
            return 1
        test_words = read_test_words_from_directory(args.test_dir)
    else:
        if not os.path.exists(args.test_csv):
            print(f"Error: Test CSV file not found: {args.test_csv}")
            return 1
        test_words = read_test_words_from_csv(args.test_csv)
    
    if not test_words:
        print("Error: No test words found")
        return 1
    
    # Compute vectors for test words
    print(f"\nComputing vectors for {len(test_words)} test words...")
    vectors = {}
    in_vocab_count = 0
    oov_count = 0
    
    for i, word in enumerate(test_words):
        vec = compute_word_vector(word, vocab, syn0, layer1_size, bucket_size, maxn)
        if vec is not None:
            vectors[word] = vec
            if word in vocab:
                in_vocab_count += 1
            else:
                oov_count += 1
        
        if args.verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_words)} words...")
    
    print(f"  In vocabulary: {in_vocab_count}")
    print(f"  Out of vocabulary (computed from subwords): {oov_count}")
    print(f"  Failed (no subwords): {len(test_words) - len(vectors)}")
    
    # Save vectors
    save_vectors(test_words, vectors, args.output, layer1_size)
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
