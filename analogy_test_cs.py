#!/usr/bin/env python3
"""
Czech Word Analogy Test Script

This script tests word vectors using Czech analogy test sets from:
https://github.com/Svobikl/cz_corpus/tree/master/pairs

Based on fastText's analogies() implementation:
https://github.com/facebookresearch/fastText/blob/main/src/main.cc

Supports both fastText .vec (text) and .bin (binary) formats.

Usage:
    python analogy_test_cs.py --vectors vec_cs.vec --download-test
    python analogy_test_cs.py --vectors vec_cs.bin --download-test
    python analogy_test_cs.py --vectors vec_cs.vec --test-file czech_emb_corpus_no_phrase.txt
    python analogy_test_cs.py --vectors vec_cs.vec --convert-csv pairs/*.csv
"""

import argparse
import os
import sys
import urllib.request
import numpy as np
import struct
from collections import defaultdict


class WordVectors:
    """Load and query word vectors in fastText .vec or .bin format"""
    
    def __init__(self, vec_path):
        """Load word vectors from fastText format file
        
        Supports both:
        - Text .vec format:
          First line: <vocab_size> <vector_dim>
          Following lines: <word> <vec[0]> <vec[1]> ... <vec[dim-1]>
        
        - Binary .bin format:
          fastText binary format with vocab and vectors
        """
        self.word_to_vec = {}
        self.vector_dim = 0
        
        print(f"Loading vectors from {vec_path}...")
        
        # Detect file format based on extension or content
        if vec_path.endswith('.bin'):
            self._load_binary(vec_path)
        else:
            self._load_text(vec_path)
        
        print(f"Loaded {len(self.word_to_vec)} word vectors")
        
        # Normalize vectors for cosine similarity (in-place for memory efficiency)
        print("Normalizing vectors...")
        for word in self.word_to_vec:
            norm = np.linalg.norm(self.word_to_vec[word])
            if norm > 0:
                self.word_to_vec[word] /= norm
    
    def _load_text(self, vec_path):
        """Load word vectors from text .vec format"""
        with open(vec_path, 'r', encoding='utf-8') as f:
            # Read header
            header = f.readline().strip().split()
            vocab_size = int(header[0])
            self.vector_dim = int(header[1])
            
            print(f"Vocabulary size: {vocab_size}, Vector dimension: {self.vector_dim}")
            
            # Read vectors
            for line_num, line in enumerate(f, start=1):
                parts = line.strip().split()
                if len(parts) < self.vector_dim + 1:
                    continue
                
                word = parts[0]
                vec = np.array([float(x) for x in parts[1:self.vector_dim + 1]], dtype=np.float32)
                self.word_to_vec[word] = vec
                
                if line_num % 10000 == 0:
                    print(f"  Loaded {line_num} vectors...", flush=True)
    
    def _load_binary(self, bin_path):
        """Load word vectors from binary .bin format
        
        Binary format structure:
        - Header with magic number, version, metadata
        - Dictionary with vocabulary
        - Matrix with vectors
        """
        try:
            with open(bin_path, 'rb') as f:
                # Read magic number and version
                magic_data = f.read(4)
                if len(magic_data) != 4:
                    raise ValueError("File too short: cannot read magic number")
                magic = struct.unpack('i', magic_data)[0]
                
                # Validate magic number (fastText uses 793712314)
                if magic != 793712314:
                    raise ValueError(f"Invalid fastText binary file: magic number {magic} != 793712314")
                
                version_data = f.read(4)
                if len(version_data) != 4:
                    raise ValueError("File too short: cannot read version")
                version = struct.unpack('i', version_data)[0]
                
                # Read dimensions
                vocab_data = f.read(8)
                if len(vocab_data) != 8:
                    raise ValueError("File too short: cannot read vocabulary size")
                vocab_size = struct.unpack('q', vocab_data)[0]
                
                dim_data = f.read(8)
                if len(dim_data) != 8:
                    raise ValueError("File too short: cannot read vector dimension")
                self.vector_dim = struct.unpack('q', dim_data)[0]
                
                # Validate dimensions
                if vocab_size <= 0 or vocab_size > 100000000:
                    raise ValueError(f"Invalid vocabulary size: {vocab_size}")
                if self.vector_dim <= 0 or self.vector_dim > 10000:
                    raise ValueError(f"Invalid vector dimension: {self.vector_dim}")
                
                # Skip other metadata (args)
                nwords = struct.unpack('i', f.read(4))[0]
                size_ = struct.unpack('q', f.read(8))[0]
                nwords_arg = struct.unpack('i', f.read(4))[0]
                bucket_arg = struct.unpack('i', f.read(4))[0]
                minn_arg = struct.unpack('i', f.read(4))[0]
                maxn_arg = struct.unpack('i', f.read(4))[0]
                neg_arg = struct.unpack('i', f.read(4))[0]
                wordNgrams_arg = struct.unpack('i', f.read(4))[0]
                loss_arg = struct.unpack('i', f.read(4))[0]
                model_arg = struct.unpack('i', f.read(4))[0]
                epoch_arg = struct.unpack('i', f.read(4))[0]
                minCount_arg = struct.unpack('q', f.read(8))[0]
                label_arg_len_data = f.read(4)
                if len(label_arg_len_data) != 4:
                    raise ValueError("File truncated: cannot read label_arg length")
                label_arg_len = struct.unpack('i', label_arg_len_data)[0]
                if label_arg_len < 0 or label_arg_len > 1000:
                    raise ValueError(f"Invalid label_arg length: {label_arg_len}")
                label_arg_data = f.read(label_arg_len)
                if len(label_arg_data) != label_arg_len:
                    raise ValueError(f"File truncated: expected {label_arg_len} bytes for label_arg")
                label_arg = label_arg_data.decode('utf-8')
                t_arg = struct.unpack('d', f.read(8))[0]
                lrUpdateRate_arg = struct.unpack('i', f.read(4))[0]
                dim_arg = struct.unpack('i', f.read(4))[0]
                ws_arg = struct.unpack('i', f.read(4))[0]
                lr_arg = struct.unpack('d', f.read(8))[0]
                verbose_arg = struct.unpack('i', f.read(4))[0]
                pretrainedVectors_arg_len_data = f.read(4)
                if len(pretrainedVectors_arg_len_data) != 4:
                    raise ValueError("File truncated: cannot read pretrainedVectors_arg length")
                pretrainedVectors_arg_len = struct.unpack('i', pretrainedVectors_arg_len_data)[0]
                if pretrainedVectors_arg_len < 0 or pretrainedVectors_arg_len > 10000:
                    raise ValueError(f"Invalid pretrainedVectors_arg length: {pretrainedVectors_arg_len}")
                pretrainedVectors_arg_data = f.read(pretrainedVectors_arg_len)
                if len(pretrainedVectors_arg_data) != pretrainedVectors_arg_len:
                    raise ValueError(f"File truncated: expected {pretrainedVectors_arg_len} bytes for pretrainedVectors_arg")
                pretrainedVectors_arg = pretrainedVectors_arg_data.decode('utf-8')
                saveOutput_arg = struct.unpack('?', f.read(1))[0]
                seed_arg = struct.unpack('i', f.read(4))[0]
                qout_arg = struct.unpack('?', f.read(1))[0]
                retrain_arg = struct.unpack('?', f.read(1))[0]
                qnorm_arg = struct.unpack('?', f.read(1))[0]
                cutoff_arg = struct.unpack('Q', f.read(8))[0]
                dsub_arg = struct.unpack('Q', f.read(8))[0]
                
                print(f"Vocabulary size: {vocab_size}, Vector dimension: {self.vector_dim}")
                
                # Read vocabulary (dictionary)
                dict_size = struct.unpack('i', f.read(4))[0]
                dict_nwords = struct.unpack('i', f.read(4))[0]
                dict_nlabels = struct.unpack('i', f.read(4))[0]
                dict_ntokens = struct.unpack('q', f.read(8))[0]
                dict_pruneidx_size = struct.unpack('q', f.read(8))[0]
                
                # Read words
                words = []
                for i in range(dict_size):
                    word_len_data = f.read(4)
                    if len(word_len_data) != 4:
                        raise ValueError(f"File truncated: cannot read word length for word {i}")
                    word_len = struct.unpack('i', word_len_data)[0]
                    
                    # Validate word length to prevent excessive memory allocation
                    if word_len < 0 or word_len > 10000:
                        raise ValueError(f"Invalid word length {word_len} for word {i}")
                    
                    word_data = f.read(word_len)
                    if len(word_data) != word_len:
                        raise ValueError(f"File truncated: expected {word_len} bytes for word {i}, got {len(word_data)}")
                    word = word_data.decode('utf-8')
                    
                    count = struct.unpack('q', f.read(8))[0]
                    entry_type = struct.unpack('b', f.read(1))[0]
                    words.append(word)
                    
                    if (i + 1) % 10000 == 0:
                        print(f"  Loaded {i + 1} words...", flush=True)
                
                # Skip pruned index if exists
                if dict_pruneidx_size > 0:
                    for _ in range(dict_pruneidx_size):
                        struct.unpack('i', f.read(4))
                        struct.unpack('i', f.read(4))
                
                # Read input matrix (word vectors)
                input_m = struct.unpack('?', f.read(1))[0]
                input_m_rows = struct.unpack('q', f.read(8))[0]
                input_m_cols = struct.unpack('q', f.read(8))[0]
                
                # Validate matrix dimensions
                if input_m_cols != self.vector_dim:
                    raise ValueError(f"Matrix dimension mismatch: expected {self.vector_dim}, got {input_m_cols}")
                
                # Read vectors
                print(f"Reading {input_m_rows} vectors of dimension {input_m_cols}...")
                expected_bytes = input_m_cols * 4  # 4 bytes per float
                for i in range(min(len(words), input_m_rows)):
                    vec_data = f.read(expected_bytes)
                    if len(vec_data) != expected_bytes:
                        raise ValueError(f"File truncated: expected {expected_bytes} bytes for vector {i}, got {len(vec_data)}")
                    vec = np.frombuffer(vec_data, dtype=np.float32)
                    if i < dict_nwords:  # Only store word vectors, not subword vectors
                        self.word_to_vec[words[i]] = vec.copy()
                    
                    if (i + 1) % 10000 == 0:
                        print(f"  Loaded {i + 1} vectors...", flush=True)
        
        except struct.error as e:
            raise ValueError(f"Failed to parse binary file: {e}. File may be corrupted or not a valid fastText binary format.")
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode text in binary file: {e}. File may be corrupted.")
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Error loading binary file: {e}")
    
    def __contains__(self, word):
        """Check if word exists in vocabulary"""
        return word in self.word_to_vec
    
    def __getitem__(self, word):
        """Get vector for word"""
        return self.word_to_vec[word]
    
    def find_nearest(self, query_vec, k=10, exclude_words=None):
        """Find k nearest neighbors to query vector
        
        Args:
            query_vec: Query vector (will be normalized)
            k: Number of neighbors to return
            exclude_words: Set of words to exclude from results
            
        Returns:
            List of (word, similarity) tuples
        """
        # Normalize query vector
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        
        if exclude_words is None:
            exclude_words = set()
        
        # Compute similarities
        similarities = []
        for word, vec in self.word_to_vec.items():
            if word not in exclude_words:
                sim = np.dot(query_vec, vec)
                similarities.append((word, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]


def download_czech_analogy_files(output_dir="./czech_analogies"):
    """Download Czech analogy test files from cz_corpus repository
    
    Downloads the pre-built analogy test corpus files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    base_url = "https://raw.githubusercontent.com/Svobikl/cz_corpus/master/corpus/"
    
    # Test files available in the repository
    # czech_emb_corpus_no_phrase.txt is recommended for single word testing
    files = [
        "czech_emb_corpus_no_phrase.txt",
        "czech_emb_corpus.txt",
    ]
    
    downloaded = []
    for filename in files:
        url = base_url + filename
        output_path = os.path.join(output_dir, filename)
        
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"  Saved to {output_path}")
            downloaded.append(output_path)
        except urllib.error.HTTPError as e:
            print(f"  HTTP Error downloading {filename}: {e.code} {e.reason}")
        except urllib.error.URLError as e:
            print(f"  URL Error downloading {filename}: {e.reason}")
        except Exception as e:
            print(f"  Error downloading {filename}: {e}")
    
    return downloaded


def convert_csv_pairs_to_analogies(csv_files, output_path):
    """Convert CSV pair files to analogy test format
    
    CSV format: word1,word2 (one pair per line, first line is header)
    Output format: 
        : category_name
        word1_a word2_a word1_b word2_b
        
    This generates all possible combinations of pairs for testing.
    """
    print(f"Converting CSV pair files to analogy format...")
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for csv_file in csv_files:
            if not os.path.exists(csv_file):
                print(f"  Warning: File not found: {csv_file}")
                continue
            
            # Get category name from filename
            category = os.path.basename(csv_file).replace('.csv', '').replace('-', ' ').title()
            
            # Read pairs from CSV
            pairs = []
            with open(csv_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line or line_num == 1:  # Skip header or empty lines
                        continue
                    
                    parts = line.split(',')
                    if len(parts) == 2:
                        pairs.append((parts[0].strip(), parts[1].strip()))
            
            print(f"  {csv_file}: {len(pairs)} pairs")
            
            # Write category
            out_f.write(f": {category}\n")
            
            # Generate analogies: for each pair of pairs, create analogy
            # a:b :: c:d where (a,b) and (c,d) are different pairs
            # Note: This generates O(n²) analogies for n pairs, which is the standard
            # approach for comprehensive analogy testing (same as cz_corpus)
            for i, (a, b) in enumerate(pairs):
                for j, (c, d) in enumerate(pairs):
                    if i != j:  # Don't use the same pair twice
                        out_f.write(f"{a} {b} {c} {d}\n")
    
    print(f"Saved converted analogies to {output_path}")


def load_analogy_test_file(test_path):
    """Load analogy test file
    
    Format (similar to Google's word2vec analogy format):
    : category_name
    word_a word_b word_c word_d
    
    Each line represents: a:b :: c:d (a is to b as c is to d)
    Lines starting with ':' are category labels
    """
    analogies = []
    current_category = "unknown"
    
    print(f"Loading analogy test file: {test_path}")
    
    with open(test_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Category line
            if line.startswith(':'):
                current_category = line[1:].strip()
                continue
            
            # Analogy line
            parts = line.split()
            if len(parts) != 4:
                print(f"Warning: Invalid format at line {line_num}: expected 4 words (a b c d), got {len(parts)}: {line}")
                continue
            
            a, b, c, d = parts
            analogies.append({
                'category': current_category,
                'a': a, 'b': b, 'c': c, 'd': d
            })
    
    print(f"Loaded {len(analogies)} analogies")
    return analogies


def test_analogies(vectors, analogies, verbose=False):
    """Test word analogies
    
    For each analogy (a:b :: c:d):
    - Compute: vec(b) - vec(a) + vec(c)
    - Find nearest neighbor (excluding a, b, c)
    - Check if result matches d
    
    Returns:
        Dictionary with test results
    """
    results = {
        'total': 0,
        'correct': 0,
        'oov': 0,  # Out of vocabulary
        'by_category': defaultdict(lambda: {'total': 0, 'correct': 0, 'oov': 0})
    }
    
    print("\nTesting analogies...")
    print("=" * 80)
    
    for idx, analogy in enumerate(analogies, start=1):
        cat = analogy['category']
        a, b, c, d = analogy['a'], analogy['b'], analogy['c'], analogy['d']
        
        # Check if all words are in vocabulary
        if a not in vectors or b not in vectors or c not in vectors or d not in vectors:
            results['oov'] += 1
            results['by_category'][cat]['oov'] += 1
            if verbose:
                print(f"{idx}. {a}:{b} :: {c}:{d} - OOV")
            continue
        
        results['total'] += 1
        results['by_category'][cat]['total'] += 1
        
        # Compute analogy vector: vec(b) - vec(a) + vec(c)
        vec_a = vectors[a]
        vec_b = vectors[b]
        vec_c = vectors[c]
        
        query_vec = vec_b - vec_a + vec_c
        
        # Find nearest neighbor (excluding input words)
        exclude = {a, b, c}
        neighbors = vectors.find_nearest(query_vec, k=1, exclude_words=exclude)
        
        if neighbors:
            predicted, similarity = neighbors[0]
            is_correct = (predicted == d)
            
            if is_correct:
                results['correct'] += 1
                results['by_category'][cat]['correct'] += 1
            
            if verbose or (idx <= 10):  # Show first 10 examples
                status = "✓" if is_correct else "✗"
                print(f"{idx}. {a}:{b} :: {c}:{d} | Predicted: {predicted} | {status}")
        
        # Progress indicator
        if idx % 100 == 0:
            accuracy = results['correct'] / results['total'] * 100 if results['total'] > 0 else 0
            print(f"Progress: {idx}/{len(analogies)} | Accuracy so far: {accuracy:.2f}%")
    
    return results


def print_results(results):
    """Print test results in a formatted way"""
    print("\n" + "=" * 80)
    print("ANALOGY TEST RESULTS")
    print("=" * 80)
    
    total = results['total']
    correct = results['correct']
    oov = results['oov']
    
    accuracy = correct / total * 100 if total > 0 else 0
    coverage = total / (total + oov) * 100 if (total + oov) > 0 else 0
    
    print(f"\nOverall Statistics:")
    print(f"  Total questions: {total + oov}")
    print(f"  In vocabulary: {total} ({coverage:.2f}%)")
    print(f"  Out of vocabulary: {oov}")
    print(f"  Correct answers: {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    # Print per-category results
    if results['by_category']:
        print(f"\nPer-Category Results:")
        print(f"{'Category':<30} {'Total':>8} {'Correct':>8} {'Accuracy':>10} {'OOV':>6}")
        print("-" * 80)
        
        for cat in sorted(results['by_category'].keys()):
            cat_stats = results['by_category'][cat]
            cat_total = cat_stats['total']
            cat_correct = cat_stats['correct']
            cat_oov = cat_stats['oov']
            cat_accuracy = cat_correct / cat_total * 100 if cat_total > 0 else 0
            
            print(f"{cat:<30} {cat_total:>8} {cat_correct:>8} {cat_accuracy:>9.2f}% {cat_oov:>6}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Test Czech word vectors using analogy test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download test files and run test
  python analogy_test_cs.py --vectors vec_cs.vec --download-test
  
  # Run test with existing test file
  python analogy_test_cs.py --vectors vec_cs.vec --test-file czech_emb_corpus_no_phrase.txt
  
  # Run with verbose output
  python analogy_test_cs.py --vectors vec_cs.vec --test-file czech_emb_corpus_no_phrase.txt --verbose
  
  # Convert CSV pair files to analogy format
  python analogy_test_cs.py --vectors vec_cs.vec --convert-csv pairs/*.csv --csv-output my_analogies.txt
        """
    )
    
    parser.add_argument("--vectors", type=str, required=True,
                        help="Path to word vectors file (.vec or .bin format)")
    parser.add_argument("--test-file", type=str,
                        help="Path to analogy test file")
    parser.add_argument("--download-test", action="store_true",
                        help="Download Czech analogy test files from cz_corpus repository")
    parser.add_argument("--test-dir", type=str, default="./czech_analogies",
                        help="Directory for downloaded test files (default: ./czech_analogies)")
    parser.add_argument("--convert-csv", type=str, nargs='+',
                        help="Convert CSV pair files to analogy test format")
    parser.add_argument("--csv-output", type=str, default="./converted_analogies.txt",
                        help="Output path for converted CSV files (default: ./converted_analogies.txt)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed results for each analogy")
    
    args = parser.parse_args()
    
    # Handle CSV conversion mode
    if args.convert_csv:
        convert_csv_pairs_to_analogies(args.convert_csv, args.csv_output)
        print(f"\nConversion complete! Use --test-file {args.csv_output} to test the converted analogies.")
        return
    
    # Check if vectors file exists
    if not os.path.exists(args.vectors):
        print(f"Error: Vectors file not found: {args.vectors}")
        sys.exit(1)
    
    # Download test files if requested
    test_files = []
    if args.download_test:
        test_files = download_czech_analogy_files(args.test_dir)
        if not test_files:
            print("Error: Failed to download test files")
            sys.exit(1)
    elif args.test_file:
        if not os.path.exists(args.test_file):
            print(f"Error: Test file not found: {args.test_file}")
            sys.exit(1)
        test_files = [args.test_file]
    else:
        print("Error: Either --test-file or --download-test must be specified")
        parser.print_help()
        sys.exit(1)
    
    # Load vectors
    vectors = WordVectors(args.vectors)
    
    # Test each file
    for test_file in test_files:
        print(f"\n{'='*80}")
        print(f"Testing with: {test_file}")
        print(f"{'='*80}")
        
        analogies = load_analogy_test_file(test_file)
        results = test_analogies(vectors, analogies, verbose=args.verbose)
        print_results(results)


if __name__ == "__main__":
    main()
