#!/usr/bin/env python3
"""
Czech Word Analogy Test Script

This script tests word vectors using Czech analogy test sets from:
https://github.com/Svobikl/cz_corpus/tree/master/pairs

Based on fastText's analogies() implementation:
https://github.com/facebookresearch/fastText/blob/main/src/main.cc

Usage:
    python analogy_test_cs.py --vectors vec_cs.vec --download-test
    python analogy_test_cs.py --vectors vec_cs.vec --test-file czech_emb_corpus_no_phrase.txt
    python analogy_test_cs.py --vectors vec_cs.vec --convert-csv pairs/*.csv
"""

import argparse
import os
import sys
import urllib.request
import numpy as np
from collections import defaultdict


class WordVectors:
    """Load and query word vectors in fastText .vec format"""
    
    def __init__(self, vec_path):
        """Load word vectors from fastText .vec format file
        
        Format:
        First line: <vocab_size> <vector_dim>
        Following lines: <word> <vec[0]> <vec[1]> ... <vec[dim-1]>
        """
        self.word_to_vec = {}
        self.vector_dim = 0
        
        print(f"Loading vectors from {vec_path}...")
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
        
        print(f"Loaded {len(self.word_to_vec)} word vectors")
        
        # Normalize vectors for cosine similarity
        print("Normalizing vectors...")
        for word in self.word_to_vec:
            norm = np.linalg.norm(self.word_to_vec[word])
            if norm > 0:
                self.word_to_vec[word] = self.word_to_vec[word] / norm
    
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
                print(f"Warning: Invalid format at line {line_num}: {line}")
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
                        help="Path to word vectors file (.vec format)")
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
