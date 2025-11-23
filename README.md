# Fastext

FastText-style word embeddings training and evaluation tools.

## Overview

This repository provides tools for training word vectors and evaluating them using word analogy tests, with a focus on Czech language support.

## Scripts

### word2vec.py

Training script for word vectors using gensim's Word2Vec implementation with fastText-compatible output.

**Features:**
- UTF-8 safe tokenization (compatible with fastText C++)
- Streaming sentence generation for large corpora
- Progress tracking with ETA
- Saves models in both gensim and fastText .vec formats

**Usage:**
```bash
python word2vec.py --corpus corpus.txt --output model_name
```

### analogy_test_cs.py

Czech word analogy test script based on the fastText analogies() implementation.

**Features:**
- Loads word vectors from fastText .vec (text) or .bin (binary) format
- Tests analogies using vector arithmetic (a:b :: c:d → vec(b) - vec(a) + vec(c) ≈ vec(d))
- Downloads Czech analogy test sets from [cz_corpus repository](https://github.com/Svobikl/cz_corpus)
- Supports CSV pair file conversion
- Per-category accuracy reporting
- Out-of-vocabulary (OOV) tracking

**Usage:**

Download and test with Czech analogy corpus:
```bash
# Download Czech test sets and run tests (supports both .vec and .bin formats)
python analogy_test_cs.py --vectors vec_cs.vec --download-test
python analogy_test_cs.py --vectors vec_cs.bin --download-test

# Use specific test file
python analogy_test_cs.py --vectors vec_cs.vec --test-file czech_emb_corpus_no_phrase.txt

# Verbose output (shows each analogy prediction)
python analogy_test_cs.py --vectors vec_cs.vec --test-file czech_emb_corpus_no_phrase.txt --verbose
```

Convert CSV pair files to analogy format:
```bash
# Convert custom CSV pair files
python analogy_test_cs.py --vectors vec_cs.vec --convert-csv pairs/*.csv --csv-output my_analogies.txt

# Then test with converted file
python analogy_test_cs.py --vectors vec_cs.vec --test-file my_analogies.txt
```

**Test Data Format:**

The script expects analogy test files in the following format:
```
: Category Name
word_a word_b word_c word_d
word_a word_b word_e word_f
```

Where each line represents the analogy: `word_a:word_b :: word_c:word_d`

**CSV Pair Format:**

CSV files should have pairs of related words:
```csv
header1,header2
word1,word2
word3,word4
```

The converter generates all possible analogies from these pairs.

## Requirements

```bash
pip install numpy gensim
```

## Czech Analogy Test Sets

The script can automatically download Czech analogy test sets from the [cz_corpus repository](https://github.com/Svobikl/cz_corpus) by Svoboda & Brychcín (2016).

Two test files are available:
- `czech_emb_corpus_no_phrase.txt` - Recommended for single word testing (no phrases)
- `czech_emb_corpus.txt` - Includes phrase testing

Reference:
```
Svoboda, L., & Brychcín, T. (2016). New word analogy corpus for exploring 
embeddings of Czech words. Computational Linguistics and Intelligent Text 
Processing, 103-114. Springer.
```

## Example Workflow

```bash
# 1. Train word vectors on Czech corpus
python word2vec.py --corpus czech_text.txt --output vec_cs --epochs 10

# 2. Test the vectors with Czech analogies
python analogy_test_cs.py --vectors vec_cs.vec --download-test

# 3. View results by category
python analogy_test_cs.py --vectors vec_cs.vec --test-file czech_analogies/czech_emb_corpus_no_phrase.txt --verbose
```