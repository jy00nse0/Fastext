# Fastext_20251116 Variant Comparison

This document compares the four variant implementations of the `computeSubwords` function.

## Quick Reference Table

| Variant | Function Signature | Key Features |
|---------|-------------------|--------------|
| v1 | `void computeSubwords(const char* word, std::vector<int>& subwords)` | Baseline C-style implementation |
| v2 | `void computeSubwords(const std::string& word, std::vector<int>& subwords)` | Modern C++ with std::string |
| v3 | `void computeSubwords(const char* word, std::vector<int>& subwords, std::vector<std::string>& ngrams)` | Captures n-gram strings |
| v4 | `void computeSubwords(const char* word, std::vector<int>& subwords, int minNgram = -1, int maxNgram = -1)` | Configurable bounds with optimization |

## Detailed Comparison

### v1 (Baseline)
**Purpose:** Original implementation from commit 28e82a97a37bfae2bae93ac2b7cef97d391e28d6

**Implementation Details:**
- Uses C-style const char* parameter
- Creates local buffer with snprintf
- Standard n-gram generation from 1 to maxn
- No performance optimizations

**Use Case:** Reference implementation for comparison

### v2 (Modern C++)
**Purpose:** Demonstrate idiomatic C++ approach

**Implementation Details:**
- Uses const std::string& parameter
- Calls word.c_str() and word.length() directly
- Same n-gram generation logic as v1
- Avoids intermediate buffer copy at call sites

**Use Case:** Projects preferring C++ standard library

**Migration from v1:**
- Change call sites to construct std::string from char*
- No logic changes required

### v3 (With N-gram Output)
**Purpose:** Enable n-gram inspection and debugging

**Implementation Details:**
- Additional std::vector<std::string>& ngrams parameter
- Stores actual n-gram strings alongside indices
- Uses minn instead of hardcoded 1
- Clears ngrams vector before populating

**Use Case:** 
- Debugging subword generation
- Analysis of n-gram patterns
- Validation of hashing behavior

**Benefits:**
- Can inspect which n-grams are generated
- Useful for understanding subword segmentation
- Helps verify Unicode handling

### v4 (Configurable Parameters)
**Purpose:** Provide flexibility and optimization

**Implementation Details:**
- Optional minNgram and maxNgram parameters (default -1)
- Falls back to global minn/maxn when parameters are -1
- Pre-allocates vector capacity: `subwords.reserve(subwords.size() + buflen * (maxNgram - minNgram + 1))`
- Same core logic but with runtime bounds

**Use Case:**
- Experimentation with different n-gram ranges
- Performance-critical paths
- A/B testing different configurations

**Benefits:**
- Can override n-gram bounds per call
- Reduces vector reallocation overhead
- More flexible for research

## Performance Considerations

1. **v1 (Baseline):** Standard performance, multiple snprintf calls at call sites
2. **v2 (Modern C++):** Similar to v1, avoids snprintf at some call sites
3. **v3 (With Output):** Slower due to string construction, use only for debugging
4. **v4 (Optimized):** Fastest due to vector pre-allocation, configurable bounds

## Compilation

All variants compile successfully with C++17:

```bash
g++ -std=c++17 -O2 -Wall -pthread -c Fastext_20251116_v1.cpp -o v1.o
g++ -std=c++17 -O2 -Wall -pthread -c Fastext_20251116_v2.cpp -o v2.o
g++ -std=c++17 -O2 -Wall -pthread -c Fastext_20251116_v3.cpp -o v3.o
g++ -std=c++17 -O2 -Wall -pthread -c Fastext_20251116_v4.cpp -o v4.o
```

## Recommendation

- **Production use:** v1 (proven) or v4 (optimized)
- **New development:** v2 (modern C++) or v4 (flexible)
- **Debugging:** v3 (with n-gram output)
- **Research:** v4 (configurable parameters)
