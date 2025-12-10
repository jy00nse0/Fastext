// Fastext_20251116.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <algorithm>
#include <random>
#include <codecvt>
#include <locale>


#ifdef _MSC_VER
#include <windows.h>
#else
#include <unistd.h>
#endif

#define _CRT_SECURE_NO_WARNINGS
#include <time.h>
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <random> // ensure random included for distributions

//#include <math.h>  /* logf */


#define MAX_STRING 1000
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define VOCAB_HASH_SIZE 30000000
#define BOW "<"
#define EOW ">"

#define MAX_LINE_SIZE 1000

/* EPS: 로그 안정성 위해 작은 값으로 클램프 */
#ifndef LOSS_EPS
#define LOSS_EPS 1e-8f
#endif

char EOS[MAX_STRING] = "</s>";
const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

struct vocab_word {
    long long cn = 0; // word count
    int* point = nullptr;
    char* word = nullptr;
    char* code = nullptr;
    char codelen = 0;
    std::vector<int> subwords; // subword id 목록 (동적 크기)
    int type = 0; // 0 : word 1 : label
};

std::vector<vocab_word> vocab;
char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING], save_model_file[MAX_STRING];
int binary = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1; int label_count = 0;
int* vocab_hash; int minn = 3; int maxn = 6; int model_name = 2; int loss_name = 0; long long seed = 0;
long long vocab_max_size = 30000000, layer1_size = 100, bucket_size = 2000000; long long tokenCount_ = 0;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
float starting_alpha = 0.025;
float alpha;
float sample = 1e-4;
float* syn0, * syn1, * syn1neg, * expTable;

clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int* table;
bool normalizeGradient_ = false;



// Provide faster getc/unlocked where available
#ifndef _MSC_VER
#define GETC(f) getc_unlocked(f)
#define UNGETC(c, f) ungetc(c, f)
#else
#define GETC(f) fgetc(f)
#define UNGETC(c, f) ungetc(c, f)
#endif

// Forward declarations for helper functions
int define_size(const int* vec);
void addVectorToRow(float* syn0, const float* grad, int row, float a);
void incrementNExamples(struct model_State* state, float loss);
void computeHidden(const std::vector<int>& input, struct model_State* state);
void averageRowsToVector(float* hidden, const std::vector<int>& input); // <-- keep only one definition
void model_update(const std::vector<int>& input, const std::vector<int>& targets, int targetIndex, float alpha, struct model_State* state);
int GetHash(char* word);
void Cleanup();
void printProgress(long long currentTokens, long long totalTokens, int iterations, clock_t startTime, float currentAlpha);
void SaveModel(const char* filename);
void computeSubwords(const char* word, std::vector<int>& subwords);
void add_subwords(std::vector<int>& line, const char* token, int wid);
float loss_forward(const int* targets, int targetIndex, struct model_State* state, float current_alpha, bool isUpdate);
void initModelState(struct model_State* state, int hiddenSize, int outputSize); // <-- add declaration
void InitNet(); // <-- add declaration

struct model_State {
    float lossValue_;
    int nexamples_;
    float* hidden;
    float* output;
    float* grad;
    //unsigned long long rng;
    std::minstd_rand rng;
    // 생성자 정의
//set_State(int hiddenSize, int outputSize, int seed) : lossValue_(0.0f), nexamples_(0), hidden(hiddenSize),
///                output(outputSize), grad(hiddenSize), rng(seed) {}
    model_State(int hiddenSize, int outputSize, int seed)
        : lossValue_(0.0f),
        nexamples_(0),
        hidden(nullptr),
        output(nullptr),
        grad(nullptr),
        rng(seed) {
    }
};

int define_size(const int* vec) {
    int i = 0;
    while (i < 1000 && vec[i] != -1) {
        i++;
    }
    return i;
}
void InitUnigramTable() {
    size_t a, i;
    double train_words_pow = 0;
    double d1, power = 0.5;
    table = (int*)malloc(table_size * sizeof(int));
    if (!table) {
        fprintf(stderr, "InitUnigramTable: failed to allocate table\n");
        exit(1);
    }
    for (a = 0; a < vocab.size(); ++a) train_words_pow += pow((double)vocab[a].cn, power);
    i = 0;
    if (!vocab.empty()) d1 = pow((double)vocab[i].cn, power) / train_words_pow;
    else d1 = 0;
    for (size_t aa = 0; aa < (size_t)table_size; ++aa) {
        table[aa] = (int)i;
        if ((double)aa / (double)table_size > d1) {
            ++i;
            if (i >= vocab.size()) i = vocab.size() > 0 ? vocab.size() - 1 : 0;
            d1 += pow((double)vocab[i].cn, power) / train_words_pow;
        }
    }
}


// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
// UTF-8 안전한 ReadWord
void ReadWord(char* word, FILE* fin) {
    int a = 0;
    int ch;

    while (!feof(fin)) {
        ch = GETC(fin);
        if (ch == EOF) break;

        // 줄바꿈, 공백류 처리
        if (ch == ' ' || ch == '\n' || ch == '\r' ||
            ch == '\t' || ch == '\v' || ch == '\f' || ch == '\0') {
            if (a > 0) {
                if (ch == '\n') UNGETC(ch, fin);  // 문장 경계 처리
                break;
            }
            if (ch == '\n') {
#ifdef _MSC_VER
                strcpy_s(word, MAX_STRING, "</s>");
#else
                strncpy(word, "</s>", MAX_STRING);
                word[MAX_STRING - 1] = 0;
#endif
                return;
            }
            else {
                continue;
            }
        }

        // UTF-8 코드포인트 읽기
        unsigned char c = (unsigned char)ch;
        word[a++] = c;

        if (c >= 0xC0) { // multi-byte 시작 바이트
            int moreBytes = 0;
            if ((c & 0xE0) == 0xC0) moreBytes = 1;     // 2-byte char
            else if ((c & 0xF0) == 0xE0) moreBytes = 2; // 3-byte char
            else if ((c & 0xF8) == 0xF0) moreBytes = 3; // 4-byte char

            for (int i2 = 0; i2 < moreBytes; i2++) {
                int cc = GETC(fin);
                if (cc == EOF) break;
                if ((cc & 0xC0) != 0x80) { // continuation byte 아님
                    UNGETC(cc, fin);
                    break;
                }
                if (a < MAX_STRING - 1) {
                    word[a++] = (unsigned char)cc;
                }
            }
        }

        if (a >= MAX_STRING - 1) {  // 안전하게 truncate
            a = MAX_STRING - 2;
        }
    }
    word[a] = 0;
}

uint32_t FastTextHash(const char* word) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; word[i] != '\0'; i++) {
        h = h ^ uint32_t(int8_t(word[i]));
        h = h * 16777619u;
    }
    return h;
}

// 서브워드 버킷 인덱스 계산용
long long GetWordHash(char* word) {
    return (long long)(FastTextHash(word) % (uint32_t)bucket_size);
}

// vocab 해시 계산용 (vocab_hash_size는 30000000으로 정의됨)
int GetHash(char* word) {
    return (int)(FastTextHash(word) % (uint32_t)vocab_hash_size);
}


int getType(const char* w) {
    if (w && w[0] != '\0') {
        if (strncmp(w, "__label__", 9) == 0) {
            return 1;
        }
    }
    return 0;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char* word) {
    unsigned int hash = (unsigned int)GetHash(word);
    while (1) {
        if (vocab_hash[hash] == -1) return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE* fin) {
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin)) return -1;
    return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char* word) {
    if (!word) return -1;

    // 메모리 재할당 검사 대신 벡터 크기 확인
    if ((size_t)vocab.size() + 2 >= (size_t)vocab_max_size) {
        vocab_max_size += 3000000000LL;
        vocab.reserve((size_t)vocab_max_size);
    }

    vocab_word new_word;
    int length = (int)strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;

    // 단어 메모리 할당
    new_word.word = (char*)calloc(length, sizeof(char));
    if (!new_word.word) return -1;
#ifdef _MSC_VER
    strcpy_s(new_word.word, length, word);
#else
    // use memcpy to avoid strncpy related warning and ensure null-termination
    memcpy(new_word.word, word, (size_t)length);
    new_word.word[length - 1] = 0;
#endif

    // 기존과 동일한 초기화 유지
    new_word.cn = 0;
    new_word.type = getType(word);

    // 중요: 포인터 멤버들 명시적 초기화
    new_word.point = NULL;      // 누락 보완
    new_word.code = NULL;       // 누락 보완
    new_word.codelen = 0;       // 누락 보완

    // subwords 초기화 - 올바른 인덱스 사용
    new_word.subwords.clear();
    new_word.subwords.reserve(20); // 평균 서브워드 개수
    int new_index = (int)vocab.size(); // push_back 전에 인덱스 계산
    new_word.subwords.push_back(new_index); // 자기 자신 ID 추가

    // 해시 테이블에 추가
    int hash = GetHash(word);
    while (vocab_hash[hash] != -1) {
        hash = (hash + 1) % vocab_hash_size;
    }
    vocab_hash[hash] = new_index;

    // 벡터에 추가
    vocab.push_back(new_word);

    return new_index;
}

// Used later for sorting by word counts
//int VocabCompare(const void* a, const void* b) {
//    return ((struct vocab_word*)b)->cn - ((struct vocab_word*)a)->cn;
//}

bool VocabCompare(const vocab_word& a, const vocab_word& b) {
    return a.cn > b.cn; // 내림차순 정렬
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
    size_t a, b;
    size_t vocab_sz = vocab.size();
    if (vocab_sz == 0) return;

    long long* count = (long long*)calloc(vocab_sz * 2 + 1, sizeof(long long));
    long long* binary = (long long*)calloc(vocab_sz * 2 + 1, sizeof(long long));
    long long* parent_node = (long long*)calloc(vocab_sz * 2 + 1, sizeof(long long));
    if (!count || !binary || !parent_node) {
        fprintf(stderr, "CreateBinaryTree: memory allocation failed\n");
        exit(1);
    }
    for (a = 0; a < vocab_sz; a++) count[a] = vocab[a].cn;
    for (a = vocab_sz; a < vocab_sz * 2; a++) count[a] = (long long)1e15;
    long long pos1 = (long long)vocab_sz - 1;
    long long pos2 = (long long)vocab_sz;
    long long min1i, min2i;
    // Following algorithm constructs the Huffman tree by adding one node at a time
    for (long long aa = 0; aa < (long long)vocab_sz - 1; aa++) {
        // First, find two smallest nodes 'min1, min2'
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min1i = pos1;
                pos1--;
            }
            else {
                min1i = pos2;
                pos2++;
            }
        }
        else {
            min1i = pos2;
            pos2++;
        }
        if (pos1 >= 0) {
            if (count[pos1] < count[pos2]) {
                min2i = pos1;
                pos1--;
            }
            else {
                min2i = pos2;
                pos2++;
            }
        }
        else {
            min2i = pos2;
            pos2++;
        }
        count[vocab_sz + aa] = count[min1i] + count[min2i];
        parent_node[min1i] = vocab_sz + aa;
        parent_node[min2i] = vocab_sz + aa;
        binary[min2i] = 1;
    }
    // Now assign binary code to each vocabulary word
    for (size_t aa = 0; aa < vocab_sz; aa++) {
        long long bidx = (long long)aa;
        int i = 0;
        // Allocate memory for point and code if not already
        if (!vocab[aa].point) vocab[aa].point = (int*)calloc(MAX_CODE_LENGTH, sizeof(int));
        if (!vocab[aa].code) vocab[aa].code = (char*)calloc(MAX_CODE_LENGTH, sizeof(char));
        long long bj = bidx;
        char code[MAX_CODE_LENGTH];
        long long point[MAX_CODE_LENGTH];
        while (1) {
            code[i] = (char)binary[bj];
            point[i] = bj;
            i++;
            bj = parent_node[bj];
            if (bj == (long long)vocab_sz * 2 - 2) break;
            if (i >= MAX_CODE_LENGTH - 1) break;
        }
        vocab[aa].codelen = (char)i;
        if (vocab[aa].point) vocab[aa].point[0] = (int)(vocab_sz - 2);
        for (int bb = 0; bb < i; bb++) {
            if (vocab[aa].code) vocab[aa].code[i - bb - 1] = code[bb];
            if (vocab[aa].point) vocab[aa].point[i - bb] = (int)(point[bb] - vocab_sz);
        }
    }
    free(count);
    free(binary);
    free(parent_node);
}

void initNgrams() {
    for (size_t i = 0; i < vocab.size(); i++) {
        char word_buf[MAX_STRING * 2];
        vocab[i].subwords.clear();      // 초기화
        vocab[i].subwords.reserve(50);  // 평균 서브워드 개수
        vocab[i].subwords.push_back((int)i); // 자기 자신 id 추가

        if (strcmp(vocab[i].word, "</s>") != 0) {   // word가 char*일 경우
            snprintf(word_buf, sizeof(word_buf), "<%s>", vocab[i].word);
            computeSubwords(word_buf, vocab[i].subwords);
        }
    }
}


void LearnVocabFromTrainFile() {
    char word[MAX_STRING];
    size_t a;
    for (a = 0; a < (size_t)vocab_hash_size; a++) {
        vocab_hash[a] = -1;
    }

    FILE* fin;
#ifdef _MSC_VER
    fopen_s(&fin, train_file, "rb");
#else
    fin = fopen(train_file, "rb");
#endif
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    AddWordToVocab((char*)"</s>");
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) break;
        train_words++;
        if ((debug_mode > 1) && (train_words % 100000 == 0)) {
            //printf("%lldK%c", train_words / 1000, '\r');
            fflush(stdout);
        }
        int i = SearchVocab(word);
        if (i == -1) {
            a = (size_t)AddWordToVocab(word);
            if ((size_t)a < vocab.size()) {
                vocab[a].cn = 1;
                vocab[a].type = getType(word);
            }
        }
        else vocab[i].cn++;
    }
    // === 여기까지 ===
    printf("Vocab size: %zu\n", vocab.size());
    printf("Words in train file: %lld\n", train_words);
    // === min_count 미만 단어 제거 및 vocab_hash 재구성 ===
    // 1. 빈도순 내림차순 정렬
    std::sort(vocab.begin(), vocab.end(), VocabCompare);

    // 2. min_count 미만 단어 제거
    auto new_end = std::remove_if(vocab.begin(), vocab.end(), 
     [](const vocab_word& w) { return w.cn < min_count; });
    vocab.erase(new_end, vocab.end());

    // ✅ 2.5. train_words 재계산 (min_count 이상 단어만 합산)
    train_words = 0;
    for (size_t i = 0; i < vocab.size(); i++) {
        train_words += vocab[i].cn;
    }

    // 3. vocab_hash 재구성
    for (a = 0; a < (size_t)vocab_hash_size; a++) vocab_hash[a] = -1;
    for (size_t i = 0; i < vocab.size(); i++) {
        int hash = GetHash(vocab[i].word);
        while (vocab_hash[hash] != -1) {
            hash = (hash + 1) % vocab_hash_size;
        }
        vocab_hash[hash] = (int)i;
    }

    // (선택) 로그 출력
    printf("After min_count filtering: vocab size = %zu, train_words = %lld\n",
        vocab.size(), train_words);

    initNgrams();
    file_size = ftell(fin);
    fclose(fin);
}

void SaveVocab() {
    size_t i;
    FILE* fo;
#ifdef _MSC_VER
    fopen_s(&fo, save_vocab_file, "wb");
#else
    fo = fopen(save_vocab_file, "wb");
#endif
    for (i = 0; i < vocab.size(); i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
    fclose(fo);
}
void computeSubwords(const char* word, std::vector<int>& subwords) {
    char buf[MAX_STRING * 2];
    snprintf(buf, sizeof(buf), "%s", word);
    int buflen = (int)strlen(buf);
    if (buflen == 0) return;
    
    // Build a list of UTF-8 character start positions
    std::vector<int> char_positions;
    char_positions.push_back(0);
    for (int i = 0; i < buflen; i++) {
        // Check if this is the start of a UTF-8 character (not a continuation byte)
        if ((buf[i] & 0xC0) != 0x80) {
            if (i > 0) {
                char_positions.push_back(i);
            }
        }
    }
    char_positions.push_back(buflen); // End position
    
    int num_chars = (int)char_positions.size() - 1;
    
    // Generate n-grams based on UTF-8 character boundaries
    for (int i = 0; i < num_chars; i++) {
        for (int n = minn; n <= maxn; n++) {
            if (i + n > num_chars) break;
            
            // Extract n-gram from character position i to i+n
            int start_pos = char_positions[i];
            int end_pos = char_positions[i + n];
            int ngram_byte_len = end_pos - start_pos;
            
            if (ngram_byte_len > 0 && ngram_byte_len < MAX_STRING) {
                char ngram[MAX_STRING] = { 0 };
                memcpy(ngram, &buf[start_pos], ngram_byte_len);
                ngram[ngram_byte_len] = '\0';
                
                // Skip n-grams at the very beginning or end if they match boundaries
                if (!(start_pos == 0 || end_pos == buflen)) {
                    int h = (int)(GetWordHash(ngram) % bucket_size);
                    int idx = (int)vocab.size() + h;
                    subwords.push_back(idx);
                }
            }
        }
    }
}



// 간단한 랜덤 생성기 (C표준 rand() 사용)
double uniform_random(void) {
    return (double)rand() / RAND_MAX;
}

void reset_file(FILE* fin) {
    if (feof(fin)) {        // 파일 끝에 도달했을 때만
        clearerr(fin);
        fseek(fin, 0, SEEK_SET);  // 파일 처음으로 되돌림
    }
    // 파일 끝이 아니면 아무것도 하지 않음 (계속 순차 읽기)
}

int get_line(FILE* fin, std::vector<int>& words) {
    char token[MAX_STRING];
    int ntokens = 0;
    int word_count = 0;
    words.clear();
    reset_file(fin);
    ReadWord(token, fin);
    while (token[0] != '\0') {
        int wid = SearchVocab(token);
        if (wid < 0) {
            ReadWord(token, fin);
            continue;
        }
        // === [추가: 빈도 기반 다운샘플링 적용] ===
        if (sample > 0) {
            float ran = (sqrt((double)vocab[wid].cn / (sample * train_words)) + 1) *
                (sample * train_words) / vocab[wid].cn;
            if (ran < uniform_random()) {
                ReadWord(token, fin);
                continue;  // skip this high-frequency word
            }
        }
        // === [끝] ===

        ntokens++;
        if (getType(vocab[wid].word) == 0) {
            word_count++;
            //if (word_count < (int)words.size())
            words.push_back(wid);
        }
        if (ntokens > MAX_LINE_SIZE || strcmp(token, EOS) == 0) {
            break;
        }
        ReadWord(token, fin);
    }
    return ntokens;
}

int get_line_sub(FILE* fin, std::vector<int>& words, std::vector<int>& labels) {
    char token[MAX_STRING];
    int ntokens = 0;
    reset_file(fin);
    words.clear();
    labels.clear();
    ReadWord(token, fin);
    while (token[0] != '\0') {
        int wid = SearchVocab(token);
        int type = wid < 0 ? getType(token) : getType(vocab[wid].word);

        ntokens++;

        // === [추가: 빈도 기반 다운샘플링 적용] ===
        if (type == 0 && wid >= 0 && sample > 0) {
            float ran = (sqrt((double)vocab[wid].cn / (sample * train_words)) + 1) *
                (sample * train_words) / vocab[wid].cn;
            if (ran < uniform_random()) {
                ReadWord(token, fin);
                continue;  // skip high-frequency word
            }
        }
        if (type == 0) {
            add_subwords(words, token, wid);
        }
        else if (type == 1 && wid >= 0) {
            labels.push_back(wid - (int)vocab.size());
        }
        if (strcmp(token, EOS) == 0) {
            break;
        }
        ReadWord(token, fin);
    }
    return ntokens;
}

void averageRowsToVector(float* hidden, const std::vector<int>& input) {
    int n = (int)input.size();
    if (n == 0) return;
    memset(hidden, 0, (size_t)layer1_size * sizeof(float));
    for (int i = 0; i < n; i++) {
        int row = input[i];
        float* src = &syn0[(size_t)row * (size_t)layer1_size];
        float* dst = hidden;
        for (int j = 0; j < layer1_size; ++j) {
            dst[j] += src[j];
        }
    }
    float inv_n = 1.0f / n;
    for (int j = 0; j < layer1_size; j++) {
        hidden[j] *= inv_n;
    }
}

float model_State_getLoss(const struct model_State* state) {
    return state->lossValue_;
}

void model_State_incrementNExamples(struct model_State* state, float loss) {
    state->lossValue_ += loss;
    state->nexamples_++;
}

double getDuration(clock_t start, clock_t end) {
    //printf("CLOCKS_PER_SEC = %ld\n", CLOCKS_PER_SEC);

    return (double)(end - start) / CLOCKS_PER_SEC; // Fixed the return statement
}

void cbow(struct model_State* state, float alpha, const std::vector<int>& line)
{
    int line_size = (int)line.size();
    int w;
    std::vector<int> bow;
    std::uniform_int_distribution<> uniform(1, window);
    bow.reserve(window * 2 * 5); // 윈도우 크기 * 양방향 * 평균 서브워드 수
    for (w = 0; w < line_size; w++) {
        int boundary = uniform(state->rng);
        bow.clear();
        for (int c = -boundary; c <= boundary; c++) {
            if (c != 0 && w + c >= 0 && w + c < line_size) {
                int idx = line[w + c];
                const std::vector<int>& subwords = vocab[idx].subwords;
                for (size_t k = 0; k < subwords.size(); k++) {
                    bow.push_back(subwords[k]);
                }
            }
        }
        model_update(bow, line, w, alpha, state);
    }
}


void skipgram(struct model_State* state, float alpha, const std::vector<int>& line) {
    std::uniform_int_distribution<> uniform(1, window);
    int line_size = (int)line.size();
    if (debug_mode > 1) printf("totaltoskip %d \n", line_size);
    for (int w = 0; w < line_size; w++) {
        int boundary = uniform(state->rng);
        // Changed to reference to avoid copying the subwords vector on every iteration
        const std::vector<int>& ngrams = vocab[line[w]].subwords;
        for (int c = -boundary; c <= boundary; c++) {
            if (c != 0 && w + c >= 0 && w + c < line_size) {
                if (debug_mode > 1) printf("model_update %lld \n", train_words);
                model_update(ngrams, line, w + c, alpha, state);
            }
        }
    }
}

void supervised(
    struct model_State* state,
    float alpha,
    const std::vector<int>& line,
    const std::vector<int>& labels) {
    int labels_size = (int)labels.size();
    int line_size = (int)line.size();

    if (labels_size == 0 || line_size == 0) {
        return;
    }
    if (loss_name == 3) {
        model_update(line, labels, -1, alpha, state); // -1 for all labels as target
    }
    else {
        // move distribution creation out of inner code so it's not re-created for no reason
        static thread_local std::uniform_int_distribution<> uniform(1, window);

        int i = uniform(state->rng) % labels_size;
        model_update(line, labels, i, alpha, state);
    }
}

// Fix model_update definition
void model_update(
    const std::vector<int>& input,
    const std::vector<int>& targets,
    int targetIndex,
    float alpha,
    struct model_State* state) {
    if (debug_mode > 1) printf("model_updateing %lld \n", train_words);
    // 디버깅: 입력 데이터 확인
    if (debug_mode > 1) printf("[model_update] input_size=%zu, targets_size=%zu, targetIndex=%d\n",
        input.size(), targets.size(), targetIndex);

    if (input.empty()) {
        if (debug_mode > 1)printf("Empty input, skipping update.\n");
        return;
    }
    if (!targets.empty() && targetIndex >= 0 && targetIndex < (int)targets.size()) {
        if (debug_mode > 1) printf("  target word index=%d\n", targets[targetIndex]);
    }
    computeHidden(input, state);

    //  Gradient 변화 모니터링
    for (int i = 0; i < layer1_size; i++) {
        state->grad[i] = 0.0f;
    }
    float lossValue = loss_forward(targets.data(), targetIndex, state, alpha, true);
    incrementNExamples(state, lossValue);


    int inputSize = (int)input.size();
    if (normalizeGradient_) {
        for (int i = 0; i < layer1_size; i++) {
            state->grad[i] *= (1.0f / (float)inputSize);
        }
    }

    // 가중치 업데이트
    for (int i = 0; i < inputSize; i++) {
        int row = input[i];
        for (int j = 0; j < layer1_size; j++) {
            syn0[(size_t)row * (size_t)layer1_size + j] += state->grad[j];
        }
    }
}

float getLoss(struct model_State* state) {
    if (state->nexamples_ == 0) {
        return 0.0f;
    }
    return state->lossValue_ / state->nexamples_;
}

void incrementNExamples(struct model_State* state, float loss) {
    state->lossValue_ += loss;
    state->nexamples_++;
}

void computeHidden(const std::vector<int>& input, struct model_State* state) {
    float norm = 0.0f;
    for (int i = 0; i < layer1_size; i++) {
        norm += state->hidden[i] * state->hidden[i];
    }
    norm = sqrt(norm);
    if (debug_mode > 1) printf("[computeHidden] hidden norm=%.6f\n", norm);
    averageRowsToVector(state->hidden, input);
}


float loss_forward(const int* targets, int targetIndex, struct model_State* state, float alpha, bool isUpdate) {
    float loss = 0.0f;
    int c, d, l2;
    float f, g;
    if (targetIndex == -1) {
        // 전체 타겟에 대한 손실 계산 필요
        // 현재 구현에서는 이 경우를 처리하지 못함
        printf("ERROR: targetIndex=-1 not implemented\n");
        return 0.0f;
    }
    int word = targets[targetIndex];
    const float EPS = LOSS_EPS;
    if (debug_mode > 1) printf("loss forward %lld \n", train_words);
    if (debug_mode > 1) printf("[loss_forward] targetIndex=%d, word=%d\n", targetIndex, word);
    // 디버깅: 기본 정보 출력
    if (debug_mode > 1) printf("[loss_forward] word=%d, hidden[0]=%.6f, alpha=%.6f\n",
        word, state->hidden[0], alpha);

    if (loss_name == 1) { /* hierarchical softmax */
        for (d = 0; d < (int)vocab[word].codelen; d++) {
            l2 = vocab[word].point[d] * layer1_size;
            f = 0.0f;
            for (c = 0; c < layer1_size; c++) f += state->hidden[c] * syn1[c + l2];

            /* sigmoid approximated by expTable where possible; handle extremes */
            float p;
            if (f > MAX_EXP) {
                p = 1.0f;
            }
            else if (f < -MAX_EXP) {
                p = 0.0f;
            }
            else {
                int idx = (int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2));
                p = expTable[idx];
            }
            // 디버깅: 중간 값 출력
            if (debug_mode > 1) printf("[HS] f=%.6f, p=%.6f, code[%d]=%d\n", f, p, d, vocab[word].code[d]);
            /* 안전한 로그 계산을 위해 클램프 */
            float p_clamped = p;
            if (p_clamped < EPS) p_clamped = EPS;
            else if (p_clamped > 1.0f - EPS) p_clamped = 1.0f - EPS;

            /* 손실 누적: code==1 -> -log(p), code==0 -> -log(1-p) */
            if (vocab[word].code[d]) {
                loss += -logf(p_clamped);
            }
            else {
                loss += -logf(1.0f - p_clamped);
            }

            /* gradient 계산 및 반영 (원래 방식 유지, p 사용) */
            g = (1 - vocab[word].code[d] - p) * alpha;
            for (c = 0; c < layer1_size; c++) state->grad[c] += g * syn1[c + l2];
            if (isUpdate) {
                for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * state->hidden[c];
            }
        }
    }

    if (loss_name == 0) { /* negative sampling */
        // reuse distribution per-thread (avoid re-constructing per call)
        static thread_local std::uniform_int_distribution<size_t> uniform_dist(0, table_size - 1);

        for (d = 0; d < negative + 1; d++) {
            int target;
            int label;
            if (d == 0) {
                target = word;
                label = 1;
            }
            else {
                do {
                    target = table[uniform_dist(state->rng)];
                } while (target == word);
                label = 0;
            }
            l2 = target * layer1_size;
            f = 0.0f;
            for (c = 0; c < layer1_size; c++) f += state->hidden[c] * syn1neg[c + l2];
            if (debug_mode > 1) printf("[loss_forward] d=%d, target=%d, label=%d, f=%.6f\n", d, target, label, f);
            /* sigmoid (p) 계산 - 같은 방식으로 처리 */
            float p;
            if (f > MAX_EXP) {
                p = 1.0f;
            }
            else if (f < -MAX_EXP) {
                p = 0.0f;
            }
            else {
                int idx = (int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2));
                p = expTable[idx];
            }

            // 디버깅: NS 중간 값 출력
            if (debug_mode > 1) printf("[NS] d=%d, target=%d, label=%d, f=%.6f, p=%.6f\n",
                d, target, label, f, p);

            /* 클램프 후 손실 누적 */
            float p_clamped = p;
            if (p_clamped < EPS) p_clamped = EPS;
            else if (p_clamped > 1.0f - EPS) p_clamped = 1.0f - EPS;
            float term_loss = 0.0f;
            if (label) term_loss += -logf(p_clamped);
            else term_loss += -logf(1.0f - p_clamped);
            loss += term_loss;

            if (debug_mode > 1) printf("[loss_forward] term_loss=%.6f, cumulative_loss=%.6f\n", term_loss, loss);
            // 디버깅: 손실 항 출력
            if (debug_mode > 1) printf("[NS] term_loss=%.6f, total_loss=%.6f\n", term_loss, loss);
            /* gradient (원래 로직 유지) */
            if (f > MAX_EXP) g = (label - 1) * alpha;
            else if (f < -MAX_EXP) g = (label - 0) * alpha;
            else g = (label - p) * alpha;

            for (c = 0; c < layer1_size; c++) state->grad[c] += g * syn1neg[c + l2];
            if (isUpdate) {
                for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * state->hidden[c];
            }
            // 디버깅: 최종 손실 출력
            if (debug_mode > 1) printf("[loss_forward] final_loss=%.6f\n", loss);
        }
    }
    return loss;
}



int ArgPos(char* str, int argc, char** argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}
// 벡터 저장

int keepTraining(const long long totaltokens, long long tokenCount_) {  // 또는 bool keepTraining
    if (tokenCount_ < (long long)iter * totaltokens) {
        //printf("Words in train file: %lld\n", train_words);
        return 1;
    }
    else {
        return 0;
    }

}
void SaveVectors() {
    size_t a, b;

#ifdef _MSC_VER
    FILE* fo = nullptr;
    if (fopen_s(&fo, output_file, "wb") != 0 || fo == nullptr) {
        // handle error
    }    if (fo == NULL) {
        printf("Error: Cannot open output file %s\n", output_file);
        return;
    }
    fprintf(fo, "%zu %lld\n", vocab.size(), layer1_size);
#else
    FILE* fo = fopen(output_file, "wb");
    if (fo == NULL) {
        printf("Error: Cannot open output file %s\n", output_file);
        return;
    }
    fprintf(fo, "%zu %lld\n", vocab.size(), layer1_size);
#endif
    float* vec = (float*)malloc((size_t)layer1_size * sizeof(float));
    if (!vec) {
        printf("Error: Cannot allocate memory for vector\n");
        fclose(fo);
        return;
    }
    for (a = 0; a < vocab.size(); a++) {
        memset(vec, 0, (size_t)layer1_size * sizeof(float));
        int subword_count = 0;
        for (size_t i = 0; i < vocab[a].subwords.size(); i++) {
            size_t subword_id = (size_t)vocab[a].subwords[i];
            for (b = 0; b < (size_t)layer1_size; b++) {
                vec[b] += syn0[subword_id * (size_t)layer1_size + b];
            }
            subword_count++;
        }
        if (subword_count > 0) {
            for (b = 0; b < (size_t)layer1_size; b++) {
                vec[b] /= subword_count;
            }
        }
        fprintf(fo, "%s", vocab[a].word);
        for (b = 0; b < (size_t)layer1_size; b++) {
            fprintf(fo, " %.6f", vec[b]);
        }
        fprintf(fo, "\n");
    }
    free(vec);
    fclose(fo);
    printf("\nFastText-compatible vectors saved to %s\n", output_file);
}

// 진행 상황 및 ETA 출력 함수
void printProgress(long long currentTokens, long long totalTokens, int iterations,
    clock_t startTime, float currentAlpha) {
    float progress = (float)currentTokens / (iterations * totalTokens);
    clock_t now = clock();
    double elapsed = (double)(now - startTime) / CLOCKS_PER_SEC;
    double total_estimated = (progress > 0) ? (elapsed / progress) : 0;
    double eta_seconds = (progress > 0) ? (total_estimated - elapsed) : 0;

    // words per second 계산
    double words_per_sec = (elapsed > 0) ? currentTokens / elapsed : 0;

    printf("[Progress] %.2f%%, Tokens: %lld/%lld, Elapsed: %.1f min, ETA: %.1f min, Alpha: %.6f, WPS: %.0f\n",
        progress * 100.0,
        currentTokens,
        iterations * totalTokens,
        elapsed / 60.0,
        eta_seconds / 60.0,
        currentAlpha,
        words_per_sec);

    fflush(stdout);
}
// TrainModelThread: 학습률 스케줄링 개선, tokenCount_ 일반 변수 사용
void* TrainModelThread(void* arg) {
    int threadId = (int)(intptr_t)arg;

    FILE* fi;
#ifdef _MSC_VER
    fopen_s(&fi, train_file, "rb");
#else
    fi = fopen(train_file, "rb");
#endif
    if (!fi) {
        fprintf(stderr, "TrainModelThread: failed to open train file\n");
        pthread_exit(NULL);
        return NULL;
    }
    fseek(fi, (long)(file_size / num_threads * threadId), SEEK_SET);
    long long localTokenCount = 0;
    struct model_State state((int)layer1_size, (int)layer1_size, threadId + (int)seed);

    initModelState(&state, (int)layer1_size, (int)layer1_size);
    long long totaltokens = train_words;
    if (debug_mode > 1) printf("totaltokens %lld \n", totaltokens);
    std::vector<int> line;
    std::vector<int> labels;
    line.reserve(MAX_SENTENCE_LENGTH);
    labels.reserve(10); // 일반적으로 라벨은 적은 수
    float progress_min = 0.01f;
    while (keepTraining(totaltokens, tokenCount_) == 1) {
        float progress = (float)(tokenCount_) / (iter * totaltokens);
        float alpha = starting_alpha * (1.0f - progress);
        alpha = (alpha > starting_alpha * 0.0001f) ? alpha : starting_alpha * 0.0001f;
        if (threadId == 0) {
            if (progress > progress_min) {
                printProgress(tokenCount_, totaltokens, iter, start, alpha);
                progress_min += 0.01f;
            }
        }

        if (model_name == 2) {
            localTokenCount += get_line_sub(fi, line, labels);
            supervised(&state, alpha, line, labels);
        }
        else if (model_name == 0) {
            localTokenCount += get_line(fi, line);
            cbow(&state, alpha, line);
        }
        else if (model_name == 1) {
            localTokenCount += get_line(fi, line);
            if (debug_mode > 1) printf("localTokenCount %lld \nz", localTokenCount);
            skipgram(&state, alpha, line);
        }
        // Increase batch size for tokenCount_ update
        if (localTokenCount > 10000) {
            tokenCount_ += localTokenCount;
            localTokenCount = 0;
        }
    }
    fclose(fi);
    pthread_exit(NULL);
    return NULL;
}

void TrainModel() {
    alpha = starting_alpha;
    LearnVocabFromTrainFile();
    if (save_vocab_file[0] != 0) SaveVocab();
    if (output_file[0] == 0) return;
    InitNet();
    if (negative > 0) InitUnigramTable();
    start = clock();

    pthread_t* pt = (pthread_t*)malloc((size_t)num_threads * sizeof(pthread_t));
    for (int a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void*)(intptr_t)a);
    for (int a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    free(pt);
    SaveVectors();
    if (save_model_file[0] != 0) SaveModel(save_model_file);
    Cleanup();
}

// 모델 저장
void SaveModel(const char* filename) {
    FILE* fo;
#ifdef _MSC_VER
    fopen_s(&fo, filename, "wb");
#else
    fo = fopen(filename, "wb");
#endif
    if (fo == NULL) {
        printf("Error: Cannot open model file %s\n", filename);
        return;
    }

    // 1. 모든 알규먼트 저장
     // 1. Save all argument variables (hyperparameters and settings)
    fwrite(&binary, sizeof(int), 1, fo);
    fwrite(&debug_mode, sizeof(int), 1, fo);
    fwrite(&window, sizeof(int), 1, fo);
    fwrite(&min_count, sizeof(int), 1, fo);
    fwrite(&num_threads, sizeof(int), 1, fo);
    fwrite(&min_reduce, sizeof(int), 1, fo);
    fwrite(&label_count, sizeof(int), 1, fo);
    fwrite(&minn, sizeof(int), 1, fo);
    fwrite(&maxn, sizeof(int), 1, fo);
    fwrite(&model_name, sizeof(int), 1, fo);
    fwrite(&loss_name, sizeof(int), 1, fo);
    fwrite(&seed, sizeof(int), 1, fo);
    fwrite(&vocab_max_size, sizeof(long long), 1, fo);
    fwrite(&layer1_size, sizeof(long long), 1, fo);
    fwrite(&bucket_size, sizeof(long long), 1, fo);
    fwrite(&tokenCount_, sizeof(long long), 1, fo);
    fwrite(&train_words, sizeof(long long), 1, fo);
    fwrite(&word_count_actual, sizeof(long long), 1, fo);
    fwrite(&iter, sizeof(long long), 1, fo);
    fwrite(&file_size, sizeof(long long), 1, fo);
    fwrite(&classes, sizeof(long long), 1, fo);
    fwrite(&starting_alpha, sizeof(float), 1, fo);
    fwrite(&alpha, sizeof(float), 1, fo);
    fwrite(&sample, sizeof(float), 1, fo);
    fwrite(&hs, sizeof(int), 1, fo);
    fwrite(&negative, sizeof(int), 1, fo);
    fwrite(&normalizeGradient_, sizeof(bool), 1, fo);

    // 2. vocab 저장
    for (size_t i = 0; i < vocab.size(); i++) {
        int len = (int)strlen(vocab[i].word) + 1;
        fwrite(&len, sizeof(int), 1, fo);
        fwrite(vocab[i].word, sizeof(char), len, fo);
        fwrite(&vocab[i].cn, sizeof(long long), 1, fo);
        fwrite(&vocab[i].type, sizeof(int), 1, fo);
        int subword_size = (int)vocab[i].subwords.size();
        fwrite(&subword_size, sizeof(int), 1, fo);
        fwrite(vocab[i].subwords.data(), sizeof(int), subword_size, fo);
    }

    // 3. word vectors 저장 (syn0)
    fwrite(syn0, sizeof(float), (size_t)(vocab.size() + (size_t)bucket_size) * (size_t)layer1_size, fo);

    // (필요시 syn1, syn1neg도 저장)
    if (loss_name == 1 && syn1) {
        fwrite(syn1, sizeof(float), (size_t)vocab.size() * (size_t)layer1_size, fo);
    }
    if (negative > 0 && syn1neg) {
        fwrite(syn1neg, sizeof(float), (size_t)vocab.size() * (size_t)layer1_size, fo);
    }

    fclose(fo);
    printf("Model saved to %s\n", filename);
}

void Cleanup() {
    // 개별 word 메모리 해제
    for (auto& word : vocab) {
        free(word.word);
        // point, code가 할당되었다면 여기서도 해제
        free(word.point);
        free(word.code);
        // subwords는 std::vector이므로 자동 해제
    }

    // 벡터 정리
    std::vector<vocab_word>().swap(vocab); // 완전히 메모리 해제

    // 다른 메모리 해제...
    free(vocab_hash);
    free(syn0);
    free(syn1);
    free(syn1neg);
    free(expTable);
    free(table);
}

// Move add_subwords definition above get_line_sub
void add_subwords(std::vector<int>& line, const char* token, int wid) {
    static thread_local char concat_buf[MAX_STRING * 4];
    if (wid < 0) {
        if (strcmp(token, EOS) != 0) {
            size_t bow_len = strlen(BOW);
            size_t token_len = strlen(token);
            size_t eow_len = strlen(EOW);
            size_t total_len = bow_len + token_len + eow_len + 1;
            if (total_len <= sizeof(concat_buf)) {
                char* concat = concat_buf;
#ifdef _MSC_VER
                strcpy_s(concat, sizeof(concat_buf), BOW);
                strcat_s(concat, sizeof(concat_buf), token);
                strcat_s(concat, sizeof(concat_buf), EOW);
#else
                strcpy(concat, BOW);
                strcat(concat, token);
                strcat(concat, EOW);
#endif
                int concat_idx = SearchVocab(concat);
                if (concat_idx >= 0) {
                    for (size_t i = 0; i < vocab[concat_idx].subwords.size(); i++) {
                        line.push_back(vocab[concat_idx].subwords[i]);
                    }
                } else {
                    int new_idx = AddWordToVocab(concat);
                    if (new_idx >= 0) {
                        computeSubwords(concat, vocab[new_idx].subwords);
                        for (size_t i = 0; i < vocab[new_idx].subwords.size(); i++) {
                            line.push_back(vocab[new_idx].subwords[i]);
                        }
                    }
                }
            } else {
                if (debug_mode > 0) {
                    printf("[WARN] Token too long for buffer: %s\n", token);
                }
            }
        }
    } else {
        if (maxn <= 0) {
            line.push_back(wid);
        } else {
            for (size_t i = 0; i < vocab[wid].subwords.size(); i++) {
                line.push_back(vocab[wid].subwords[i]);
            }
        }
    }
}

void initModelState(struct model_State* state, int hiddenSize, int outputSize) {
#ifdef _MSC_VER
    state->hidden = (float*)_aligned_malloc(hiddenSize * sizeof(float), 64);
    state->output = (float*)_aligned_malloc(outputSize * sizeof(float), 64);
    state->grad = (float*)_aligned_malloc(hiddenSize * sizeof(float), 64);
#else
    int ret1 = posix_memalign((void**)&state->hidden, 64, (size_t)hiddenSize * sizeof(float));
    int ret2 = posix_memalign((void**)&state->output, 64, (size_t)outputSize * sizeof(float));
    int ret3 = posix_memalign((void**)&state->grad, 64, (size_t)hiddenSize * sizeof(float));
    if (ret1 != 0) state->hidden = NULL;
    if (ret2 != 0) state->output = NULL;
    if (ret3 != 0) state->grad = NULL;
#endif
    if (state->hidden) memset(state->hidden, 0, (size_t)hiddenSize * sizeof(float));
    if (state->output) memset(state->output, 0, (size_t)outputSize * sizeof(float));
    if (state->grad) memset(state->grad, 0, (size_t)hiddenSize * sizeof(float));
    if (!state->hidden || !state->output || !state->grad) {
        printf("Aligned memory allocation failed\n");
    }
}

void InitNet() {
    size_t a, b;
    std::minstd_rand rng((unsigned)seed);
#ifdef _MSC_VER
    syn0 = (float*)_aligned_malloc(((size_t)vocab.size() + (size_t)bucket_size) * (size_t)layer1_size * sizeof(float), 128);
#else
    int ret = posix_memalign((void**)&syn0, 64, ((size_t)vocab.size() + (size_t)bucket_size) * (size_t)layer1_size * sizeof(float));
    if (ret != 0) syn0 = NULL;
#endif
    if (syn0 == NULL) { printf("Memory allocation failed\n"); exit(1); }
    if (loss_name == 1) {
#ifdef _MSC_VER
        syn1 = (float*)_aligned_malloc((size_t)vocab.size() * (size_t)layer1_size * sizeof(float), 128);
#else
        int ret2 = posix_memalign((void**)&syn1, 64, (size_t)vocab.size() * (size_t)layer1_size * sizeof(float));
        if (ret2 != 0) syn1 = NULL;
#endif
        if (syn1 == NULL) { printf("Memory allocation failed\n"); exit(1); }
        for (a = 0; a < vocab.size(); a++) for (b = 0; b < (size_t)layer1_size; b++)
            syn1[a * (size_t)layer1_size + b] = 0;
        CreateBinaryTree();
    }
    if (negative > 0) {
#ifdef _MSC_VER
        syn1neg = (float*)_aligned_malloc((size_t)vocab.size() * (size_t)layer1_size * sizeof(float), 128);
#else
        int ret3 = posix_memalign((void**)&syn1neg, 64, (size_t)vocab.size() * (size_t)layer1_size * sizeof(float));
        if (ret3 != 0) syn1neg = NULL;
#endif
        if (syn1neg == NULL) { printf("Memory allocation failed\n"); exit(1); }
        for (a = 0; a < vocab.size(); a++) for (b = 0; b < (size_t)layer1_size; b++)
            syn1neg[a * (size_t)layer1_size + b] = 0;
    }
    float uni = 1.0f / static_cast<float>(layer1_size);
    std::uniform_real_distribution<float> uniform_dist_w_in(-uni, uni);
    for (a = 0; a < vocab.size(); a++) for (b = 0; b < (size_t)layer1_size; b++) {
        syn0[a * (size_t)layer1_size + b] = uniform_dist_w_in(rng);
    }
    if (expTable) {
        printf("expTable[500]=%.6f (should be ~0.5)\n", expTable[500]);
        printf("expTable[0]=%.6f\n", expTable[0]);
        printf("expTable[%d]=%.6f\n", EXP_TABLE_SIZE - 1, expTable[EXP_TABLE_SIZE - 1]);
    }
}
int main(int argc, char** argv)
{
    //std::cout << "Hello World!\n";

    output_file[0] = 0;
    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;

    // Initialize requested defaults before parsing command-line arguments
    seed = 0;
    starting_alpha = 0.05f;
    num_threads = 12;
    debug_mode = 0;
    minn = 3;
    maxn = 6;
    window = 5;
    min_count = 5;

    int i;
#ifdef _MSC_VER

    if ((i = ArgPos((char*)"-train", argc, argv)) > 0) strcpy_s(train_file, sizeof(train_file), argv[i + 1]);
    if ((i = ArgPos((char*)"-output", argc, argv)) > 0) strcpy_s(output_file, sizeof(output_file), argv[i + 1]);
    if ((i = ArgPos((char*)"-save-model", argc, argv)) > 0) strcpy_s(save_model_file, sizeof(save_model_file), argv[i + 1]);
    if ((i = ArgPos((char*)"-save-vocab", argc, argv)) > 0) strcpy_s(save_vocab_file, sizeof(save_vocab_file), argv[i + 1]);
    if ((i = ArgPos((char*)"-read-vocab", argc, argv)) > 0) strcpy_s(read_vocab_file, sizeof(read_vocab_file), argv[i + 1]);
    if ((i = ArgPos((char*)"-eos", argc, argv)) > 0) strcpy_s(EOS, sizeof(EOS), argv[i + 1]);

#else // 

    if ((i = ArgPos((char*)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char*)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char*)"-save-model", argc, argv)) > 0) strcpy(save_model_file, argv[i + 1]);
    if ((i = ArgPos((char*)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char*)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char*)"-eos", argc, argv)) > 0) strcpy(EOS, argv[i + 1]);
#endif
    if ((i = ArgPos((char*)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-model", argc, argv)) > 0) model_name = atoi(argv[i + 1]); // 0 = cbow; 1 = skip-gram; 2 = supervised
    if ((i = ArgPos((char*)"-minn", argc, argv)) > 0) minn = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-maxn", argc, argv)) > 0) maxn = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-alpha", argc, argv)) > 0) starting_alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char*)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
    if ((i = ArgPos((char*)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-seed", argc, argv)) > 0) seed = atoi(argv[i + 1]);

    // Removed the previous hard-coded debug/training overrides so command-line args are honored.
    // Defaults were already set above. Users should supply -train, -output etc. via command-line.

    vocab.clear();
    vocab.reserve((size_t)vocab_max_size);
#ifdef _MSC_VER
    vocab_hash = (int*)_aligned_malloc(vocab_hash_size * sizeof(int), 64);
#else
    int memalign_ret = posix_memalign((void**)&vocab_hash, 64, (size_t)vocab_hash_size * sizeof(int));
    if (memalign_ret != 0) vocab_hash = NULL;
#endif
    if (!vocab_hash) {
        fprintf(stderr, "posix_memalign for vocab_hash failed\n");
        exit(1);
    }
    expTable = (float*)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));
    for (int ii = 0; ii < EXP_TABLE_SIZE; ii++) {
        expTable[ii] = exp((ii / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
        expTable[ii] = expTable[ii] / (expTable[ii] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    start = clock();
    TrainModel();

    //return 0;


}
