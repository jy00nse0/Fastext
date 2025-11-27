// bring_subword_vectors.cpp
//
// 사용법:
//   g++ -O2 -std=c++17 bring_subword_vectors.cpp -o bring_subword_vectors
//   chmod +x bring_subword_vectors
//
// 실행 인자 (argc = 5):
//   argv[1] : subword_file_path
//             - 한 줄에 하나의 서브워드 문자열 (예: "<deu", "deu", "<german", ...)
//   argv[2] : output_file_path
//             - 각 서브워드에 대해 "subword f1 f2 ... fD" 형식 텍스트 출력
//   argv[3] : model.bin 파일 경로
//   argv[4] : vocab_size   (파이썬에서 vec_xx.txt 읽어 len(vectors.keys())로 계산하여 전달)
//
// output:
//   - 입력된 모든 subword에 대해 syn0에서 벡터를 직접 읽어 텍스트로 저장
//   - OOV word 복원 시 필요한 subword embedding 재구성 가능

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <iomanip>

#define MAX_STRING 1000


// ====== fastText 해시 함수 (원본 동일) ======
uint32_t FastTextHash(const char* word) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; word[i] != '\0'; i++) {
        h = h ^ uint32_t(int8_t(word[i]));
        h = h * 16777619u;
    }
    return h;
}

long long GetWordHash(char* word, long long bucket_size) {
    return (long long)(FastTextHash(word) % (uint32_t)bucket_size);
}


// ====== 모델 로딩 (header + syn0) ======
bool loadModel(
    const char* filename,
    long long vocab_size,          // 외부 입력
    long long& bucket_size,
    long long& layer1_size,
    long long& train_words,
    int& negative,
    int& loss_name,
    std::vector<float>& syn0_out
) {
    FILE* fi = std::fopen(filename, "rb");
    if (!fi) {
        std::cerr << "Error: cannot open model file: " << filename << std::endl;
        return false;
    }

    // ====== SaveModel 순서 그대로 header 1단계 읽기 ======
    int binary, debug_mode, window, min_count, num_threads, min_reduce, label_count;
    int minn, maxn, model_name, seed_i;
    int hs;
    bool normalizeGradient;
    long long vocab_max_size, bucket_size_ll, layer1_size_ll;
    long long tokenCount_, word_count_actual, iter_ll, file_size_ll, classes_ll;
    float starting_alpha, alpha, sample;
    long long train_words_ll;

    // header block
    std::fread(&binary,           sizeof(int),        1, fi);
    std::fread(&debug_mode,       sizeof(int),        1, fi);
    std::fread(&window,           sizeof(int),        1, fi);
    std::fread(&min_count,        sizeof(int),        1, fi);
    std::fread(&num_threads,      sizeof(int),        1, fi);
    std::fread(&min_reduce,       sizeof(int),        1, fi);
    std::fread(&label_count,      sizeof(int),        1, fi);
    std::fread(&minn,             sizeof(int),        1, fi);
    std::fread(&maxn,             sizeof(int),        1, fi);
    std::fread(&model_name,       sizeof(int),        1, fi);
    std::fread(&loss_name,        sizeof(int),        1, fi);
    std::fread(&seed_i,           sizeof(int),        1, fi);

    std::fread(&vocab_max_size,   sizeof(long long),  1, fi);
    std::fread(&layer1_size_ll,   sizeof(long long),  1, fi);
    std::fread(&bucket_size_ll,   sizeof(long long),  1, fi);

    std::fread(&tokenCount_,      sizeof(long long),  1, fi);
    std::fread(&train_words_ll,   sizeof(long long),  1, fi);
    std::fread(&word_count_actual,sizeof(long long),  1, fi);
    std::fread(&iter_ll,          sizeof(long long),  1, fi);
    std::fread(&file_size_ll,     sizeof(long long),  1, fi);
    std::fread(&classes_ll,       sizeof(long long),  1, fi);

    std::fread(&starting_alpha,   sizeof(float),      1, fi);
    std::fread(&alpha,            sizeof(float),      1, fi);
    std::fread(&sample,           sizeof(float),      1, fi);
    std::fread(&hs,               sizeof(int),        1, fi);
    std::fread(&negative,         sizeof(int),        1, fi);
    std::fread(&normalizeGradient,sizeof(bool),       1, fi);

    layer1_size = layer1_size_ll;
    bucket_size = bucket_size_ll;
    train_words = train_words_ll;

    // ====== vocab 영역 스킵 ======
    for (long long i = 0; i < vocab_size; i++) {
        int len;
        std::fread(&len, sizeof(int), 1, fi);

        std::vector<char> buf(len);
        std::fread(buf.data(), sizeof(char), len, fi);

        long long cn;
        int type;
        int subword_size;
        std::fread(&cn, sizeof(long long), 1, fi);
        std::fread(&type, sizeof(int), 1, fi);
        std::fread(&subword_size, sizeof(int), 1, fi);

        if (subword_size > 0) {
            std::fseek(fi, sizeof(int) * subword_size, SEEK_CUR);
        }
    }

    // ===== syn0 (vocab_size + bucket_size) * layer1_size float =====
    long long total_rows = vocab_size + bucket_size;
    long long total_dim  = total_rows * layer1_size;

    syn0_out.resize((size_t)total_dim);

    size_t read_amount =
        std::fread(syn0_out.data(), sizeof(float), (size_t)total_dim, fi);

    if (read_amount != (size_t)total_dim) {
        std::cerr << "Error: failed to read syn0. Expected "
                  << total_dim << " floats, got " << read_amount << std::endl;
        std::fclose(fi);
        return false;
    }

    std::fclose(fi);
    return true;
}


// ====== subword 문자열 → syn0 vector ======
bool getSubwordVector(
    const std::string& subword,
    long long vocab_size,
    long long bucket_size,
    long long layer1_size,
    const std::vector<float>& syn0,
    std::vector<float>& out_vec
) {
    char buf[MAX_STRING * 4];
    if (subword.size() >= sizeof(buf)) return false;

    std::strcpy(buf, subword.c_str());
    long long h = GetWordHash(buf, bucket_size);
    long long row = vocab_size + h;

    long long total_rows = vocab_size + bucket_size;
    if (row < 0 || row >= total_rows) return false;

    out_vec.resize((size_t)layer1_size);
    long long offset = row * layer1_size;
    for (long long i = 0; i < layer1_size; i++) {
        out_vec[(size_t)i] = syn0[(size_t)(offset + i)];
    }
    return true;
}


// ====== main ======
int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage:\n  "
                  << argv[0]
                  << " <subword_file> <output_vec_file> <model.bin> <vocab_size>\n";
        return 1;
    }

    const char* subword_file  = argv[1];
    const char* output_file   = argv[2];
    const char* model_file    = argv[3];
    long long vocab_size      = std::atoll(argv[4]);

    long long bucket_size = 0;
    long long layer1_size = 0;
    long long train_words = 0;
    int negative = 0;
    int loss_name = 0;
    std::vector<float> syn0;

    if (!loadModel(model_file, vocab_size,
                   bucket_size, layer1_size,
                   train_words, negative, loss_name, syn0)) {
        return 1;
    }

    std::cerr << "Loaded model.\n"
              << "  vocab_size  = " << vocab_size << "\n"
              << "  bucket_size = " << bucket_size << "\n"
              << "  layer1_size = " << layer1_size << "\n";

    std::ifstream fin(subword_file);
    if (!fin.is_open()) {
        std::cerr << "Error: cannot open subword file\n";
        return 1;
    }

    std::ofstream fout(output_file);
    if (!fout.is_open()) {
        std::cerr << "Error: cannot open output file\n";
        return 1;
    }

    fout << std::fixed << std::setprecision(6);

    std::string line;
    long long cnt = 0;

    while (std::getline(fin, line)) {
        size_t s = line.find_first_not_of(" \t\r\n");
        size_t e = line.find_last_not_of(" \t\r\n");
        if (s == std::string::npos) continue;
        std::string sub = line.substr(s, e - s + 1);
        if (sub.empty()) continue;

        std::vector<float> vec;
        if (!getSubwordVector(sub, vocab_size, bucket_size, layer1_size, syn0, vec)) {
            std::cerr << "[WARN] failed for subword: " << sub << "\n";
            continue;
        }

        fout << sub;
        for (float v : vec) fout << " " << v;
        fout << "\n";

        cnt++;
    }

    std::cerr << "Done. extracted " << cnt << " vectors.\n";
    return 0;
}
