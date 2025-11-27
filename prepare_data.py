import os
import requests
import pandas as pd
from itertools import permutations

# 디렉토리 설정
DATA_DIR = "./fastext/data"
os.makedirs(DATA_DIR, exist_ok=True)

def download_file(url, save_path):
    print(f"Downloading {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Saved to {save_path}")
        return True
    else:
        print(f"Failed to download. Status code: {response.status_code}")
        return False

def prepare_en_data():
    """English Word Analogy Test Set (Google)"""
    url = "https://raw.githubusercontent.com/dav/word2vec/master/data/questions-words.txt"
    save_path = os.path.join(DATA_DIR, "questions-words.txt")
    download_file(url, save_path)

def prepare_cs_data():
    """Czech Word Analogy Pairs -> Analogy Format conversion"""
    # 체코어 Pairs 데이터셋 중 대표적인 CSV 파일들
    # 전체 목록: https://github.com/Svobikl/cz_corpus/tree/master/pairs
    base_url = "https://raw.githubusercontent.com/Svobikl/cz_corpus/master/pairs/"
    files = [
        "rg65-cs.csv", "ws353-cs.csv", "gur350-cs.csv" 
        # 필요에 따라 다른 파일 추가 가능
    ]
    
    analogy_output_path = os.path.join(DATA_DIR, "cs_analogies_generated.txt")
    
    with open(analogy_output_path, 'w', encoding='utf-8') as out_f:
        for filename in files:
            csv_path = os.path.join(DATA_DIR, filename)
            # 1. CSV 다운로드
            if not download_file(base_url + filename, csv_path):
                continue
            
            # 2. CSV 로드 및 어놀로지 포맷 변환 (Logic from analogy_test_cs.py)
            # 형식: A:B :: C:D (A B C D)
            try:
                # CSV 포맷에 따라 구분자 확인 필요 (보통 콤마)
                df = pd.read_csv(csv_path)
                # 헤더가 없거나 다를 수 있으므로 첫 두 컬럼을 단어 쌍으로 간주
                pairs = df.iloc[:, 0:2].values.tolist()
                
                category = filename.replace('.csv', '').replace('-', '_')
                out_f.write(f": {category}\n")
                
                # 모든 쌍의 순열 생성 (Pair A vs Pair B)
                # (a, b)와 (c, d)가 있을 때: a b c d 생성
                count = 0
                for i, (a, b) in enumerate(pairs):
                    for j, (c, d) in enumerate(pairs):
                        if i != j:
                            # 데이터 정제 (공백 제거 등)
                            a, b = str(a).strip(), str(b).strip()
                            c, d = str(c).strip(), str(d).strip()
                            out_f.write(f"{a} {b} {c} {d}\n")
                            count += 1
                print(f"  Generated {count} analogies from {filename}")
                
            except Exception as e:
                print(f"  Error processing {filename}: {e}")

    print(f"\nCzech analogy dataset created at: {analogy_output_path}")

if __name__ == "__main__":
    print(">>> Preparing English Data...")
    prepare_en_data()
    
    print("\n>>> Preparing Czech Data...")
    prepare_cs_data()
