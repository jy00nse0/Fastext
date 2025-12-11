import os

PERCENTS = [1, 2, 5, 10, 20, 50]

def make_subsets(wiki_path):
    assert os.path.isfile(wiki_path), f"File not found: {wiki_path}"

    base_dir = os.path.dirname(wiki_path)
    base_name = os.path.basename(wiki_path)  # wiki_en.txt
    name, ext = os.path.splitext(base_name)  # wiki_en, .txt

    print(f"Loading lines from {wiki_path}...")
    with open(wiki_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    print(f"Total lines: {total}")

    for p in PERCENTS:
        n = max(1, total * p // 100)  # subset size
        out_path = os.path.join(base_dir, f"{name}_{p}p{ext}")

        print(f"Writing {p}% subset → {out_path}  ({n} lines)")

        with open(out_path, "w", encoding="utf-8") as out:
            out.writelines(lines[:n])

    print("Done.")

if __name__ == "__main__":
    # 원하는 파일 경로를 직접 넣거나 argparse 로 확장 가능
    make_subsets("/content/drive/MyDrive/fasttext/data/data/wiki.en.txt")
    make_subsets("/content/drive/MyDrive/fasttext/data/data/wiki.de.txt")
