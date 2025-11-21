import argparse
import io
from gensim.models import Word2Vec

############################################################
# 1) UTF-8 안전 ReadWord (fastText C++ 동일)
############################################################

def utf8_readword_generator(path):
    """
    fastText C++ ReadWord()와 거의 동일한 UTF-8 안전 토크나이저.
    파일에서 단어를 하나씩 스트림 형태로 yield.
    """

    with io.open(path, "rb") as f:
        buf = []
        while True:
            ch = f.read(1)
            if not ch:
                if buf:
                    yield bytes(buf).decode('utf-8', errors='ignore')
                return

            # whitespace
            if ch in b' \t\r\v\f':
                if buf:
                    yield bytes(buf).decode('utf-8', errors='ignore')
                    buf = []
                continue

            # newline -> </s>
            if ch == b'\n':
                if buf:
                    yield bytes(buf).decode('utf-8', errors='ignore')
                    buf = []
                yield "</s>"
                continue

            # UTF-8 다바이트 처리
            c = ch[0]
            buf.append(c)

            if c >= 0xC0:
                if (c & 0xE0) == 0xC0:
                    more = 1
                elif (c & 0xF0) == 0xE0:
                    more = 2
                elif (c & 0xF8) == 0xF0:
                    more = 3
                else:
                    more = 0

                for _ in range(more):
                    cc = f.read(1)
                    if not cc:
                        break
                    if (cc[0] & 0xC0) != 0x80:
                        break
                    buf.append(cc[0])


############################################################
# 2) 문장 단위로 묶기 (gensim 학습 인풋)
############################################################

def sentence_generator(path):
    sent = []
    for w in utf8_readword_generator(path):
        if w == "</s>":
            if sent:
                yield sent
                sent = []
        else:
            sent.append(w)

    if sent:
        yield sent


############################################################
# 3) Word2Vec 학습 함수
############################################################

def train_word2vec(
    corpus_path,
    vector_size,
    window,
    min_count,
    workers,
    sg,
    negative,
    sample,
    epochs,
    alpha
):
    sentences = sentence_generator(corpus_path)
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        negative=negative,
        sample=sample,
        epochs=epochs,
        alpha=alpha,          # ← 추가
    )
    return model


############################################################
# 4) fastText SaveVectors() 동일한 .vec로 저장
############################################################

def save_as_fasttext_vec(model, out_path):
    vocab = model.wv.index_to_key
    dim = model.wv.vector_size

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{len(vocab)} {dim}\n")
        for w in vocab:
            vec = model.wv[w]
            vec_str = " ".join(f"{x:.6f}" for x in vec)
            f.write(f"{w} {vec_str}\n")


############################################################
# 5) main
############################################################

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--sg", type=int, default=1)                # 1=SG, 0=CBOW
    parser.add_argument("--vector_size", type=int, default=300)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--negative", type=int, default=5)
    parser.add_argument("--sample", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.05)

    args = parser.parse_args()

    print("Training model...")
    model = train_word2vec(
        corpus_path=args.corpus,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=args.sg,
        negative=args.negative,
        sample=args.sample,
        epochs=args.epochs,
        alpha=args.alpha
    )

    model_path = args.output + ".model"
    vec_path = args.output + ".vec"

    print("Saving gensim model:", model_path)
    model.save(model_path)

    print("Saving fastText-format .vec:", vec_path)
    save_as_fasttext_vec(model, vec_path)

    print("Done.")


if __name__ == "__main__":
    main()
