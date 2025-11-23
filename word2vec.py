import argparse
import io
import os
import time
from gensim.models import Word2Vec

############################################################
# 1) UTF-8 안전 ReadWord (fastText C++ 동일) - Chunked/buffered version
############################################################

def utf8_readword_generator(path, chunk_size=65536, progress_callback=None):
    """
    fastText C++ ReadWord()와 거의 동일한 UTF-8 안전 토크나이저.
    파일에서 단어를 하나씩 스트림 형태로 yield.
    
    Optimized version using chunked reads (64KB by default) instead of per-byte reads
    for better performance on Windows.
    """

    with io.open(path, "rb") as f:
        buf = []
        chunk_buf = b''
        pos = 0
        total_processed = 0
        
        while True:
            # Refill chunk buffer if needed
            if pos >= len(chunk_buf):
                chunk_buf = f.read(chunk_size)
                pos = 0
                if progress_callback and chunk_buf:
                    total_processed += len(chunk_buf)
                    progress_callback(len(chunk_buf))
                if not chunk_buf:
                    if buf:
                        yield bytes(buf).decode('utf-8', errors='ignore')
                    return
            
            ch = chunk_buf[pos:pos+1]
            pos += 1

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
                    # Refill chunk buffer if needed for multi-byte UTF-8
                    if pos >= len(chunk_buf):
                        chunk_buf = f.read(chunk_size)
                        pos = 0
                        if progress_callback and chunk_buf:
                            total_processed += len(chunk_buf)
                            progress_callback(len(chunk_buf))
                        if not chunk_buf:
                            break
                    
                    cc = chunk_buf[pos:pos+1]
                    pos += 1
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


class Sentences:
    """
    Re-iterable sentences iterator without progress tracking.
    Used when progress is disabled to maintain backward compatibility.
    """
    
    def __init__(self, corpus_path, epochs):
        self.corpus_path = corpus_path
        self.epochs = epochs
    
    def __iter__(self):
        """Make this class re-iterable for gensim"""
        for epoch in range(self.epochs):
            sent = []
            for w in utf8_readword_generator(self.corpus_path):
                if w == "</s>":
                    if sent:
                        yield sent
                        sent = []
                else:
                    sent.append(w)
            
            if sent:
                yield sent


############################################################
# 3) Progress tracking helpers
############################################################

def format_eta(seconds):
    """Format seconds as H:MM:SS"""
    if seconds < 0 or seconds == float('inf'):
        return "??:??:??"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def format_speed(bytes_per_sec):
    """Format speed as MB/s"""
    mb_per_sec = bytes_per_sec / (1024 * 1024)
    return f"{mb_per_sec:.2f}"


class ProgressTracker:
    """Tracks progress and prints throttled updates"""
    
    def __init__(self, total_bytes, interval=5.0, min_step_percent=0.5):
        self.total_bytes = total_bytes
        self.processed_bytes = 0
        self.start_time = time.time()
        self.last_print_time = self.start_time
        self.last_print_percent = 0.0
        self.interval = interval
        self.min_step_percent = min_step_percent
    
    def update(self, bytes_count):
        """Update progress by bytes_count"""
        self.processed_bytes += bytes_count
    
    def should_print(self):
        """Check if we should print progress (based on time or percent change)"""
        current_time = time.time()
        time_elapsed = current_time - self.last_print_time
        
        current_percent = (self.processed_bytes / self.total_bytes * 100) if self.total_bytes > 0 else 0
        percent_change = current_percent - self.last_print_percent
        
        return time_elapsed >= self.interval or percent_change >= self.min_step_percent
    
    def print_progress(self, force=False):
        """Print progress line if should_print or force is True"""
        if not force and not self.should_print():
            return
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if elapsed > 0 and self.processed_bytes > 0:
            speed = self.processed_bytes / elapsed
            if self.total_bytes > 0:
                percent = self.processed_bytes / self.total_bytes * 100
                remaining_bytes = self.total_bytes - self.processed_bytes
                eta_seconds = remaining_bytes / speed if speed > 0 else float('inf')
                eta_str = format_eta(eta_seconds)
            else:
                percent = 0
                eta_str = "??:??:??"
            
            speed_str = format_speed(speed)
            print(f"Progress: {percent:.1f}% ({self.processed_bytes}/{self.total_bytes} bytes) "
                  f"ETA: {eta_str} Speed: {speed_str} MB/s", flush=True)
            
            self.last_print_time = current_time
            self.last_print_percent = percent
    
    def finish(self):
        """Print final progress"""
        self.print_progress(force=True)


class SentencesWithProgress:
    """
    Re-iterable sentences iterator with progress tracking.
    Yields sentences for multiple epochs, tracking progress and invoking callback.
    """
    
    def __init__(self, corpus_path, epochs, progress_tracker=None):
        self.corpus_path = corpus_path
        self.epochs = epochs
        self.progress_tracker = progress_tracker
    
    def __iter__(self):
        """Make this class re-iterable for gensim"""
        for epoch in range(self.epochs):
            sent = []
            
            def progress_callback(bytes_count):
                if self.progress_tracker:
                    self.progress_tracker.update(bytes_count)
                    self.progress_tracker.print_progress()
            
            for w in utf8_readword_generator(self.corpus_path, progress_callback=progress_callback):
                if w == "</s>":
                    if sent:
                        yield sent
                        sent = []
                else:
                    sent.append(w)
            
            if sent:
                yield sent


############################################################
# 4) Word2Vec 학습 함수
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
    alpha,
    progress=False,
    progress_interval=5.0,
    progress_min_step_percent=0.5
):
    if progress:
        # Use progress-enabled re-iterable sentences
        file_size = os.path.getsize(corpus_path)
        # Note: We use epochs=1 in SentencesWithProgress and let Word2Vec handle epochs
        # because gensim needs to iterate over corpus: 1 pass for vocab + epochs passes for training
        total_bytes = file_size * (epochs + 1)
        tracker = ProgressTracker(total_bytes, progress_interval, progress_min_step_percent)
        sentences = SentencesWithProgress(corpus_path, 1, tracker)
        
        print(f"Training with progress tracking enabled (file: {file_size} bytes, epochs: {epochs}, total: {total_bytes} bytes)")
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg,
            negative=negative,
            sample=sample,
            epochs=epochs,  # Let Word2Vec handle epochs
            alpha=alpha,
        )
        tracker.finish()
    else:
        # Use re-iterable sentences without progress tracking
        sentences = Sentences(corpus_path, epochs)
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg,
            negative=negative,
            sample=sample,
            epochs=1,  # Sentences class handles epochs internally
            alpha=alpha,
        )
    return model


############################################################
# 5) fastText SaveVectors() 동일한 .vec로 저장
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
# 6) main
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
    
    # Progress tracking options
    parser.add_argument("--progress", action="store_true", 
                        help="Enable progress tracking and ETA display")
    parser.add_argument("--progress-interval", type=float, default=5.0,
                        help="Minimum seconds between progress prints (default: 5.0)")
    parser.add_argument("--progress-min-step-percent", type=float, default=0.5,
                        help="Minimum percent change to trigger progress print (default: 0.5)")

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
        alpha=args.alpha,
        progress=args.progress,
        progress_interval=args.progress_interval,
        progress_min_step_percent=args.progress_min_step_percent
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
