#!/bin/bash

# 경로 설정
EVAL_SCRIPT="fasttext_evaluation.py"
DATA_DIR="./fastext/data"
RESULT_DIR="./fastext/result"
OOV_DIR="./fastext/oov"

# 1. English Analogy Test
# 가정: result 폴더에 model_en.vec, oov 폴더에 en_oov.npz가 있다고 가정
echo "----------------------------------------------------------------"
echo "Running English Word Analogy Evaluation..."
echo "----------------------------------------------------------------"
# 주의: 실제 파일명(model_en.vec, en_oov.npz)은 사용자의 실제 파일명으로 변경해야 합니다.
python $EVAL_SCRIPT \
    "$RESULT_DIR/model_en.vec" \
    "$DATA_DIR/questions-words.txt" \
    --sisg \
    --oov_npz "$OOV_DIR/en_oov.npz"


# 2. Czech Analogy Test
# 가정: result 폴더에 model_cs.vec, oov 폴더에 cs_oov.npz가 있다고 가정
echo -e "\n----------------------------------------------------------------"
echo "Running Czech Word Analogy Evaluation..."
echo "----------------------------------------------------------------"
# 주의: 실제 파일명(model_cs.vec, cs_oov.npz)은 사용자의 실제 파일명으로 변경해야 합니다.
python $EVAL_SCRIPT \
    "$RESULT_DIR/model_cs.vec" \
    "$DATA_DIR/cs_analogies_generated.txt" \
    --sisg \
    --oov_npz "$OOV_DIR/cs_oov.npz"
