#!/bin/bash

. ./path.sh

source=tst-COMMON/en-de/wavs.txt
target=tst-COMMON/en-de/refs.txt

_cmd=$cuda_cmd
${_cmd} --gpu 1 "tmp"/simuleval.log \
    simuleval \
    --agent pyscripts/utils/simuleval_agent.py \
    --batch_size 1 \
    --ngpu 1 \
    --st_train_config st_model_final/config.yaml \
    --st_model_file st_model_final/valid.acc.ave_10best_at300.pth \
    --disable_repetition_detection false \
    --beam_size 6 \
    --ctc_weight 0.1 \
    --sim_chunk_length 32000 \
    --backend streaming \
    --incremental_decode true \
    --penalty 0.0 \
    --hugging_face_decoder true \
    --source-segment-size 2000 \
    --recompute true \
    --token_delay true \
    --target-type text \
    --use_word_list true \
    --source $source \
    --target $target \
