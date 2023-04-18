#!/bin/bash

. ./path.sh

simuleval --standalone --remote-port 2023 \
--agent pyscripts/utils/simuleval_agent_s2st_cascaded.py \
--batch_size 1 \
--ngpu 1 \
--st_train_config /st_model_final/config.yaml \
--st_model_file /st_model_final/valid.acc.ave_10best_at300.pth \
--disable_repeatition_detection false \
--beam_size 5 \
--ctc_weight 0.3 \
--sim_chunk_length 40000 \
--backend streaming \
--incremental_decode true \
--penalty 0.0 \
--hugging_face_decoder true \
--source-segment-size 2500 \
--recompute true \
--token_delay false \
--tts_model /tts_model_final/tts_model_final.pth \
--target-type speech \
--vocoder none \
--tts_sampling_rate 16000 \
--tts_speed_control_alpha 1.0 \
--target-speech-lang de
