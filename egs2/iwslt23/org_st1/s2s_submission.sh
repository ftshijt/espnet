#!/bin/bash

. ./path.sh

simuleval --standalone --remote-port 2023 \
--agent pyscripts/utils/simuleval_agent_s2st_cascaded.py \
--batch_size 1 \
--ngpu 1 \
--st_train_config /espnet/egs2/iwslt23/org_st1/st_model_final/config.yaml \
--st_model_file /espnet/egs2/iwslt23/org_st1/st_model_final/valid.acc.ave_10best_at300.pth \
--disable_repetition_detection false \
--beam_size 6 \
--ctc_weight 0.2 \
--sim_chunk_length 40000 \
--backend streaming \
--incremental_decode true \
--penalty 0.0 \
--hugging_face_decoder true \
--source-segment-size 2500 \
--recompute true \
--token_delay true \
--use_word_list false \
--tts_model /espnet/egs2/iwslt23/org_st1/tts_model_final/tts_model_final.pth \
--target-type speech \
--vocoder none \
--tts_sampling_rate 16000 \
--tts_speed_control_alpha 1.0
