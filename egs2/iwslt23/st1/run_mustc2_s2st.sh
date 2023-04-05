#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# set -e
# set -u
# set -o pipefail

src_lang=en
tgt_lang=de

train_set=s2st_train.en-de
train_dev=s2st_dev.en-${tgt_lang}
test_set="s2st_tst-COMMON.en-${tgt_lang} s2st_tst-HE.en-${tgt_lang}"

st_config=conf/tuning/unity.yaml
inference_config=conf/tuning/decode_st_conformer.yaml

src_nbpe=4000
tgt_nbpe=4000

tgt_suffix=.km
src_suffix=.tc.de

./s2st.sh \
    --local_data_opts "${tgt_lang}" \
    --audio_format "flac.ark" \
    --nj 40 \
    --inference_nj 40 \
    --gpu_inference true \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "bpe" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "bpe" \
    --tgt_nbpe $tgt_nbpe \
    --feats_type raw \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --st_config "${st_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" "$@" \
    --expdir exp \
    --st_stats_dir exp5/st_stats_wavlm_hf 
