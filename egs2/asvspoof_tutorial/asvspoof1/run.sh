#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


train_set=train
train_dev=dev
test_set=eval

asvspoof_config=conf/train_asvspoof.yaml
inference_config=conf/decode_asvspoof.yaml


./asvspoof.sh \
    --stage 1 \
    --ngpu 1 \
    --local_data_opts "--stage 1" \
    --asvspoof_config "${asvspoof_config}" \
    --inference_config "${inference_config}" \
    --gpu_inference true \
    --inference_nj 1 \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" "$@"

