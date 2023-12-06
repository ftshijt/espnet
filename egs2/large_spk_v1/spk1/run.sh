#!/usr/bin/env bash
set -e
set -u
set -o pipefail


spk_config=conf/train_rawnet3_best_trnVox12_emb192_amp_subcentertopk.yaml

train_set="train_all"
valid_set="voxceleb1_test"
test_sets="voxceleb1_test"
feats_type="raw"

./spk.sh \
    --feats_type ${feats_type} \
    --spk_config ${spk_config} \
    --train_set ${train_set} \
    --valid_set ${valid_set} \
    --test_sets ${test_sets} \
    --speed_perturb_factors "" \
    "$@"
