#!/bin/bash

# Copyright 2023 Carnegie Mellon University (Jiatong Shi)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


. ./path.sh

data_dir=
gt_dir=eval_gt

nj=10
score_mcd=true
score_f0=true
score_asr=true

. utils/parse_options.sh

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SECONDS=0

log "data preprocess"
python local/process_submission.py --submission ${data_dir}
log "finished data preprocess in ${SECONDS}s"

# score MCD
if ${score_mcd}; then
    log "Score MCD from ${data_dir}"

    for task in "idf" "idm" "cdf" "cdm"; do
        ./pyscripts/utils/evaluate_mcd.py \
            ${data_dir}/${task}_wav.scp ${gt_dir}/${task}_wav.scp    
    done
    log "finished scoring MCD in ${SECONDS}s"
fi

# score F0
if ${score_f0}; then
    log "Score F0-RMSE ${data_dir}"

    for task in "idf" "idm" "cdf" "cdm"; do
        ./pyscripts/utils/evaluate_f0.py \
            --f0min 100 --f0max 8000 \
            ${data_dir}/${task}_wav.scp ${gt_dir}/${task}_wav.scp
    done
    log "finished scoring F0-RMSE in ${SECONDS}s"
fi

# score ASR
if ${score_asr}; then
    log "Score with ASR ${data_dir}"

    for task in "idf" "idm" "cdf" "cdm"; do
        ./scripts/utils/evalute_asr.sh \
            --model_tag espnet/ftshijt_espnet2_asr_dsing_hubert_conformer \
            --nj ${nj} \
            --stop_stage 3 \
            --inference_args "--beam_size 10 --ctc_weight 0.4 --lm_weight 0.0" \
            --gt_text ${gt_dir}/text \
            ${data_dir}/${task}_wav.scp ${data_dir}/${task}_asr_result
    done
    log "finished asr scoring in ${SECONDS}s"
fi


