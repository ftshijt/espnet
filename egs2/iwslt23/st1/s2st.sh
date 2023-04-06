#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages.
skip_train=false     # Skip training stages.
skip_eval=false      # Skip decoding and evaluation stages.
skip_upload=true     # Skip packing and uploading stages.
skip_upload_hf=true  # Skip uploading to hugging face stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw       # Feature type (raw or fbank_pitch).
audio_format=flac    # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k               # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.

# Tokenization related
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole
token_joint=false       # whether to use a single bpe system for both source and target languages
src_token_type=bpe      # Tokenization type (char or bpe) for source languages.
src_nbpe=30             # The number of BPE vocabulary for source language.
src_bpemode=unigram     # Mode of BPE for source language (unigram or bpe).
src_bpe_input_sentence_size=100000000 # Size of input sentence for BPE for source language.
src_bpe_nlsyms=         # non-linguistic symbols list, separated by a comma, for BPE of source language
src_bpe_char_cover=1.0  # character coverage when modeling BPE for source language
tgt_token_type=bpe      # Tokenization type (char or bpe) for target language.
tgt_nbpe=30             # The number of BPE vocabulary for target language.
tgt_bpemode=unigram     # Mode of BPE (unigram or bpe) for target language.
tgt_bpe_input_sentence_size=100000000 # Size of input sentence for BPE for target language.
tgt_bpe_nlsyms=         # non-linguistic symbols list, separated by a comma, for BPE for target language.
tgt_bpe_char_cover=1.0  # character coverage when modeling BPE for target language.
hugging_face_model_name_or_path="" # Hugging Face model or path for hugging_face tokenizer

tgt_suffix=
src_suffix=

# ST model related
st_tag=        # Suffix to the result dir for st model training.
st_exp=        # Specify the directory path for ST experiment.
               # If this option is specified, st_tag is ignored.
st_stats_dir=  # Specify the directory path for ST statistics.
st_config=     # Config for st model training.
st_args=       # Arguments for st model training, e.g., "--max_epoch 10".
               # Note that it will overwrite args in st config.
pretrained_asr=               # Pretrained model to load
ignore_init_mismatch=false      # Ignore initial mismatch
feats_normalize=global_mvn # Normalizaton layer type.
num_splits_st=1            # Number of splitting for lm corpus.
src_lang=es                # source language abbrev. id (e.g., es)
tgt_lang=en                # target language abbrev. id (e.g., en)
use_src_lang=true          # Incorporate ASR loss (use src texts) or not

# Upload model related
hf_repo=

# Decoding related
use_k2=false      # Whether to use k2 based decoder
use_streaming=false # Whether to use streaming decoding
batch_size=1
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
                  # Note that it will overwrite args in inference config.
inference_asr_lm=valid.loss.ave.pth       # Language model path for decoding.
inference_st_model=valid.acc.ave.pth # ST model path for decoding.
                                      # e.g.
                                      # inference_st_model=train.loss.best.pth
                                      # inference_st_model=3epoch.pth
                                      # inference_st_model=valid.acc.best.pth
                                      # inference_st_model=valid.loss.ave.pth
download_model= # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training.
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
src_bpe_train_text=  # Text file path of bpe training set for source language.
tgt_bpe_train_text=  # Text file path of bpe training set for target language.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
g2p=none         # g2p method (needed if token_type=phn).
score_opts=                # The options given to sclite scoring
local_score_opts=          # The options given to local/score.sh.
st_speech_fold_length=800 # fold_length for speech data during ST training.
st_text_fold_length=150   # fold_length for text data during ST training.

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference  # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type       # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").

    # Tokenization related
    --oov                     # Out of vocabulary symbol (default="${oov}").
    --blank                   # CTC blank symbol (default="${blank}").
    --sos_eos                 # sos and eos symbole (default="${sos_eos}").
    --token_joint=false       # Whether to use a single bpe system for both source and target languages.
                              # if set as true, will use tgt_* for processing (default="${token_joint}").
    --src_token_type=bpe      # Tokenization type (char or bpe) for source languages. (default="${src_token_type}").
    --src_nbpe=30             # The number of BPE vocabulary for source language. (default="${src_nbpe}").
    --src_bpemode=unigram     # Mode of BPE for source language (unigram or bpe). (default="${src_bpemode}").
    --src_bpe_input_sentence_size=100000000 # Size of input sentence for BPE for source language. (default="${src_bpe_input_sentence_size}").
    --src_bpe_nlsyms=         # Non-linguistic symbols list, separated by a comma, for BPE of source language. (default="${src_bpe_nlsyms}").
    --src_bpe_char_cover=1.0  # Character coverage when modeling BPE for source language. (default="${src_bpe_char_cover}").
    --tgt_token_type=bpe      # Tokenization type (char or bpe) for target language. (default="${tgt_token_type}").
    --tgt_nbpe=30             # The number of BPE vocabulary for target language. (default="${tgt_nbpe}").
    --tgt_bpemode=unigram     # Mode of BPE (unigram or bpe) for target language. (default="${tgt_bpemode}").
    --tgt_bpe_input_sentence_size=100000000 # Size of input sentence for BPE for target language. (default="${tgt_bpe_input_sentence_size}").
    --tgt_bpe_nlsyms=         # Non-linguistic symbols list, separated by a comma, for BPE for target language. (default="${tgt_bpe_nlsyms}").
    --tgt_bpe_char_cover=1.0  # Character coverage when modeling BPE for target language. (default="${tgt_bpe_char_cover}").

    # Language model related

    # ST model related
    --st_tag           # Suffix to the result dir for st model training (default="${st_tag}").
    --st_exp           # Specify the directory path for ST experiment.
                       # If this option is specified, st_tag is ignored (default="${st_exp}").
    --st_stats_dir     # Specify the directory path for ST statistics (default="${st_stats_dir}").
    --st_config        # Config for st model training (default="${st_config}").
    --st_args          # Arguments for st model training (default="${st_args}").
                       # e.g., --st_args "--max_epoch 10"
                       # Note that it will overwrite args in st config.
    --pretrained_asr=          # Pretrained model to load (default="${pretrained_asr}").
    --ignore_init_mismatch=      # Ignore mismatch parameter init with pretrained model (default="${ignore_init_mismatch}").
    --feats_normalize  # Normalizaton layer type. (default="${feats_normalize}").
    --num_splits_st    # Number of splitting for lm corpus.  (default="${num_splits_st}").
    --src_lang=        # source language abbrev. id (e.g., es). (default="${src_lang}")
    --tgt_lang=        # target language abbrev. id (e.g., en). (default="${tgt_lang}")
    --use_src_lang=    # Incorporate ASR loss (use src texts) or not

    # Decoding related
    --inference_tag       # Suffix to the result dir for decoding (default="${inference_tag}").
    --inference_config    # Config for decoding (default="${inference_config}").
    --inference_args      # Arguments for decoding (default="${inference_args}").
                          # e.g., --inference_args "--lm_weight 0.1"
                          # Note that it will overwrite args in inference config.
    --inference_st_model # ST model path for decoding (default="${inference_st_model}").
    --download_model      # Download a model from Model Zoo and use it for decoding (default="${download_model}").

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --test_sets     # Names of test sets.
                    # Multiple items (e.g., both dev and eval sets) can be specified (required).
    --src_bpe_train_text # Text file path of bpe training set for source language.
    --tgt_bpe_train_text # Text file path of bpe training set for target language
    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --cleaner       # Text cleaner (default="${cleaner}").
    --g2p           # g2p method (default="${g2p}").
    --score_opts             # The options given to sclite scoring (default="{score_opts}").
    --local_score_opts       # The options given to local/score.sh (default="{local_score_opts}").
    --st_speech_fold_length # fold_length for speech data during ST training (default="${st_speech_fold_length}").
    --st_text_fold_length   # fold_length for text data during ST training (default="${st_text_fold_length}").
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(scripts/utils/print_args.sh $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh


# Check required arguments
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
[ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/raw
elif [ "${feats_type}" = fbank_pitch ]; then
    data_feats=${dumpdir}/fbank_pitch
elif [ "${feats_type}" = fbank ]; then
    data_feats=${dumpdir}/fbank
elif [ "${feats_type}" == extracted ]; then
    data_feats=${dumpdir}/extracted
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

# Extra files for translation process
if [ $use_src_lang = true ]; then
    utt_extra_files="text.${src_suffix} text.${tgt_suffix}"
else
    utt_extra_files="text.${tgt_suffix}"
fi

# Use the same text as ST for bpe training if not specified.
[ -z "${src_bpe_train_text}" ] && [ $use_src_lang = true ] && src_bpe_train_text="${data_feats}/${train_set}/text.${src_suffix}"
[ -z "${tgt_bpe_train_text}" ] && tgt_bpe_train_text="${data_feats}/${train_set}/text.${tgt_suffix}"

# Check tokenization type
token_listdir=data/${src_lang}_${tgt_lang}_token_list
# The tgt bpedir is set for all cases when using bpe
tgt_bpedir="${token_listdir}/tgt_bpe_${tgt_bpemode}${tgt_nbpe}"
tgt_bpeprefix="${tgt_bpedir}"/bpe
tgt_bpemodel="${tgt_bpeprefix}".model
tgt_bpetoken_list="${tgt_bpedir}"/tokens.txt
tgt_chartoken_list="${token_listdir}"/char/tgt_tokens.txt
hugging_face_token_list="${token_listdir}/hugging_face_"${hugging_face_model_name_or_path/\//-}/tokens.txt
if "${token_joint}"; then
    # if token_joint, the bpe training will use both src_lang and tgt_lang to train a single bpe model
    src_bpedir="${tgt_bpedir}"
    src_bpeprefix="${tgt_bpeprefix}"
    src_bpemodel="${tgt_bpemodel}"
    src_bpetoken_list="${tgt_bpetoken_list}"
    src_chartoken_list="${tgt_chartoken_list}"
else
    src_bpedir="${token_listdir}/src_bpe_${src_bpemode}${src_nbpe}"
    src_bpeprefix="${src_bpedir}"/bpe
    src_bpemodel="${src_bpeprefix}".model
    src_bpetoken_list="${src_bpedir}"/tokens.txt
    src_chartoken_list="${token_listdir}"/char/src_tokens.txt
fi

# NOTE: keep for future development.
# shellcheck disable=SC2034
tgt_wordtoken_list="${token_listdir}"/word/tgt_tokens.txt
if "${token_joint}"; then
    src_wordtoken_list="${tgt_wordtoken_list}"
else
    src_wordtoken_list="${token_listdir}"/word/src_tokens.txt
fi

speech_token_list="${token_listdir}"/unit/tokens.txt

# Set token types for src and tgt langs
if [ $use_src_lang = false ]; then
    src_token_type=none
    src_token_list=none
elif [ "${src_token_type}" = bpe ]; then
    src_token_list="${src_bpetoken_list}"
elif [ "${src_token_type}" = char ]; then
    src_token_list="${src_chartoken_list}"
    src_bpemodel=none
elif [ "${src_token_type}" = word ]; then
    src_token_list="${src_wordtoken_list}"
    src_bpemodel=none
else
    log "Error: not supported --src_token_type '${src_token_type}'"
    exit 2
fi
if [ "${tgt_token_type}" = bpe ]; then
    tgt_token_list="${tgt_bpetoken_list}"
elif [ "${tgt_token_type}" = char ]; then
    tgt_token_list="${tgt_chartoken_list}"
    tgt_bpemodel=none
elif [ "${tgt_token_type}" = word ]; then
    tgt_token_list="${tgt_wordtoken_list}"
    tgt_bpemodel=none
elif [ "${tgt_token_type}" = hugging_face ]; then
    tgt_token_list="${hugging_face_token_list}"
    tgt_bpemodel=${hugging_face_model_name_or_path}
else
    log "Error: not supported --tgt_token_type '${tgt_token_type}'"
    exit 2
fi

# Set tag for naming of model directory
if [ -z "${st_tag}" ]; then
    if [ -n "${st_config}" ]; then
        st_tag="$(basename "${st_config}" .yaml)_${feats_type}"
    else
        st_tag="train_${feats_type}"
    fi
    st_tag+="_s2st_${src_lang}_${tgt_lang}_${tgt_token_type}"
    if [ "${tgt_token_type}" = bpe ]; then
        st_tag+="${tgt_nbpe}"
    fi
    if [ "${tgt_token_type}" = hugging_face ]; then
        st_tag+="_"${hugging_face_model_name_or_path/\//-}
    fi
    # Add overwritten arg's info
    if [ -n "${st_args}" ]; then
        st_tag+="$(echo "${st_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        st_tag+="_sp"
    fi
fi


# The directory used for collect-stats mode
if [ -z "${st_stats_dir}" ]; then
    st_stats_dir="${expdir}/st_stats_${feats_type}_${src_lang}_${tgt_lang}_${tgt_token_type}"
    if [ "${tgt_token_type}" = bpe ]; then
        st_stats_dir+="${tgt_nbpe}"
    fi
    if [ "${tgt_token_type}" = hugging_face ]; then
        st_stats_dir+="_"${hugging_face_model_name_or_path/\//-}
    fi
    if [ -n "${speed_perturb_factors}" ]; then
        st_stats_dir+="_sp"
    fi
fi

# The directory used for training commands
if [ -z "${st_exp}" ]; then
    st_exp="${expdir}/st_${st_tag}"
fi


if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
    # Add overwritten arg's info
    if [ -n "${inference_args}" ]; then
        inference_tag+="$(echo "${inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    inference_tag+="_st_model_$(echo "${inference_st_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"

    if "${use_k2}"; then
      inference_tag+="_use_k2"
    fi
fi

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then

    # NOTE(jiatong): Just use st.sh for data preparation
    if [ -n "${speed_perturb_factors}" ]; then
        train_set="${train_set}_sp"
    fi

else
    log "Skip the stages for data preparation"
fi


# ========================== Data preparation is done here. ==========================


if ! "${skip_train}"; then


    if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
        _st_train_dir="${data_feats}/${train_set}"
        _st_valid_dir="${data_feats}/${valid_set}"
        log "Stage 10: ST collect stats: train_set=${_st_train_dir}, valid_set=${_st_valid_dir}"

        _opts=
        if [ -n "${st_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.st_train --print_config --optim adam
            _opts+="--config ${st_config} "
        fi

        _feats_type="$(<${_st_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                # "sound" supports "wav", "flac", etc.
                _type=sound
            fi
            _opts+="--frontend_conf fs=${fs} "
        else
            _scp=feats.scp
            _type=kaldi_ark
            _input_size="$(<${_st_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "
        fi

        # 1. Split the key file
        _logdir="${st_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_st_train_dir}/${_scp} wc -l)" "$(<${_st_valid_dir}/${_scp} wc -l)")

        key_file="${_st_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_st_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${st_stats_dir}/run.sh'. You can resume the process from stage 10 using this script"
        mkdir -p "${st_stats_dir}"; echo "${run_args} --stage 10 \"\$@\"; exit \$?" > "${st_stats_dir}/run.sh"; chmod +x "${st_stats_dir}/run.sh"

        # 3. Submit jobs
        log "ST collect-stats started... log: '${_logdir}/stats.*.log'"

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.

        if [ $use_src_lang = true ]; then
            # IWSLT NOTE(Jiatong): replace src to tgt
            _opts+="--src_bpemodel ${tgt_bpemodel} "
            _opts+="--train_data_path_and_name_and_type ${_st_train_dir}/text.${src_suffix},src_text,text "
            _opts+="--valid_data_path_and_name_and_type ${_st_valid_dir}/text.${src_suffix},src_text,text "
        fi
        # TODO(jiatong): fix different bpe model
        # shellcheck disable=SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m espnet2.bin.st_train \
                --collect_stats true \
                --use_preprocessor true \
                --bpemodel "${tgt_bpemodel}" \
                --token_type "word" \
                --src_token_type "${tgt_token_type}" \
                --token_list "${speech_token_list}" \
                --src_token_list "${tgt_token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --train_data_path_and_name_and_type "${_st_train_dir}/${_scp},speech,${_type}" \
                --train_data_path_and_name_and_type "${_st_train_dir}/text.${tgt_suffix},text,text" \
                --valid_data_path_and_name_and_type "${_st_valid_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_st_valid_dir}/text.${tgt_suffix},text,text" \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${st_args} || { cat "${_logdir}"/stats.1.log; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        if [ "${feats_normalize}" != global_mvn ]; then
            # Skip summerizaing stats if not using global MVN
            _opts+="--skip_sum_stats"
        fi
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${st_stats_dir}"

        # Append the num-tokens at the last dimensions. This is used for batch-bins count
        <"${st_stats_dir}/train/text_shape" \
            awk -v N="$(<${speech_token_list} wc -l)" '{ print $0 "," N }' \
            >"${st_stats_dir}/train/text_shape.word"

        <"${st_stats_dir}/valid/text_shape" \
            awk -v N="$(<${speech_token_list} wc -l)" '{ print $0 "," N }' \
            >"${st_stats_dir}/valid/text_shape.word"


        if [ $use_src_lang = true ]; then
            <"${st_stats_dir}/train/src_text_shape" \
                awk -v N="$(<${src_token_list} wc -l)" '{ print $0 "," N }' \
                >"${st_stats_dir}/train/src_text_shape.${tgt_token_type}"

            <"${st_stats_dir}/valid/src_text_shape" \
                awk -v N="$(<${src_token_list} wc -l)" '{ print $0 "," N }' \
                >"${st_stats_dir}/valid/src_text_shape.${tgt_token_type}"
        fi
    fi


    if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
        _st_train_dir="${data_feats}/${train_set}"
        _st_valid_dir="${data_feats}/${valid_set}"
        log "Stage 11: ST Training: train_set=${_st_train_dir}, valid_set=${_st_valid_dir}"

        _opts=
        if [ -n "${st_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.st_train --print_config --optim adam
            _opts+="--config ${st_config} "
        fi

        _feats_type="$(<${_st_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            # "sound" supports "wav", "flac", etc.
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                _type=sound
            fi
            _fold_length="$((st_speech_fold_length * 100))"
            _opts+="--frontend_conf fs=${fs} "
        else
            _scp=feats.scp
            _type=kaldi_ark
            _fold_length="${st_speech_fold_length}"
            _input_size="$(<${_st_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "

        fi
        if [ "${feats_normalize}" = global_mvn ]; then
            # Default normalization is utterance_mvn and changes to global_mvn
            _opts+="--normalize=global_mvn --normalize_conf stats_file=${st_stats_dir}/train/feats_stats.npz "
        fi

        _num_splits_opts=
        if [ $use_src_lang = true ]; then
            _num_splits_opts+="${_st_train_dir}/text.${src_suffix} "
            _num_splits_opts+="${st_stats_dir}/train/src_text_shape.${tgt_token_type} "
        fi

        if [ "${num_splits_st}" -gt 1 ]; then
            # If you met a memory error when parsing text files, this option may help you.
            # The corpus is split into subsets and each subset is used for training one by one in order,
            # so the memory footprint can be limited to the memory required for each dataset.

            _split_dir="${st_stats_dir}/splits${num_splits_st}"
            if [ ! -f "${_split_dir}/.done" ]; then
                rm -f "${_split_dir}/.done"
                ${python} -m espnet2.bin.split_scps \
                  --scps \
                      "${_st_train_dir}/${_scp}" \
                      "${_st_train_dir}/text.${tgt_suffix}" \
                      "${st_stats_dir}/train/speech_shape" \
                      "${st_stats_dir}/train/text_shape.word" \
                      $_num_splits_opts \
                  --num_splits "${num_splits_st}" \
                  --output_dir "${_split_dir}"
                touch "${_split_dir}/.done"
            else
                log "${_split_dir}/.done exists. Spliting is skipped"
            fi

            _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/text.${tgt_suffix},text,text "
            _opts+="--train_shape_file ${_split_dir}/speech_shape "
            _opts+="--train_shape_file ${_split_dir}/text_shape.word "
            _opts+="--multiple_iterator true "
            if [ $use_src_lang = true ]; then
                _opts+="--train_data_path_and_name_and_type ${_split_dir}/text.${src_suffix},src_text,text "
                _opts+="--train_shape_file ${_split_dir}/src_text_shape.${tgt_token_type} "
            fi
        else
            _opts+="--train_data_path_and_name_and_type ${_st_train_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_st_train_dir}/text.${tgt_suffix},text,text "
            _opts+="--train_shape_file ${st_stats_dir}/train/speech_shape "
            _opts+="--train_shape_file ${st_stats_dir}/train/text_shape.word "
            if [ $use_src_lang = true ]; then
                _opts+="--train_data_path_and_name_and_type ${_st_train_dir}/text.${src_suffix},src_text,text "
                _opts+="--train_shape_file ${st_stats_dir}/train/src_text_shape.${tgt_token_type} "
            fi
        fi

        log "Generate '${st_exp}/run.sh'. You can resume the process from stage 11 using this script"
        mkdir -p "${st_exp}"; echo "${run_args} --stage 11 \"\$@\"; exit \$?" > "${st_exp}/run.sh"; chmod +x "${st_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "ST training started... log: '${st_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${st_exp})"
        else
            jobname="${st_exp}/train.log"
        fi

        if [ $use_src_lang = true ]; then
            _opts+="--src_bpemodel ${tgt_bpemodel} "
            _opts+="--valid_data_path_and_name_and_type ${_st_valid_dir}/text.${src_suffix},src_text,text "
            _opts+="--valid_shape_file ${st_stats_dir}/valid/src_text_shape.${tgt_token_type} "
            _opts+="--fold_length ${st_text_fold_length} "
        fi

        # TODO(jiatong): fix bpe
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${st_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${st_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.st_train \
                --use_preprocessor true \
                --bpemodel "${tgt_bpemodel}" \
                --token_type "word" \
                --token_list "${speech_token_list}" \
                --src_token_type "${tgt_token_type}" \
                --src_token_list "${tgt_token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --valid_data_path_and_name_and_type "${_st_valid_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_st_valid_dir}/text.${tgt_suffix},text,text" \
                --valid_shape_file "${st_stats_dir}/valid/speech_shape" \
                --valid_shape_file "${st_stats_dir}/valid/text_shape.word" \
                --resume true \
                --fold_length "${_fold_length}" \
                --fold_length "${st_text_fold_length}" \
                --output_dir "${st_exp}" \
                ${_opts} ${st_args}

    fi
else
    log "Skip the training stages"
fi


if ! "${skip_eval}"; then
    if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
        log "Stage 12: Decoding: training_dir=${st_exp}"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        _opts=
        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi

        # 2. Generate run.sh
        log "Generate '${st_exp}/${inference_tag}/run.sh'. You can resume the process from stage 12 using this script"
        mkdir -p "${st_exp}/${inference_tag}"; echo "${run_args} --stage 12 \"\$@\"; exit \$?" > "${st_exp}/${inference_tag}/run.sh"; chmod +x "${st_exp}/${inference_tag}/run.sh"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${st_exp}/${inference_tag}/${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _feats_type="$(<${_data}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _scp=wav.scp
                if [[ "${audio_format}" == *ark* ]]; then
                    _type=kaldi_ark
                else
                    _type=sound
                fi
            else
                _scp=feats.scp
                _type=kaldi_ark
            fi

            # 1. Split the key file
            key_file=${_data}/${_scp}
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            if "${use_streaming}"; then
                st_inference_tool="espnet2.bin.st_inference_streaming"
            else
                st_inference_tool="espnet2.bin.st_inference"
            fi

            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/st_inference.*.log'"
            # shellcheck disable=SC2046,SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/st_inference.JOB.log \
                ${python} -m ${st_inference_tool} \
                    --batch_size ${batch_size} \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --st_train_config "${st_exp}"/config.yaml \
                    --st_model_file "${st_exp}"/"${inference_st_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} ${inference_args} || { cat $(grep -l -i error "${_logdir}"/st_inference.*.log) ; exit 1; }

            # 3. Concatenates the output files from each jobs
            for f in token token_int score text; do
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/1best_recog/${f}"
                done | LC_ALL=C sort -k1 >"${_dir}/${f}"
            done

            if [ -d "${_logdir}/output.1/1asr_best_recog" ]; then
                for f in asr_token asr_token_int asr_score asr_text; do
                    for i in $(seq "${_nj}"); do
                        cat "${_logdir}/output.${i}/1asr_best_recog/${f}"
                    done | LC_ALL=C sort -k1 >"${_dir}/${f}"
                done
            fi
        done
    fi

else
    log "Skip the evaluation stages"
fi
