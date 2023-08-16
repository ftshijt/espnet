<!-- Generated by scripts/utils/show_asr_result.sh and scripts/utils/show_enh_score.sh -->

# MIMOIRIS (TF-GridNet-SS, WavLM_Large + SpecAug + Conformer-ASR with Transformer-LM)

## Notes
- Joint finetuning requires pre-trained Enh and ASR models.
- The language model is also trained in the ASR recipe for the original wsj.

## Environments
- date: `Mon Aug  7 09:48:51 UTC 2023`
- python version: `3.7.4 (default, Aug 13 2019, 20:35:49)  [GCC 7.3.0]`
- espnet version: `espnet 202304`
- pytorch version: `pytorch 1.10.1+cu111`
- Git hash: `277ec3c33d2ca7f47d9d31c84e4dae54ce017bd7`
  - Commit date: `Wed Aug 10 13:32:09 2022 -0400`
- run.sh
  - min_or_max=max
  - sample_rate=16k
  - mode=multi

## exp/enh_asr_train_enh_asr_tfgridnet_waspaa2023_raw_en_char
- Enh pre-training config: [../enh1/conf/tuning/train_enh_tfgridnet_waspaa2023.yaml](../enh1/conf/tuning/train_enh_tfgridnet_waspaa2023.yaml)
- ASR pre-training config: [../../wsj/asr1/conf/tuning/train_asr_conformer_s3prlfrontend_wavlm.yaml](../../wsj/asr1/conf/tuning/train_asr_conformer_s3prlfrontend_wavlm.yaml)
- Enh-ASR finetuning config: [./conf/tuning/train_enh_asr_tfgridnet_waspaa2023.yaml](./conf/tuning/train_enh_asr_tfgridnet_waspaa2023.yaml)
- LM config: [../../wsj/asr1/conf/train_lm_transformer.yaml](../../wsj/asr1/conf/train_lm_transformer.yaml)
- Pretrained model: [https://huggingface.co/espnet/yoshiki_wsj0_2mix_spatialized_enh_asr_tfgridnet_waspaa2023_raw_en_char](https://huggingface.co/espnet/yoshiki_wsj0_2mix_spatialized_enh_asr_tfgridnet_waspaa2023_raw_en_char)

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.best/tt_spatialized_anechoic_multich_max_16k|6000|98613|98.7|1.2|0.1|0.5|1.7|16.5|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.best/tt_spatialized_reverb_multich_max_16k|6000|98613|98.7|1.3|0.1|0.4|1.7|17.8|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.best/tt_spatialized_anechoic_multich_max_16k|6000|598296|99.6|0.2|0.2|0.3|0.7|21.6|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.best/tt_spatialized_reverb_multich_max_16k|6000|598296|99.6|0.2|0.3|0.3|0.7|23.0|

### Speech Separation Metrics

|dataset|STOI|SAR|SDR|SIR|SI_SNR|
|---|---|---|---|---|---|
|enhanced_tt_spatialized_anechoic_multich_max_16k|97.73|13.36|13.19|29.59|12.63|
|enhanced_tt_spatialized_reverb_multich_max_16k|95.15|11.68|11.47|26.41|10.91|

# RESULTS
## Environments
- date: `Fri Aug 26 17:20:12 CST 2022`
- python version: `3.8.11 (default, Aug  3 2021, 15:09:35)  [GCC 7.5.0]`
- espnet version: `espnet 202207`
- pytorch version: `pytorch 1.7.0`
- Git hash: `277ec3c33d2ca7f47d9d31c84e4dae54ce017bd7`
  - Commit date: `Wed Aug 10 13:32:09 2022 -0400`

## enh_asr_train_raw_en_char
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/tt_spatialized_anechoic_multich_max_16k|6000|98613|92.9|6.0|1.2|1.0|8.1|45.3|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_transformer_normalize_output_wavtrue_lm_lm_train_lm_transformer_en_char_valid.loss.ave_enh_asr_model_valid.acc.ave/tt_spatialized_anechoic_multich_max_16k|6000|598296|96.7|1.6|1.7|0.9|4.3|48.1|

### Speech Separation Metrics

|dataset|STOI|SAR|SDR|SIR|SI_SNR|
|---|---|---|---|---|---|
|enhanced_tt_spatialized_anechoic_multich_max_16k|95.25|12.03|10.24|21.74|-3.35|