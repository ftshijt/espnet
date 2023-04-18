# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from simuleval.utils import entrypoint
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.data.segments import SpeechSegment
from simuleval.agents import SpeechToSpeechAgent


from espnet2.bin.st_inference import Speech2Text
from espnet2.bin.tts_inference import Text2Speech
from espnet2.bin.st_inference_streaming import Speech2TextStreaming
from espnet2.utils.types import str2bool, str2triple_str, str_or_none
from typing import Any, List, Optional, Sequence, Tuple, Union
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
import torch
import logging
from mosestokenizer import MosesDetokenizer
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
import pickle


# text preprocessor
from german_transliterate.core import GermanTransliterate
ops = {'acronym_phoneme', 'accent_peculiarity', 'amount_money', 'date', 'timestamp',
        'weekday', 'month', 'time_of_day', 'ordinal', 'special', 'spoken_symbol'}
import re
def text_normalizer(content):
    content = re.sub(" &quot;", "", content)
    content = re.sub(" &apos;", "", content)
    content = re.sub("&apos; ", "", content)
    content = re.sub("&quot; ", "", content)
    content = re.sub("\(.*?\)", "", content)
    content = re.sub("\(.*?\) ", "", content)
    content = re.sub(" \(.*?\) ", " ", content)
    content = re.sub(" %", " prozent ", content)
    content = re.sub(" \.\.\. ", ". ", content)
    content = re.sub(" ,", ",", content)
    content = re.sub(" \.", ".", content)
    content = re.sub(" !", ".", content)
    content = re.sub(" : ", " ", content)
    content = re.sub(" \?", ".", content)
    content = re.sub(" &#91;", "", content)
    content = re.sub(" &#93;", "", content)
    content = GermanTransliterate(transliterate_ops=list(ops-{'spoken_symbol', 'acronym_phoneme'})).transliterate(content)
    content = content.upper()
    return content

def simple_text_normalizer(content):
    content = re.sub(" &quot;", "", content)
    content = re.sub(" &apos;", "", content)
    content = re.sub("&apos; ", "", content)
    content = re.sub("&quot; ", "", content)
    content = re.sub("\(.*?\)", "", content)
    content = re.sub("\(.*?\) ", "", content)
    content = re.sub(" \(.*?\) ", " ", content)
    content = re.sub(" %", " prozent ", content)
    content = re.sub(" \.\.\. ", ". ", content)
    content = re.sub(" ,", ",", content)
    content = re.sub(" \.", ".", content)
    content = re.sub(" !", ".", content)
    content = re.sub(" : ", " ", content)
    content = re.sub(" \?", ".", content)
    content = re.sub(" &#91;", "", content)
    content = re.sub(" &#93;", "", content)
    content = content.upper()
    return content

@entrypoint
class DummyAgent(SpeechToSpeechAgent):
    """
    DummyAgent operates in an offline mode.
    Waits until all source is read to run inference.
    """

    def __init__(self, args):
        super().__init__(args)
        kwargs = vars(args)
    
        logging.basicConfig(
            level=kwargs['log_level'],
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )

        if kwargs['ngpu'] >= 1:
            device = "cuda"
        else:
            device = "cpu"

        # 1. Set random-seed
        set_all_random_seed(kwargs['seed'])

        # 2. Build speech2text
        if kwargs['backend'] == 'offline':
            speech2text_kwargs = dict(
                st_train_config=kwargs['st_train_config'],
                st_model_file=kwargs['st_model_file'],
                transducer_conf=kwargs['transducer_conf'],
                lm_train_config=kwargs['lm_train_config'],
                lm_file=kwargs['lm_file'],
                ngram_file=kwargs['ngram_file'],
                src_lm_train_config=kwargs['src_lm_train_config'],
                src_lm_file=kwargs['src_lm_file'],
                src_ngram_file=kwargs['src_ngram_file'],
                token_type=kwargs['token_type'],
                bpemodel=kwargs['bpemodel'],
                src_token_type=kwargs['src_token_type'],
                src_bpemodel=kwargs['src_bpemodel'],
                device=device,
                maxlenratio=kwargs['maxlenratio'],
                minlenratio=kwargs['minlenratio'],
                asr_maxlenratio=kwargs['asr_maxlenratio'],
                asr_minlenratio=kwargs['asr_minlenratio'],
                dtype=kwargs['dtype'],
                beam_size=kwargs['beam_size'],
                ctc_weight=kwargs['ctc_weight'],
                lm_weight=kwargs['lm_weight'],
                ngram_weight=kwargs['ngram_weight'],
                penalty=kwargs['penalty'],
                nbest=kwargs['nbest'],
                asr_beam_size=kwargs['asr_beam_size'],
                asr_ctc_weight=kwargs['asr_ctc_weight'],
                asr_lm_weight=kwargs['asr_lm_weight'],
                asr_ngram_weight=kwargs['asr_ngram_weight'],
                asr_penalty=kwargs['asr_penalty'],
                asr_nbest=kwargs['asr_nbest'],
                enh_s2t_task=kwargs['enh_s2t_task'],
                ctc_greedy=kwargs['ctc_greedy']
            )
            self.speech2text = Speech2Text.from_pretrained(
                model_tag=kwargs['model_tag'],
                **speech2text_kwargs,
            )
        else:
            if kwargs['rnnt']:
                transducer_conf = {"search_type":"tsd2", "score_norm":True, "max_sym_exp":8}
            else:
                transducer_conf = None

            speech2text_kwargs = dict(
                st_train_config=kwargs['st_train_config'],
                st_model_file=kwargs['st_model_file'],
                lm_train_config=kwargs['lm_train_config'],
                lm_file=kwargs['lm_file'],
                token_type=kwargs['token_type'],
                bpemodel=kwargs['bpemodel'],
                device=device,
                maxlenratio=kwargs['maxlenratio'],
                minlenratio=kwargs['minlenratio'],
                dtype=kwargs['dtype'],
                beam_size=kwargs['beam_size'],
                ctc_weight=kwargs['ctc_weight'],
                lm_weight=kwargs['lm_weight'],
                penalty=kwargs['penalty'],
                nbest=kwargs['nbest'],
                disable_repetition_detection=kwargs['disable_repetition_detection'],
                decoder_text_length_limit=kwargs['decoder_text_length_limit'],
                encoded_feat_length_limit=kwargs['encoded_feat_length_limit'],
                time_sync=kwargs['time_sync'],
                incremental_decode=kwargs['incremental_decode'],
                blank_penalty=kwargs['blank_penalty'],
                hold_n=kwargs['hold_n'],
                transducer_conf=transducer_conf,
                hugging_face_decoder=kwargs['hugging_face_decoder'],
            )
            self.speech2text = Speech2TextStreaming(**speech2text_kwargs)
        
        # 3. build text2speech
        self.text2speech = Text2Speech.from_pretrained(
            model_file=kwargs["tts_model"],
            vocoder_file=kwargs["vocoder"],
            device=device,
            # Only for Tacotron 2 & Transformer
            threshold=kwargs["tts_threshold"],
            # Only for Tacotron 2
            minlenratio=kwargs["tts_minlenratio"],
            maxlenratio=kwargs["tts_maxlenratio"],
            use_att_constraint=kwargs["tts_use_att_constraint"],
            backward_window=kwargs["tts_backward_window"],
            forward_window=kwargs["tts_forward_window"],
            # Only for FastSpeech & FastSpeech2 & VITS
            speed_control_alpha=kwargs["tts_speed_control_alpha"],
            # Only for VITS
            noise_scale=kwargs["tts_noise_scale"],
            noise_scale_dur=kwargs["tts_noise_scale_dur"],
        )
        
        self.sim_chunk_length = kwargs['sim_chunk_length']
        self.backend = kwargs['backend']
        self.token_delay = kwargs['token_delay']
        self.lang = kwargs['lang']
        self.recompute = kwargs['recompute']
        self.fs = kwargs["tts_sampling_rate"]
        self.clean()
        self.word_list = pickle.load(open('german_dict.obj', 'rb')) if kwargs['use_word_list'] else None

    @staticmethod
    def add_args(parser):
        # Note(kamo): Use '_' instead of '-' as separator.
        # '-' is confusing if written in yaml.
        parser.add_argument(
            "--log_level",
            type=lambda x: x.upper(),
            default="INFO",
            choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
            help="The verbose level of logging",
        )

        parser.add_argument(
            "--ngpu",
            type=int,
            default=0,
            help="The number of gpus. 0 indicates CPU mode",
        )
        parser.add_argument("--seed", type=int, default=0, help="Random seed")
        parser.add_argument(
            "--dtype",
            default="float32",
            choices=["float16", "float32", "float64"],
            help="Data type",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=1,
            help="The number of workers used for DataLoader",
        )

        group = parser.add_argument_group("The model configuration related")
        group.add_argument(
            "--st_train_config",
            type=str,
            help="ST training configuration",
        )
        group.add_argument(
            "--st_model_file",
            type=str,
            help="ST model parameter file",
        )
        group.add_argument(
            "--lm_train_config",
            type=str,
            help="LM training configuration",
        )
        group.add_argument(
            "--src_lm_train_config",
            type=str,
            help="LM training configuration",
        )
        group.add_argument(
            "--lm_file",
            type=str,
            help="LM parameter file",
        )
        group.add_argument(
            "--src_lm_file",
            type=str,
            help="LM parameter file",
        )
        group.add_argument(
            "--word_lm_train_config",
            type=str,
            help="Word LM training configuration",
        )
        group.add_argument(
            "--src_word_lm_train_config",
            type=str,
            help="Word LM training configuration",
        )
        group.add_argument(
            "--word_lm_file",
            type=str,
            help="Word LM parameter file",
        )
        group.add_argument(
            "--src_word_lm_file",
            type=str,
            help="Word LM parameter file",
        )
        group.add_argument(
            "--ngram_file",
            type=str,
            help="N-gram parameter file",
        )
        group.add_argument(
            "--src_ngram_file",
            type=str,
            help="N-gram parameter file",
        )
        group.add_argument(
            "--model_tag",
            type=str,
            help="Pretrained model tag. If specify this option, *_train_config and "
            "*_file will be overwritten",
        )
        group.add_argument(
            "--enh_s2t_task",
            type=str2bool,
            default=False,
            help="enhancement and asr joint model",
        )

        group = parser.add_argument_group("Beam-search related")
        group.add_argument(
            "--batch_size",
            type=int,
            default=1,
            help="The batch size for inference",
        )
        group.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
        group.add_argument("--asr_nbest", type=int, default=1, help="Output N-best hypotheses")
        group.add_argument("--beam_size", type=int, default=20, help="Beam size")
        group.add_argument("--asr_beam_size", type=int, default=20, help="Beam size")
        group.add_argument("--penalty", type=float, default=0.0, help="Insertion penalty")
        group.add_argument("--asr_penalty", type=float, default=0.0, help="Insertion penalty")
        group.add_argument(
            "--maxlenratio",
            type=float,
            default=0.0,
            help="Input length ratio to obtain max output length. "
            "If maxlenratio=0.0 (default), it uses a end-detect "
            "function "
            "to automatically find maximum hypothesis lengths."
            "If maxlenratio<0.0, its absolute value is interpreted"
            "as a constant max output length",
        )
        group.add_argument(
            "--asr_maxlenratio",
            type=float,
            default=0.0,
            help="Input length ratio to obtain max output length. "
            "If maxlenratio=0.0 (default), it uses a end-detect "
            "function "
            "to automatically find maximum hypothesis lengths."
            "If maxlenratio<0.0, its absolute value is interpreted"
            "as a constant max output length",
        )
        group.add_argument(
            "--minlenratio",
            type=float,
            default=0.0,
            help="Input length ratio to obtain min output length",
        )
        group.add_argument(
            "--asr_minlenratio",
            type=float,
            default=0.0,
            help="Input length ratio to obtain min output length",
        )
        group.add_argument("--lm_weight", type=float, default=1.0, help="RNNLM weight")
        group.add_argument("--asr_lm_weight", type=float, default=1.0, help="RNNLM weight")
        group.add_argument("--ngram_weight", type=float, default=0.9, help="ngram weight")
        group.add_argument("--asr_ngram_weight", type=float, default=0.9, help="ngram weight")
        group.add_argument("--ctc_weight", type=float, default=0.0, help="ST CTC weight")
        group.add_argument("--asr_ctc_weight", type=float, default=0.3, help="ASR CTC weight")

        group.add_argument(
            "--transducer_conf",
            default=None,
            help="The keyword arguments for transducer beam search.",
        )

        group = parser.add_argument_group("Text converter related")
        group.add_argument(
            "--token_type",
            type=str_or_none,
            default=None,
            choices=["char", "bpe", None],
            help="The token type for ST model. "
            "If not given, refers from the training args",
        )
        group.add_argument(
            "--src_token_type",
            type=str_or_none,
            default=None,
            choices=["char", "bpe", None],
            help="The token type for ST model. "
            "If not given, refers from the training args",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model path of sentencepiece. "
            "If not given, refers from the training args",
        )
        group.add_argument(
            "--src_bpemodel",
            type=str_or_none,
            default=None,
            help="The model path of sentencepiece. "
            "If not given, refers from the training args",
        )
        group.add_argument(
            "--ctc_greedy",
            type=str2bool,
            default=False,
        )

        group.add_argument(
            "--sim_chunk_length",
            type=int,
            default=0,
            help="The length of one chunk, to which speech will be "
            "divided for evalution of streaming processing.",
        )
        group.add_argument("--disable_repetition_detection", type=str2bool, default=False)
        group.add_argument(
            "--encoded_feat_length_limit",
            type=int,
            default=0,
            help="Limit the lengths of the encoded feature" "to input to the decoder.",
        )
        group.add_argument(
            "--decoder_text_length_limit",
            type=int,
            default=0,
            help="Limit the lengths of the text" "to input to the decoder.",
        )

        group.add_argument(
            "--backend",
            type=str,
            default="offline",
            help="Limit the lengths of the text" "to input to the decoder.",
        )
        group.add_argument(
            "--time_sync",
            type=str2bool,
            default=False,
        )
        group.add_argument(
            "--incremental_decode",
            type=str2bool,
            default=False,
        )
        group.add_argument(
            "--blank_penalty",
            type=float,
            default=1.0,
        )
        group.add_argument(
            "--hold_n",
            type=int,
            default=0,
        )
        group.add_argument(
            "--token_delay",
            type=str2bool,
            default=True,
        )
        group.add_argument(
            "--rnnt",
            type=str2bool,
            default=False,
        )
        group.add_argument(
            "--lang",
            type=str,
            default="de",
        )
        group.add_argument("--hugging_face_decoder", type=str2bool, default=False)
        group.add_argument(
            "--recompute",
            type=str2bool,
            default=False,
        )
        group.add_argument(
            "--use_word_list",
            type=str2bool,
            default=False,
        )
        group = parser.add_argument_group("The TTS model configuration related")
        group.add_argument("--tts_model", type=str, required=True)
        group.add_argument("--vocoder", type=str_or_none, default=None)
        group.add_argument("--tts_threshold", type=float, default=0.5) # for tacotron/transformer-based AR
        group.add_argument("--tts_minlenratio", type=float, default=0.0) # for tacotron/transformer-based AR
        group.add_argument("--tts_maxlenratio", type=float, default=10.0) # for tacotron/transformer-based AR
        group.add_argument("--tts_use_att_constraint", type=str2bool, default=False) # for tacotron/transformer-based AR
        group.add_argument("--tts_backward_window", type=int, default=1) # for tacotron/transformer-based AR
        group.add_argument("--tts_forward_window", type=int, default=3) # for tacotron/transformer-based AR
        group.add_argument("--tts_speed_control_alpha", type=float, default=1.0) # for NAR
        group.add_argument("--tts_noise_scale", type=float, default=0.333) # for VITS
        group.add_argument("--tts_noise_scale_dur", type=float, default=0.333) # for VITS
        group.add_argument("--tts_sampling_rate", type=int, default=16000)
        return parser

    def clean(self):
        self.processed_index = -1
        self.maxlen = 0
        self.prev_prediction = ""
        self.prev_ret = ""

    def policy(self):

        # dummy offline policy
        if self.backend == 'offline':
            if self.states.source_finished:
                results = self.speech2text(torch.tensor(self.states.source))
                if self.speech2text.st_model.use_multidecoder:
                    prediction = results[0][0][0] # multidecoder result is in this format
                else:
                    prediction = results[0][0]

                try:
                    normalize_prediction = text_normalizer(prediction)
                except:
                    normalize_prediction = simple_text_normalizer(prediction)
                if len(normalize_prediction) < 2:
                    samples = []
                else:
                    samples = self.text2speech(normalize_prediction)["wav"].tolist()
                return WriteAction(
                    SpeechSegment(
                        content=samples,
                        sample_rate=self.fs,
                        finished=True,
                    ),
                    finished=True,
                )
            else:
                return ReadAction()
        
        # hacked streaming policy. takes running beam search hyp as an incremental output 
        else:
            unread_length = len(self.states.source) - self.processed_index - 1
            if unread_length >= self.sim_chunk_length or self.states.source_finished:
                if self.recompute:
                    speech = torch.tensor(self.states.source)
                else:
                    speech = torch.tensor(self.states.source[self.processed_index+1:])
                
                try:
                    results = self.speech2text(speech=speech, is_final=self.states.source_finished)
                except TooShortUttError:
                    print("skipping inference for too short input")
                    results = [[""]]

                self.processed_index = len(self.states.source) - 1
                # if len(results) > 0:
                #     logging.info("results: {}".format(results[0][0]))
                # else:
                #     logging.info("empty results")
                # logging.info("self.stats.source_finished: {}".format(self.states.source_finished))
                if not self.states.source_finished:
                    if len(results) > 0:
                        prediction = results[0][0]                  
                    elif self.speech2text.beam_search.running_hyps and len(self.speech2text.beam_search.running_hyps.yseq[0]) > self.maxlen:
                        prediction = self.speech2text.beam_search.running_hyps.yseq[0][1:]
                        prediction = self.speech2text.converter.ids2tokens(prediction)
                        prediction = self.speech2text.tokenizer.tokens2text(prediction)
                        self.maxlen = len(self.speech2text.beam_search.running_hyps.yseq[0])
                    else:
                        # logging.info("return early")
                        return ReadAction()
                else:
                    if len(results) > 0:
                        prediction = results[0][0]
                    else:
                        prediction = self.prev_prediction

                if prediction != self.prev_prediction or self.states.source_finished:
                    self.prev_prediction = prediction
                    prediction = MosesDetokenizer(self.lang)(prediction.split(" "))
                    
                    if self.token_delay and not self.states.source_finished:
                        prediction_split = prediction.rsplit(" ",1)
                        if self.word_list is not None and prediction_split[-1] in self.word_list:
                            # no token delay
                            pass
                        elif prediction_split[-1][-1] in [",", "\"", "?", ".", ")"]:
                            # no token delay
                            pass
                        else:
                            # token delay
                            print("token delay occurred:", prediction_split[-1])
                            if len(prediction) == 1:
                                prediction = ""
                            else:
                                prediction = prediction_split[0]

                    # unwritten_length = len(prediction) - len("".join(self.states.target))
                    # logging.info("prediction: {}, self.prev_ret: {}".format(prediction, self.prev_ret))
                    unwritten_length = len(prediction) - len(self.prev_ret)
                else:
                    # logging.info("prediction equal prev prediction and state source finished false")
                    unwritten_length = 0

                if self.states.source_finished:
                    self.clean()
                
                logging.info("prediction: {}, self.prev_ret: {}".format(prediction, self.prev_ret))

                if unwritten_length > 0:
                    ret = prediction[-unwritten_length:]
                    print(self.processed_index, ret)
                    self.prev_ret += ret
                    if self.states.source_finished:
                        self.clean()

                    # logging.info("input for tts: {}".format(ret))
                    
                    try:
                        normalize_prediction = text_normalizer(ret)
                    except:
                        normalize_prediction = simple_text_normalizer(ret)
                    if len(normalize_prediction) < 2:
                        samples = []
                    else:
                        samples = self.text2speech(normalize_prediction)["wav"].tolist()
                    logging.info("output samples of length: {}".format(len(samples)))
                    return WriteAction(
                        SpeechSegment(
                            content=samples,
                            sample_rate=self.fs,
                            finished=self.states.source_finished,
                        ),
                        finished=self.states.source_finished,
                    )
                elif self.states.source_finished:
                    logging.info("output samples of length: 0 and finished")
                    return WriteAction(
                        SpeechSegment(
                            content=[],
                            sample_rate=self.fs,
                            finished=self.states.source_finished,
                        ),
                        finished=self.states.source_finished,
                    )

            return ReadAction()