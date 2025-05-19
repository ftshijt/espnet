# Copyright 2025 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Speech Enhancement Argumented DAC."""
import copy
import functools
import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typeguard import typechecked

from espnet2.bin.enh_inference import SeparateSpeech
from espnet2.gan_codec.abs_gan_codec import AbsGANCodec
from espnet2.gan_codec.dac.dac import DACDiscriminator
from espnet2.gan_codec.shared.decoder.seanet import SEANetDecoder
from espnet2.gan_codec.shared.encoder.seanet import SEANetEncoder
from espnet2.gan_codec.shared.loss.freq_loss import MultiScaleMelSpectrogramLoss
from espnet2.gan_codec.shared.quantizer.residual_vq import ResidualVectorQuantizer
from espnet2.gan_tts.hifigan.loss import (
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    GeneratorAdversarialLoss,
)
from espnet2.torch_utils.device_funcs import force_gatherable

logger = logging.getLogger(__name__)


class SEDAC2(AbsGANCodec):
    """Speech Enhancement Argumented DAC."""

    @typechecked
    def __init__(
        self,
        sampling_rate: int = 24000,
        generator_params: Dict[str, Any] = {
            "hidden_dim": 128,
            "se_model_source": "espnet",
            "se_model_tag": "wyz/tfgridnet_for_urgent24",
            "se_model_path": None,
            "se_model_config": None,
            "enhanced_n_streams": 1,
            "encdec_channels": 1,
            "encdec_n_filters": 32,
            "encdec_n_residual_layers": 1,
            "encdec_ratios": [8, 6, 5, 2],
            "encdec_activation": "Snake",
            "encdec_activation_params": {},
            "encdec_norm": "weight_norm",
            "encdec_norm_params": {},
            "encdec_kernel_size": 7,
            "encdec_residual_kernel_size": 7,
            "encdec_last_kernel_size": 7,
            "encdec_dilation_base": 2,
            "encdec_causal": False,
            "encdec_pad_mode": "reflect",
            "encdec_true_skip": False,
            "encdec_compress": 2,
            "encdec_lstm": 2,
            "decoder_trim_right_ratio": 1.0,
            "decoder_final_activation": None,
            "decoder_final_activation_params": None,
            "quantizer_n_q": 8,
            "quantizer_bins": 1024,
            "quantizer_decay": 0.99,
            "quantizer_kmeans_init": True,
            "quantizer_kmeans_iters": 50,
            "quantizer_threshold_ema_dead_code": 2,
            "quantizer_target_bandwidth": [7.5, 15],
            "quantizer_dropout": True,
        },
        discriminator_params: Dict[str, Any] = {
            "scale_follow_official_norm": False,
            "msmpmb_discriminator_params": {
                "rates": [],
                "periods": [2, 3, 5, 7, 11],
                "fft_sizes": [2048, 1024, 512],
                "sample_rate": 24000,
                "period_discriminator_params": {
                    "in_channels": 1,
                    "out_channels": 1,
                    "kernel_sizes": [5, 3],
                    "channels": 32,
                    "downsample_scales": [3, 3, 3, 3, 1],
                    "max_downsample_channels": 1024,
                    "bias": True,
                    "nonlinear_activation": "LeakyReLU",
                    "nonlinear_activation_params": {"negative_slope": 0.1},
                    "use_weight_norm": True,
                    "use_spectral_norm": False,
                },
                "band_discriminator_params": {
                    "hop_factor": 0.25,
                    "sample_rate": 24000,
                    "bands": [
                        (0.0, 0.1),
                        (0.1, 0.25),
                        (0.25, 0.5),
                        (0.5, 0.75),
                        (0.75, 1.0),
                    ],
                    "channel": 32,
                },
            },
        },
        # loss related
        generator_adv_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
            "loss_type": "mse",
        },
        discriminator_adv_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
            "loss_type": "mse",
        },
        use_feat_match_loss: bool = True,
        feat_match_loss_params: Dict[str, Any] = {
            "average_by_discriminators": False,
            "average_by_layers": False,
            "include_final_outputs": True,
        },
        use_mel_loss: bool = True,
        mel_loss_params: Dict[str, Any] = {
            "fs": 24000,
            "range_start": 6,
            "range_end": 11,
            "window": "hann",
            "n_mels": 80,
            "fmin": 0,
            "fmax": None,
            "log_base": None,
        },
        use_dual_decoder: bool = True,
        skip_quantizer_updates: int = 0,
        activate_enh: int = 0,
        enhanced_prob: float = 0.5,
        lambda_quantization: float = 1.0,
        lambda_reconstruct: float = 1.0,
        lambda_commit: float = 1.0,
        lambda_adv: float = 1.0,
        lambda_feat_match: float = 2.0,
        lambda_mel: float = 45.0,
        cache_generator_outputs: bool = False,
        inference_only: bool = False,
    ):
        """Initialize SEDAC model.
        Args:
            sampling_rate: The sample rate of the input audio.
            generator_params: Parameters for the generator model.
            discriminator_params: Parameters for the discriminator model.
            generator_adv_loss_params: Parameters for generator adversarial loss.
            discriminator_adv_loss_params:
                Parameters for discriminator adversarial loss.
            use_feat_match_loss: Whether to use feature matching loss.
            feat_match_loss_params: Parameters for feature matching loss.
            use_mel_loss: Whether to use mel-spectrogram loss.
            mel_loss_params: Parameters for mel-spectrogram loss.
            use_dual_decoder: Whether to use dual decoder mode.
            skip_quantizer_updates (int): Number of updates to skip quantizer training.
            activate_enh: Num of updates to start activate enh
            enhanced_prob: enhancement probability
            lambda_quantization: Weight for quantization loss.
            lambda_reconstruct: Weight for reconstruction loss.
            lambda_commit: Weight for commitment loss.
            lambda_adv: Weight for adversarial loss.
            lambda_feat_match: Weight for feature matching loss.
            lambda_mel: Weight for mel-spectrogram loss.
            cache_generator_outputs: Whether to cache generator outputs.
        """
        super().__init__()

        # Update sample rate for all components
        generator_params["sample_rate"] = sampling_rate
        generator_params["inference_only"] = inference_only

        if "sample_rate" in discriminator_params.get("msmpmb_discriminator_params", {}):
            discriminator_params["msmpmb_discriminator_params"][
                "sample_rate"
            ] = sampling_rate

        if use_mel_loss:
            mel_loss_params["fs"] = sampling_rate

        # Define modules
        self.generator = SEDAC2Generator(**generator_params)
        self.discriminator = DACDiscriminator(**discriminator_params)
        self.enhanced_discriminator = DACDiscriminator(**discriminator_params)

        # Define loss functions
        self.generator_adv_loss = GeneratorAdversarialLoss(**generator_adv_loss_params)
        self.generator_reconstruct_loss = nn.L1Loss(reduction="mean")
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss(
            **discriminator_adv_loss_params
        )

        # Optional losses
        self.use_feat_match_loss = use_feat_match_loss
        if self.use_feat_match_loss:
            self.feat_match_loss = FeatureMatchLoss(**feat_match_loss_params)

        self.use_mel_loss = use_mel_loss
        if self.use_mel_loss:
            self.mel_loss = MultiScaleMelSpectrogramLoss(**mel_loss_params)

        # Handle dual decoder mode
        self.use_dual_decoder = use_dual_decoder
        if self.use_dual_decoder and not self.use_mel_loss:
            logger.warning(
                "Dual decoder is enabled but Mel loss is disabled."
                "This configuration is ineffective."
            )
            self.use_dual_decoder = False

        # Training configuration
        self.skip_quantizer_updates = skip_quantizer_updates
        self.activate_enh = activate_enh
        self.register_buffer("num_updates", torch.zeros(1, dtype=torch.float))

        # Loss coefficients
        self.lambda_quantization = lambda_quantization
        self.lambda_reconstruct = lambda_reconstruct
        self.lambda_commit = lambda_commit
        self.lambda_adv = lambda_adv
        self.lambda_feat_match = lambda_feat_match if use_feat_match_loss else 0.0
        self.lambda_mel = lambda_mel if use_mel_loss else 0.0
        self.enhanced_prob = enhanced_prob

        # Cache settings
        self.cache_generator_outputs = cache_generator_outputs
        self._cache = None

        # Meta information
        self.fs = sampling_rate
        self.num_streams = generator_params.get("quantizer_n_q", 1)
        self.frame_shift = functools.reduce(
            lambda x, y: x * y, generator_params.get("encdec_ratios", [2, 2])
        )
        self.code_size_per_stream = [
            generator_params.get("quantizer_bins", 32)
        ] * self.num_streams

    def meta_info(self) -> Dict[str, Any]:
        """Return model meta information.
        Returns:
            Dict with model information.
        """
        return {
            "fs": self.fs,
            "num_streams": self.num_streams,
            "frame_shift": self.frame_shift,
            "code_size_per_stream": self.code_size_per_stream,
        }

    def forward(
        self,
        audio: torch.Tensor,
        forward_generator: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform forward pass.
        Args:
            audio: Audio waveform tensor (B, T_wav).
            forward_generator: Whether to forward generator.
        Returns:
            Dict with loss, stats, weight, and optimizer index.
        """
        if forward_generator:
            return self._forward_generator(audio=audio, **kwargs)
        else:
            return self._forward_discriminator(audio=audio, **kwargs)

    def _forward_generator(
        self,
        audio: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform generator forward pass.
        Args:
            audio: Audio waveform tensor (B, T_wav).
        Returns:
            Dict with loss, stats, weight, and optimizer index.
        """
        # Setup
        batch_size = audio.size(0)

        # Add channel dimension if needed
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        # Determine which audio reconstruction to use based on training phase
        is_quantizer_active = self.skip_quantizer_updates <= self.num_updates
        activate_enh = is_quantizer_active and (self.activate_enh <= self.num_updates)

        # Calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            outputs = self.generator(audio, use_dual_decoder=self.use_dual_decoder, use_enh=activate_enh)
            (
                audio_hat,
                codec_commit_loss,
                quantization_loss,
                enhanced_quantization_loss,
                enhanced_audio_hat,
                enhanced_audio,
                audio_hat_real,
            ) = outputs
        else:
            (
                audio_hat,
                codec_commit_loss,
                quantization_loss,
                enhanced_quantization_loss,
                enhanced_audio_hat,
                enhanced_audio,
                audio_hat_real,
            ) = self._cache

        # Store cache
        if self.training and self.cache_generator_outputs and not reuse_cache:
            self._cache = (
                audio_hat,
                codec_commit_loss,
                quantization_loss,
                enhanced_quantization_loss,
                enhanced_audio_hat,
                enhanced_audio,
                audio_hat_real,
            )

        target_audio = (
            audio_hat
            if (is_quantizer_active and self.use_dual_decoder)
            else audio_hat_real
        )

        # Calculate discriminator outputs
        p_hat = self.discriminator(target_audio)
        with torch.no_grad():
            # Do not store discriminator gradient in generator turn
            p = self.discriminator(audio)

        if is_quantizer_active:
            enhanced_p_hat = self.enhanced_discriminator(enhanced_audio_hat)
            with torch.no_grad():
                enhanced_p = self.enhanced_discriminator(enhanced_audio)

        # Calculate losses
        adv_loss = self.generator_adv_loss(p_hat) * self.lambda_adv
        codec_commit_loss = codec_commit_loss * self.lambda_commit
        codec_quantization_loss = quantization_loss * self.lambda_quantization
        reconstruct_loss = (
            self.generator_reconstruct_loss(audio, target_audio) * self.lambda_reconstruct
        )

        if is_quantizer_active:
            enhanced_adv_loss = self.generator_adv_loss(enhanced_p_hat) * self.lambda_adv
            codec_enhanced_quantization_loss = enhanced_quantization_loss * self.lambda_quantization
            enhanced_reconstruct_loss = (
            	self.generator_reconstruct_loss(enhanced_audio, enhanced_audio_hat) * self.lambda_reconstruct
            )

        if is_quantizer_active:
            if random.random() < self.enhanced_prob:
                loss = codec_commit_loss + enhanced_adv_loss + enhanced_quantization_loss + enhanced_reconstruct_loss
            else:
                loss = codec_commit_loss + codec_quantization_loss + adv_loss + reconstruct_loss
        else:
            codec_loss = codec_commit_loss + codec_quantization_loss
            loss = adv_loss + codec_loss + reconstruct_loss

        # Collect statistics
        stats = {
            "adv_loss": adv_loss.item(),
            "codec_commit_loss": codec_commit_loss.item(),
            "codec_quantization_loss": codec_quantization_loss.item(),
            "reconstruct_loss": reconstruct_loss.item(),
            "enhanced_adv_loss": enhanced_adv_loss.item() if is_quantizer_active else 0.0,
            "enhanced_quantization_loss":enhanced_quantization_loss.item() if is_quantizer_active else 0.0,
            "enhanced_reconstruct_loss": enhanced_reconstruct_loss.item() if is_quantizer_active else 0.0,
        }

        # Add feature matching loss if enabled
        if self.use_feat_match_loss:
            feat_match_loss = self.feat_match_loss(p_hat, p) * self.lambda_feat_match
            stats["feat_match_loss"] = feat_match_loss.item()

            if is_quantizer_active:
                enhanced_feat_match_loss = self.feat_match_loss(enhanced_p_hat, enhanced_p) * self.lambda_feat_match
                stats["enhanced_feat_match_loss"] = enhanced_feat_match_loss.item()

                if random.random() < self.enhanced_prob:
                    loss = loss + enhanced_feat_match_loss
                else:
                    loss = loss + feat_match_loss
            else:
                loss = loss + feat_match_loss

        # Add mel-spectrogram loss if enabled
        if self.use_mel_loss:
            mel_loss = self.mel_loss(audio_hat, audio) * self.lambda_mel
            stats["mel_loss"] = mel_loss.item()

            if is_quantizer_active:
                enhanced_mel_loss = self.mel_loss(enhanced_audio_hat, enhanced_audio) * self.lambda_mel
                stats["enhanced_mel_loss"] = enhanced_mel_loss.item()
                if random.random() < self.enhanced_prob:
                    loss = loss + enhanced_mel_loss
                else:
                    loss = loss + mel_loss
            else:
                loss = loss + mel_loss

            if self.use_dual_decoder and audio_hat_real is not None:
                mel_loss_real = self.mel_loss(audio_hat_real, audio) * self.lambda_mel
                stats["mel_loss_real"] = mel_loss_real.item()

        # Increment update counter
        self.num_updates += 1

        stats["loss"] = loss.item()

        # Check for NaN in all stats values and set to 0
        has_nan = False
        for key in stats:
            if torch.is_tensor(stats[key]):
                if torch.isnan(stats[key]).any():
                    stats[key] = 0.0
                    has_nan = True
            elif math.isnan(stats[key]):
                stats[key] = 0.0
                has_nan = True

        # If any stat had NaN, also set loss to zero
        if has_nan or torch.isnan(loss).any():
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
            stats["loss"] = 0.0

        # Make values gatherable across devices
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        # Reset cache if needed
        if reuse_cache or not self.training:
            self._cache = None

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 0,  # Needed for trainer
        }

    def _forward_discriminator(
        self,
        audio: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform discriminator forward pass.
        Args:
            audio: Audio waveform tensor (B, T_wav).
        Returns:
            Dict with loss, stats, weight, and optimizer index.
        """
        # Setup
        batch_size = audio.size(0)

        # Add channel dimension if needed
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        # Determine which audio reconstruction to use based on training phase
        is_quantizer_active = self.skip_quantizer_updates <= self.num_updates
        activate_enh = is_quantizer_active and (self.activate_enh <= self.num_updates)

        # Calculate generator outputs
        reuse_cache = True
        if not self.cache_generator_outputs or self._cache is None:
            reuse_cache = False
            outputs = self.generator(audio, use_dual_decoder=self.use_dual_decoder, use_enh=activate_enh)
            (
                audio_hat,
                codec_commit_loss,
                quantization_loss,
                enhanced_quantization_loss,
                enhanced_audio_hat,
                enhanced_audio,
                audio_hat_real,
            ) = outputs
        else:
            (
                audio_hat,
                codec_commit_loss,
                quantization_loss,
                enhanced_quantization_loss,
                enhanced_audio_hat,
                enhanced_audio,
                audio_hat_real,
            ) = self._cache

        # Store cache
        if self.cache_generator_outputs and not reuse_cache:
            self._cache = (
                audio_hat,
                codec_commit_loss,
                quantization_loss,
                enhanced_quantization_loss,
                enhanced_audio_hat,
                enhanced_audio,
                audio_hat_real,
            )

        target_audio = (
            audio_hat
            if (is_quantizer_active and self.use_dual_decoder)
            else audio_hat_real
        )

        # Calculate discriminator outputs
        p_hat = self.discriminator(
            target_audio.detach()
        )  # Detach to avoid grad flow to generator
        p = self.discriminator(audio)

        if is_quantizer_active:
            enhanced_p_hat = self.enhanced_discriminator(
            	enhanced_audio_hat.detach()
            )  # Detach to avoid grad flow to generator
            enhanced_p = self.enhanced_discriminator(enhanced_audio)

        # Calculate losses
        real_loss, fake_loss = self.discriminator_adv_loss(p_hat, p)
        loss = real_loss + fake_loss
        if is_quantizer_active:
            enhanced_real_loss, enhanced_fake_loss = self.discriminator_adv_loss(enhanced_p_hat, enhanced_p)
            loss = loss + enhanced_real_loss + enhanced_fake_loss

        # Collect statistics
        stats = {
            "discriminator_loss": loss.item(),
            "real_loss": real_loss.item(),
            "fake_loss": fake_loss.item(),
            "enhanced_real_loss": enhanced_real_loss.item() if is_quantizer_active else 0.0,
            "enhanced_fake_loss": enhanced_fake_loss.item() if is_quantizer_active else 0.0,
        }

        # Check for NaN in all stats values and set to 0
        has_nan = False
        for key in stats:
            if torch.is_tensor(stats[key]):
                if torch.isnan(stats[key]).any():
                    stats[key] = 0.0
                    has_nan = True
            elif math.isnan(stats[key]):
                stats[key] = 0.0
                has_nan = True


        # Make values gatherable across devices
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)

        # Reset cache if needed
        if reuse_cache or not self.training:
            self._cache = None

        return {
            "loss": loss,
            "stats": stats,
            "weight": weight,
            "optim_idx": 1,  # Needed for trainer
        }

    def inference(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Run inference.
        Args:
            x: Input audio (T_wav,).
        Returns:
            Dict with generated waveform and neural codec.
        """
        codec = self.generator.encode(x)
        wav = self.generator.decode(codec)

        return {"wav": wav, "codec": codec}

    def encode(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Run encoding.
        Args:
            x: Input audio (T_wav,).
        Returns:
            Generated codes (T_code, N_stream).
        """
        return self.generator.encode(x)

    def decode(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Run decoding.
        Args:
            x: Input codes (T_code, N_stream).
        Returns:
            Generated waveform (T_wav,).
        """
        return self.generator.decode(x)


class SEDAC2Generator(nn.Module):
    """DAC generator module augmented with speech enhancement model."""

    @typechecked
    def __init__(
        self,
        sample_rate: int = 24000,
        hidden_dim: int = 128,
        se_model_source: str = "espnet",
        se_model_tag: str = "wyz/tfgridnet_for_urgent24",
        se_model_path: Optional[str] = None,
        se_model_config: Optional[str] = None,
        enhanced_n_streams: int = 1,
        codebook_dim: int = 8,
        encdec_channels: int = 1,
        encdec_n_filters: int = 32,
        encdec_n_residual_layers: int = 1,
        encdec_ratios: List[int] = [8, 5, 4, 2],
        encdec_activation: str = "Snake",
        encdec_activation_params: Dict[str, Any] = {},
        encdec_norm: str = "weight_norm",
        encdec_norm_params: Dict[str, Any] = {},
        encdec_kernel_size: int = 7,
        encdec_residual_kernel_size: int = 7,
        encdec_last_kernel_size: int = 7,
        encdec_dilation_base: int = 2,
        encdec_causal: bool = False,
        encdec_pad_mode: str = "reflect",
        encdec_true_skip: bool = False,
        encdec_compress: int = 2,
        encdec_lstm: int = 2,
        decoder_trim_right_ratio: float = 1.0,
        decoder_final_activation: Optional[str] = None,
        decoder_final_activation_params: Optional[dict] = None,
        quantizer_n_q: int = 8,
        quantizer_bins: int = 1024,
        quantizer_decay: float = 0.99,
        quantizer_kmeans_init: bool = True,
        quantizer_kmeans_iters: int = 50,
        quantizer_threshold_ema_dead_code: int = 2,
        quantizer_target_bandwidth: List[float] = [7.5, 15],
        quantizer_dropout: bool = True,
        inference_only: bool = False,
    ):
        """Initialize DAC Generator.
        Args:
            sample_rate: Sample rate of input audio.
            hidden_dim: Hidden dimension for models.
            se_model_source: Speech enhancement model source.
            se_model_tag: Speech enhancement model tag.
            se_model_path: Speech enhancement model path.
            se_model_config: Speech enhancement model configuration.
            enhanced_n_streams: Number of codec streams for enhancement.
            codebook_dim: Dimension of codebook.
            encdec_channels: Number of encoder/decoder channels.
            encdec_n_filters: Number of filters in encoder/decoder.
            encdec_n_residual_layers: Number of residual layers.
            encdec_ratios: Upsampling/downsampling ratios.
            encdec_activation: Activation function.
            encdec_activation_params: Parameters for activation function.
            encdec_norm: Normalization method.
            encdec_norm_params: Parameters for normalization method.
            encdec_kernel_size: Kernel size.
            encdec_residual_kernel_size: Residual kernel size.
            encdec_last_kernel_size: Last kernel size.
            encdec_dilation_base: Base for dilation calculation.
            encdec_causal: Whether to use causal convolution.
            encdec_pad_mode: Padding mode.
            encdec_true_skip: Whether to use true skip connections.
            encdec_compress: Compression factor.
            encdec_lstm: Number of LSTM layers.
            decoder_trim_right_ratio: Trim ratio for decoder output.
            decoder_final_activation: Final activation function.
            decoder_final_activation_params: Parameters for final activation.
            quantizer_n_q: Number of quantizers.
            quantizer_bins: Number of bins per quantizer.
            quantizer_decay: Decay factor for EMA updates.
            quantizer_kmeans_init: Whether to initialize with k-means.
            quantizer_kmeans_iters: Number of k-means iterations.
            quantizer_threshold_ema_dead_code: Threshold for resetting dead codes.
            quantizer_target_bandwidth: Target bandwidth ranges.
            quantizer_dropout: Whether to use dropout in quantizer.
            inference_only: Whether to use for inference only cases
        """
        super().__init__()

        # Initialize encoder
        self.encoder = SEANetEncoder(
            channels=encdec_channels,
            dimension=hidden_dim,
            n_filters=encdec_n_filters,
            n_residual_layers=encdec_n_residual_layers,
            ratios=encdec_ratios,
            activation=encdec_activation,
            activation_params=encdec_activation_params,
            norm=encdec_norm,
            norm_params=encdec_norm_params,
            kernel_size=encdec_kernel_size,
            residual_kernel_size=encdec_residual_kernel_size,
            last_kernel_size=encdec_last_kernel_size,
            dilation_base=encdec_dilation_base,
            causal=encdec_causal,
            pad_mode=encdec_pad_mode,
            true_skip=encdec_true_skip,
            compress=encdec_compress,
            lstm=encdec_lstm,
        )

        # Initialize quantizer
        self.quantizer = ResidualVectorQuantizer(
            dimension=hidden_dim,
            codebook_dim=codebook_dim,
            n_q=quantizer_n_q,
            bins=quantizer_bins,
            decay=quantizer_decay,
            kmeans_init=quantizer_kmeans_init,
            kmeans_iters=quantizer_kmeans_iters,
            threshold_ema_dead_code=quantizer_threshold_ema_dead_code,
            quantizer_dropout=quantizer_dropout,
        )

        # Set model parameters
        self.target_bandwidths = quantizer_target_bandwidth
        self.sample_rate = sample_rate
        self.frame_rate = math.ceil(sample_rate / np.prod(encdec_ratios))
        self.enhanced_n_streams = enhanced_n_streams

        # Initialize decoder
        self.decoder = SEANetDecoder(
            channels=encdec_channels,
            dimension=hidden_dim,
            n_filters=encdec_n_filters,
            n_residual_layers=encdec_n_residual_layers,
            ratios=encdec_ratios,
            activation=encdec_activation,
            activation_params=encdec_activation_params,
            norm=encdec_norm,
            norm_params=encdec_norm_params,
            kernel_size=encdec_kernel_size,
            residual_kernel_size=encdec_residual_kernel_size,
            last_kernel_size=encdec_last_kernel_size,
            dilation_base=encdec_dilation_base,
            causal=encdec_causal,
            pad_mode=encdec_pad_mode,
            true_skip=encdec_true_skip,
            compress=encdec_compress,
            lstm=encdec_lstm,
            trim_right_ratio=decoder_trim_right_ratio,
            final_activation=decoder_final_activation,
            final_activation_params=decoder_final_activation_params,
        )


        # quantization loss
        self.l1_quantization_loss = torch.nn.L1Loss(reduction="mean")
        self.l2_quantization_loss = torch.nn.MSELoss(reduction="mean")

        if not inference_only:
            # Initialize speech enhancement model
            self.se_model_source = se_model_source
            self.se_model_tag = se_model_tag
            self.se_model_path = se_model_path
            self.se_model_config = se_model_config
            self._initialize_enhancement_model()

    def _initialize_enhancement_model(self) -> None:
        """Initialize enhancement model model."""
        if self.se_model_source == "espnet":
            # NOTE(jiatong): need a better way to organize device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.se_model_path is not None and self.se_model_config is not None:
                self.enhance_module = SeparateSpeech.from_pretrained(
                    train_config=se_model_config,
                    model_file=se_model_path,
                    normalize_output_wav=False,
                    device=device
                )
            else:
                self.enhance_module = SeparateSpeech.from_pretrained(
                    model_tag=self.se_model_tag,
                    normalize_output_wav=False,
                    device=device
                )

            # NOTE(jiatong): explicitly add enh_model to align the device
            self.enh_model = self.enhance_module.enh_model
        else:
            raise NotImplementedError(f"The enhancement model source {self.se_model_source} is not supported yet")

        # Freeze enhancement model parameters
        for param in self.enhance_module.enh_model.parameters():
            param.requires_grad = False

    def _extract_enhanced_speech(self, x: torch.Tensor) -> torch.Tensor:
        """Extract enhanced wav.
        Args:
            x: Input audio tensor.
        Returns:
            Enhanced wav.
        """
        with torch.no_grad():

            if self.se_model_source == "espnet":
                # TODO(jiatong): to add batch processing with inference
                #                note: initial trial is not successful
                batch_size = x.size(0)
                enhanced_wavs = [x.new(self.enhance_module(x[i], fs=self.sample_rate)[0]) for i in range(batch_size)]
                return torch.stack(enhanced_wavs, dim=0)
            else:
                raise ValueError("se model source not detected")

    def forward(
        self, x: torch.Tensor, use_dual_decoder: bool = False, use_enh: bool = True,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
    ]:
        """DAC forward propagation.
        Args:
            x: Input tensor of shape (B, 1, T).
            use_dual_decoder: Whether to use dual decoder for encoder out.
            use_enh: Whether to use enhancemet model for enhanced speech
        Returns:
            Tuple containing:
                - resynthesized audio
                - commitment loss
                - quantization loss
                - enhanced quantization loss
                - resynthesized audio from enhanced audio
                - enhanced audio
                - resynthesized audio from encoder (if dual decoder is used)
        """
        # Extract enhanced speech
        if use_enh:
            enhanced_x = self._extract_enhanced_speech(x)
        else:
            enhanced_x = x

        # Encode input
        encoder_out = self.encoder(x)
        enhanced_encoder_out = self.encoder(enhanced_x)

        # Select target bandwidth
        bw_idx = random.randint(0, len(self.target_bandwidths) - 1)
        bw = self.target_bandwidths[bw_idx]

        # Apply quantization
        quantized_list, _, _, commit_loss, _ = self.quantizer(
            encoder_out, self.frame_rate, bw, return_list=True
        )

        # print(self.quantizer.quantizer_dropout, [q.shape for q in quantized_list], flush=True)
        # Extract enhanced stream and final quantized output
        enhanced_stream = quantized_list[self.enhanced_n_streams - 1]
        quantized = quantized_list[-1]

        # Calculate quantization loss
        quantization_loss = self.l1_quantization_loss(
            encoder_out, quantized.detach()
        ) + self.l2_quantization_loss(encoder_out, quantized.detach())
        enhanced_quantization_loss = self.l1_quantization_loss(
            enhanced_encoder_out, enhanced_stream.detach()
        ) + self.l2_quantization_loss(enhanced_encoder_out, enhanced_stream.detach())

        # Decode quantized representation
        resyn_audio = self.decoder(quantized)
        resyn_audio_enhanced = self.decoder(enhanced_stream)

        # Optionally decode directly from encoder output
        resyn_audio_real = None
        if use_dual_decoder:
            resyn_audio_real = self.decoder(encoder_out)

        return (
            resyn_audio,
            commit_loss,
            quantization_loss,
            enhanced_quantization_loss,
            resyn_audio_enhanced,
            enhanced_x,
            resyn_audio_real,
        )

    def encode(
        self,
        x: torch.Tensor,
        target_bw: Optional[float] = None,
    ) -> torch.Tensor:
        """DAC codec encoding.
        Args:
            x: Input tensor of shape (B, 1, T) or (T,).
            target_bw: Target bandwidth.
        Returns:
            Neural codecs.
        """
        # Ensure input has correct dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension

        # Run through encoder
        encoder_out = self.encoder(x.float())

        # Select bandwidth
        if target_bw is None:
            bw = self.target_bandwidths[-1]  # Use maximum bandwidth by default
        else:
            bw = target_bw

        # Encode to discrete codes
        codes = self.quantizer.encode(encoder_out, self.frame_rate, bw)

        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """DAC codec decoding.
        Args:
            codes: Neural codecs.
        Returns:
            Resynthesized audio.
        """
        # Convert codes back to continuous representation
        quantized = self.quantizer.decode(codes)

        # Decode to audio
        resyn_audio = self.decoder(quantized)

        return resyn_audio
