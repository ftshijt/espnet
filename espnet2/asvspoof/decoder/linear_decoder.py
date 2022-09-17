import torch

from typing import Optional
from espnet2.asvspoof.decoder.abs_decoder import AbsDecoder


class LinearDecoder(AbsDecoder):
    """Linear decoder for speaker diarization"""

    def __init__(
        self,
        encoder_output_size: int,
    ):
        super().__init__()
        self.linear_decoder = torch.nn.Linear(encoder_output_size, 1)

    def forward(self, input: torch.Tensor, ilens: Optional[torch.Tensor]):
        """Forward.
        Args:
            input (torch.Tensor): hidden_space [Batch, T, F]
            ilens (torch.Tensor): input lengths [Batch]
        """
        input = torch.mean(input, dim=1)
        output = self.linear_decoder(input)

        return output
