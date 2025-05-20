from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union
import typing
from pathlib import Path
import os

import numpy as np
import torch
from lhotse.features import FeatureExtractor
from lhotse.utils import Seconds, compute_num_frames
from dac import DAC
from audiotools import AudioSignal


class DACAudioTokenizer:

    def __init__(self, model_path: str, device: str = "cuda") -> None:

        dac_model = DAC.load(model_path)
        self.device = torch.device(device)
        self.model = dac_model.to(device)
        self.sample_rate = self.model.sample_rate
        self.channels = 1

    def process_data(self, wav: typing.Union[torch.Tensor, str, Path, np.ndarray]):

        signal = AudioSignal(wav, self.sample_rate)
        signal = signal.resample(self.sample_rate)
        signal = signal.to(self.device)
        return self.model.preprocess(signal.audio_data, signal.sample_rate)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # input:  x: [B, 1, T]
        # output: token index tensor: [B, N, T'], N is n_codebook
        assert x.ndim == 3 and x.shape[1] == 1, f"Input shape must be [B, 1, T], got {x.shape}"
        _, codes, *_ = self.model.encode(x)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:

        assert codes.ndim == 3 and codes.shape[1] == self.model.n_codebooks
        z = self.model.quantizer.from_codes(codes)[0]
        recon = self.model.decode(z)
        return recon


def tokenize_audio(tokenizer: DACAudioTokenizer, audio_path: str, sample_rate: int = 48000):

    # Extract discrete codes from dac
    audio = tokenizer.process_data(audio_path)
    with torch.no_grad():
        encoded_frames = tokenizer.encode(audio)
    return encoded_frames


@dataclass
class DACAudioTokenConfig:
    frame_shift: Seconds = 512.0 / 48000
    num_quantizers: int = 9

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DACAudioTokenConfig":
        return DACAudioTokenConfig(**data)


class DACAudioTokenExtractor(FeatureExtractor):
    name = "dac"
    config_type = DACAudioTokenConfig

    def __init__(self, config: Optional[Any] = None, model_path: str = "ckpt/dac/weights.pth"):
        super(DACAudioTokenExtractor, self).__init__(config)
        self.tokenizer = DACAudioTokenizer(model_path = model_path)

    def extract(
        self, samples: Union[np.ndarray, torch.Tensor], sampling_rate: int
    ) -> np.ndarray:
        
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
        
        with torch.no_grad():
            # batch_size = 1
            samples = self.tokenizer.process_data(samples)
            codes = self.tokenizer.encode(samples.detach())

        if True:
            duration = round(samples.shape[-1] / sampling_rate, ndigits=12)
            expected_num_frames = compute_num_frames(
                duration=duration,
                frame_shift=self.frame_shift,
                sampling_rate=sampling_rate,
            )
            assert abs(codes.shape[-1] - expected_num_frames) <= 1
            codes = codes[..., :expected_num_frames]
        return codes.cpu().squeeze(0).permute(1, 0).numpy()

    @property
    def frame_shift(self) -> Seconds:
        return self.config.frame_shift

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_quantizers

    def pad_tensor_list(self, tensor_list, device, padding_value=0):
        # 计算每个张量的长度
        lengths = [tensor.shape[0] for tensor in tensor_list]
        # 使用pad_sequence函数进行填充
        tensor_list = [torch.Tensor(t).to(device) for t in tensor_list]
        padded_tensor = torch.nn.utils.rnn.pad_sequence(
            tensor_list, batch_first=True, padding_value=padding_value
        )
        return padded_tensor, lengths

    def extract_batch(self, samples, sampling_rate, lengths) -> np.ndarray:
        samples = [wav.squeeze() for wav in samples]
        device = self.tokenizer.device
        samples, lengths = self.pad_tensor_list(samples, device)
        samples = samples.unsqueeze(1)

        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)

        if len(samples.shape) != 3:
            raise ValueError()
        
        samples = [self.tokenizer.process_data(wav)  for wav in samples]
        samples = torch.stack(samples, 0) # convert samples from list to tensor
        if samples.ndim == 4:
            samples = samples.squeeze(1) 
        
        # Extract discrete codes from dac
        with torch.no_grad():
            encoded_frames = self.tokenizer.encode(samples.detach().to(device))

        batch_codes = []
        for b, length in enumerate(lengths):
            codes = encoded_frames[b]
            duration = round(length / sampling_rate, ndigits=12)
            expected_num_frames = compute_num_frames(
                duration=duration,
                frame_shift=self.frame_shift,
                sampling_rate=sampling_rate,
            )
            batch_codes.append(codes[..., :expected_num_frames])
        return [codes.cpu().permute(1, 0).numpy() for codes in batch_codes]


if __name__ == "__main__":

    audio_path = "marmoset/prompts/test_48k.wav"
    output_path = "marmoset/prompts/output_48k.wav"
    tokenizer = DACAudioTokenizer(model_path="ckpt/dac/weights.pth")

    audio = tokenizer.process_data(audio_path)
    print("audio shape: ", audio.shape)
    
    codes = tokenizer.encode(audio)
    print("codex shape: ", codes.shape)
    
    recon = tokenizer.decode(codes)
    print("recon shape: ", recon.shape)

    y = AudioSignal(recon.cpu().detach().numpy(), 48000)
    y.write(output_path)
