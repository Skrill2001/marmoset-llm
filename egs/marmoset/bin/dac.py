import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Pattern, Union

import numpy as np
import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from lhotse.features import FeatureExtractor
from lhotse.utils import Seconds, compute_num_frames
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator
from ....dac import *
from audiotools import AudioSignal

try:
    from pypinyin import Style, pinyin
    from pypinyin.style._utils import get_finals, get_initials
except Exception:
    pass


def remove_encodec_weight_norm(model):
    from encodec.modules import SConv1d
    from encodec.modules.seanet import SConvTranspose1d, SEANetResnetBlock
    from torch.nn.utils import remove_weight_norm

    encoder = model.encoder.model
    for key in encoder._modules:
        if isinstance(encoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(encoder._modules[key].shortcut.conv.conv)
            block_modules = encoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(encoder._modules[key], SConv1d):
            remove_weight_norm(encoder._modules[key].conv.conv)

    decoder = model.decoder.model
    for key in decoder._modules:
        if isinstance(decoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(decoder._modules[key].shortcut.conv.conv)
            block_modules = decoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(decoder._modules[key], SConvTranspose1d):
            remove_weight_norm(decoder._modules[key].convtr.convtr)
        elif isinstance(decoder._modules[key], SConv1d):
            remove_weight_norm(decoder._modules[key].conv.conv)


class DACAudioTokenizer:

    def __init__(self, model_path: str, device: str = "cuda") -> None:

        dac_model = DAC.load(model_path)
        self.device = torch.device(device)
        self.model = dac_model.to(device)
        self.sample_rate = self.model.sample_rate
        self.channels = 1

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        # input:    wav: [B, 1, T]
        # output: token index tensor: [B, N, T'], N is n_codebook
        _, codes, *_ = self.model.encode(wav)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:

        assert codes.ndim == 3 and codes.shape[1] == self.model.n_codebooks
        z = self.model.quantizer.from_codes(codes)[0]
        recon = self.decode(z)
        return recon


def tokenize_audio(tokenizer: DACAudioTokenizer, audio_path: str, sample_rate: int = 48000):

    # Load and pre-process the audio waveform
    signal = AudioSignal(audio_path)
    signal = signal.resample(sample_rate)
    signal.to(tokenizer.device)
    wav = tokenizer.model.preprocess(signal.audio_data, signal.sample_rate)

    # Extract discrete codes from dac
    with torch.no_grad():
        encoded_frames = tokenizer.encode(wav)
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
        if sampling_rate != self.tokenizer.sample_rate:
            samples = convert_audio(
                samples,
                sampling_rate,
                self.tokenizer.sample_rate,
                self.tokenizer.channels,
            )
        if len(samples.shape) == 2:
            samples = samples.unsqueeze(0)
        else:
            raise ValueError()

        device = self.tokenizer.device
        encoded_frames = self.tokenizer.encode(samples.detach().to(device))
        codes = encoded_frames[0][0]  # [B, n_q, T]
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
        if sampling_rate != self.tokenizer.sample_rate:
            samples = [
                convert_audio(
                    wav,
                    sampling_rate,
                    self.tokenizer.sample_rate,
                    self.tokenizer.channels,
                )
                for wav in samples
            ]
            samples = torch.stack(samples, 0) # convert samples from list to tensor
        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = self.tokenizer.encode(samples.detach().to(device))
        encoded_frames = encoded_frames[0][0]  # [B, n_q, T]
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
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)

    samples = torch.from_numpy(np.random.random([4, 1, 1600])).type(
        torch.float32
    )
    codes_raw = model.encode(samples)

    remove_encodec_weight_norm(model)
    codes_norm = model.encode(samples)

    assert torch.allclose(codes_raw[0][0], codes_norm[0][0])
