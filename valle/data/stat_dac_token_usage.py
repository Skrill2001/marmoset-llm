from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union
import typing
from pathlib import Path
import os
import h5py

import numpy as np
import torch
from lhotse.features import FeatureExtractor
from lhotse.utils import Seconds, compute_num_frames
from dac import DAC
from audiotools import AudioSignal

import os
import torch
from collections import Counter
from tqdm import tqdm


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

if __name__ == "__main__":

    # 设置路径
    wav_dir = "/cpfs02/user/housiyuan/dataset/monkey/merge_audio_no_s"
    log_path = "log/log_token.txt"
    tokenizer = DACAudioTokenizer(model_path="dac/ckpt/weights.pth")

    # 初始化 token 计数器
    token_counter = Counter()
    total_tokens = 0

    # 遍历目录中的所有wav文件
    for filename in tqdm(os.listdir(wav_dir)):
        if not filename.endswith(".wav"):
            continue

        filepath = os.path.join(wav_dir, filename)

        try:
            # 加载和编码音频
            audio = tokenizer.process_data(filepath)  # 输出 shape: [1, T]
            codes = tokenizer.encode(audio)          # 输出 shape: [1, 9, L]
            codes = codes[:, 0, :]                   # 取 codebook 0，shape: [1, L]
            codes = codes.squeeze(0)                 # shape: [L]

            # 更新统计信息
            token_counter.update(codes.cpu().tolist())
            total_tokens += codes.numel()

        except Exception as e:
            print(f"[Error] Processing {filename} failed: {e}")

    # 计算使用率
    usage = {i: token_counter[i] / total_tokens for i in range(1024)}

    # 排序：按使用率从高到低
    sorted_usage = sorted(usage.items(), key=lambda x: x[1], reverse=True)

    # 保存到文件
    with open(log_path, "w") as f:
        f.write("Token_ID\tUsage_Rate\tCount\n")
        for token_id, rate in sorted_usage:
            count = token_counter[token_id]
            f.write(f"{token_id}\t{rate:.8f}\t{count}\n")

    print(f"Token usage saved to {log_path}")

