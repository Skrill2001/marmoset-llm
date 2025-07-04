#!/usr/bin/env python3
# Copyright    2023                            (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Phonemize Text and EnCodec Audio.

Usage example:
    python3 bin/infer.py \
        --decoder-dim 128 --nhead 4 --num-decoder-layers 4 --model-name valle \
        --text-prompts "Go to her." \
        --audio-prompts ./prompts/61_70970_000007_000001.wav \
        --output-dir infer/demo_valle_epoch20 \
        --checkpoint exp/valle_nano_v2/epoch-20.pt

"""
import argparse
import logging
import os
from pathlib import Path

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import torchaudio
from icefall.utils import AttributeDict, str2bool

from valle.data import (
    DACAudioTokenizer,
    tokenize_audio_dac,
)
from valle.models import get_model
from valle.models.macros import ENCODED_FRAME_RATE

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio-prompts",
        type=str,
        default="",
        help="Audio prompts path",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="exp/valle/best-valid-loss.pt",
        help="Path to the saved checkpoint.",
    )

    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="../dac/ckpt/weights.pth",
        help="Path to the tokenizer checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("infer/demo"),
        help="Path to the tokenized files.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=-100,
        help="Whether AR Decoder do top_k(if > 0) sampling.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.2,
        help="The temperature of AR Decoder top_k sampling.",
    )

    parser.add_argument(
        "--continual",
        type=str2bool,
        default=False,
        help="Do continual task.",
    )

    parser.add_argument(
        "--prompt-segment",
        type=str2bool,
        default=True,
        help="Select a segment from the beginning of the audio prompt as the prompt.",
    )

    return parser.parse_args()


def load_model(checkpoint, device):
    if not checkpoint:
        return None

    checkpoint = torch.load(checkpoint, map_location=device)

    args = AttributeDict(checkpoint)
    model = get_model(args)

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model"], strict=False
    )
    assert (not missing_keys) or (missing_keys == ["ar_length_embedding.word_embeddings.weight"])
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def main():
    args = get_args()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    model = load_model(args.checkpoint, device)

    audio_tokenizer = DACAudioTokenizer(model_path=args.tokenizer_path)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.audio_prompts:
        for n, audio_file in enumerate(args.audio_prompts.split("|")):

            audio_name = os.path.basename(audio_file).split('.')[0]
            encoded_frames = tokenize_audio_dac(audio_tokenizer, audio_file)
            print("encoded_frames shape:", encoded_frames.shape)

            # 截取最开始的 一段 作为prompt
            if args.prompt_segment:
                y_len= encoded_frames.shape[2]
                int_low = int(0.25 * y_len)
                prefix_len = torch.randint(int_low, int_low * 2, size=()).item()
                prefix_len = min(prefix_len, int(ENCODED_FRAME_RATE * 3))  # 48000/512 * 3s = 281.25 frames
                encoded_frames = encoded_frames[:, :, :prefix_len]
                print(f"Clip a segment of length {prefix_len} as prompt.")

            audio_prompts = encoded_frames.transpose(2, 1).to(device)
            print("audio prompt shape: ", audio_prompts.shape)

            if False:
                samples = audio_tokenizer.decode(encoded_frames)
                torchaudio.save(
                    f"{args.output_dir}/p{n}.wav", samples[0], 48000
                )
            
            # synthesis
            if args.continual:
                encoded_frames = model.continual(audio_prompts)
            else:
                encoded_frames = model.inference(
                    audio_prompts,
                    top_k=args.top_k,
                    temperature=args.temperature,
                )

            assert encoded_frames.ndim == 3 and encoded_frames.shape[2] == model.num_quantizers
            samples = audio_tokenizer.decode(encoded_frames.transpose(2, 1))
            # store
            torchaudio.save(f"{args.output_dir}/{audio_name}_output.wav", samples[0].cpu(), 48000)
    else:
        print("Please provide --audio-prompts")


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)
if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
