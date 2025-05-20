import os
import dac
from audiotools import AudioSignal

# input_path = "./resources/input_audio_48k.wav"
input_path = "./resources/test_48k.wav"
# model_path = "./ckpt/weights_24khz.pth"
model_path = "./runs/baseline_48k/best/dac/weights.pth"
save_dir = "./resources"

model = dac.DAC.load(model_path)
model.to('cuda')

signal = AudioSignal(input_path)
signal = signal.resample(model.sample_rate)
signal.to(model.device)

x = model.preprocess(signal.audio_data, signal.sample_rate)
z, codes, latents, _, _ = model.encode(x)
y = model.decode(z)
recons = AudioSignal(y.cpu().detach().numpy(), 48000)
recons.write(os.path.join(save_dir, 'output_baseline_48k_best_test_c.wav'))

signal = signal.cpu()
x = model.compress(signal)

x.save("./resources/compressed.dac")
x = dac.DACFile.load("./resources/compressed.dac")

y = model.decompress(x)
y.write(os.path.join(save_dir, 'output_baseline_48k_best_test.wav'))