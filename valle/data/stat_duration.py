import os
import librosa
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def analyze_wav_durations(wav_dir, output_log='duration_stats.txt', output_plot='duration_hist.png', sample_rate=48000):
    durations = []
    file_paths = []

    # 遍历目录及子目录下所有wav文件
    for root, _, files in os.walk(wav_dir):
        for file in tqdm(files):
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                try:
                    y, sr = librosa.load(path=file_path, sr=sample_rate)
                    duration = len(y)/sr
                    durations.append(duration)
                    file_paths.append(file_path)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    if not durations:
        print("No .wav files found or readable in the directory.")
        return

    total_files = len(durations)
    total_duration = sum(durations)
    average_duration = total_duration / total_files

    # 保存文本统计信息
    with open(output_log, 'w') as f:
        f.write("WAV Duration Stats\n")
        f.write("==================\n")
        f.write(f"Total files: {total_files}\n")
        f.write(f"Total duration: {total_duration:.2f} seconds\n")
        f.write(f"Average duration: {average_duration:.2f} seconds\n\n")
        f.write("Per-file durations:\n")
        for path, dur in zip(file_paths, durations):
            f.write(f"{os.path.basename(path)}\t{dur:.2f} sec\n")

    print(f"Duration stats saved to {output_log}")

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of files")
    plt.title("WAV File Duration Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Histogram saved to {output_plot}")


output_dir = "log/"
log_path = os.path.join(output_dir, "duration_stats.txt")
plot_path = os.path.join(output_dir, 'duration_hist.png')
analyze_wav_durations("/cpfs02/user/housiyuan/dataset/monkey/merge_audio_no_s", log_path, plot_path)