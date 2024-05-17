import librosa
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_beat_times(y, beat_frames, sr, alpha=0.5, save_path=None):
    # محاسبه زمان‌های ضربان در فایل صوتی
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # نمایش نمودار فایل صوتی همراه با زمان‌های ضربان
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(y)) / sr, y, alpha=alpha)
    plt.vlines(beat_times, min(y), max(y), color='r', linestyle='--', alpha=0.7, label='Beats')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio waveform with beat times')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def calculate_and_plot_beats(audio_file, alpha=0.5, save_path=None):
    # بارگذاری فایل صوتی
    y, sr = librosa.load(audio_file)

    # محاسبه ضربان‌ها
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # نمایش ضربان‌ها بر روی نمودار
    plot_beat_times(y, beat_frames, sr, alpha, save_path)
    return tempo

def main():
    parser = argparse.ArgumentParser(description="Rhythm Finder: Beat detection in audio files.")
    parser.add_argument('audio_file', type=str, help='Path to the audio file')
    parser.add_argument('--alpha', type=float, default=0.5, help='Transparency of the waveform plot')
    parser.add_argument('--save', type=str, help='Path to save the plot image')
    args = parser.parse_args()

    bpm = calculate_and_plot_beats(args.audio_file, args.alpha, args.save)
    print("BPM of the audio file:", bpm)

if __name__ == "__main__":
    main()
