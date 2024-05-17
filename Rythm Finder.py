import librosa
import numpy as np
import matplotlib.pyplot as plt

def plot_beat_times(y, beat_frames, sr):
    # محاسبه زمان‌های ضربان در فایل صوتی
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # نمایش نمودار فایل صوتی همراه با زمان‌های ضربان
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(y)) / sr, y, alpha=0.5)
    plt.vlines(beat_times, -1, 1, color='r', linestyle='--', alpha=0.7, label='Beats')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio waveform with beat times')
    plt.legend()
    plt.show()

def calculate_and_plot_beats(audio_file):
    # بارگذاری فایل صوتی
    y, sr = librosa.load(audio_file)

    # محاسبه ضربان‌ها
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # نمایش ضربان‌ها بر روی نمودار
    plot_beat_times(y, beat_frames, sr)
    return tempo

def main():
    audio_file = "sample_audio.wav"
    bpm = calculate_and_plot_beats(audio_file)
    print("BPM of the audio file:", bpm)

if __name__ == "__main__":
    main()
