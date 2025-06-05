import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import read
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.signal import find_peaks
import tkinter as tk
import os
import random

# --- DTMF Frequencies ---
dtmf_freqs = {
    '1': (697, 1209), '2': (697, 1336), '3': (697, 1477),
    '4': (770, 1209), '5': (770, 1336), '6': (770, 1477),
    '7': (852, 1209), '8': (852, 1336), '9': (852, 1477),
    '*': (941, 1209), '0': (941, 1336), '#': (941, 1477)
}

# --- Generate DTMF Tone ---
def generate_dtmf(digit, duration=0.5, fs=8000, noise_level=0.0):
    f1, f2 = dtmf_freqs[digit]
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    tone /= np.max(np.abs(tone))
    tone += noise_level * np.random.randn(len(tone))  # Add noise
    return tone, t

# --- Play Sound ---
def play_sound(signal, fs=8000):
    sd.play(signal, fs)
    sd.wait()

# --- FFT Analysis ---
def compute_fft(signal_data, fs):
    N = len(signal_data)
    windowed = signal_data * np.hamming(N)
    yf = np.abs(fft(windowed))[:N // 2]
    xf = fftfreq(N, 1 / fs)[:N // 2]
    return xf, yf

# --- Detect Peaks ---
def detect_peaks(xf, yf, min_height=50):
    peaks, props = find_peaks(yf, height=min_height)
    peak_freqs = xf[peaks]
    magnitudes = props['peak_heights']
    sorted_peaks = sorted(zip(peak_freqs, magnitudes), key=lambda x: -x[1])[:2]
    return sorted([f for f, _ in sorted_peaks])

# --- Decode DTMF ---
def decode_dtmf(f1, f2):
    rows = [697, 770, 852, 941]
    cols = [1209, 1336, 1477]
    keypad = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9'], ['*', '0', '#']]
    row = min(rows, key=lambda x: abs(x - min(f1, f2)))
    col = min(cols, key=lambda x: abs(x - max(f1, f2)))
    return keypad[rows.index(row)][cols.index(col)]

# --- Bandpass Filter ---
def bandpass_filter(signal_data, fs, lowcut=650, highcut=1600):
    sos = signal.butter(10, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return signal.sosfilt(sos, signal_data)

# --- Button Click Handler ---
def on_button_click(digit, noise_scale):
    noise_level = noise_scale.get()
    tone, t = generate_dtmf(digit, noise_level=noise_level)
    play_sound(tone)
    
    xf, yf = compute_fft(tone, 8000)
    peak_freqs = detect_peaks(xf, yf)
    decoded = decode_dtmf(*peak_freqs)

    print(f"📞 You Pressed: {digit} => Detected: {decoded}")

    plt.figure(figsize=(10, 4))
    plt.plot(t[:500], tone[:500])
    plt.title(f"Waveform of '{digit}' with noise={noise_level:.2f}")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(xf, yf)
    plt.title(f"FFT Spectrum of '{digit}'")
    plt.xlabel("Frequency (Hz)")
    plt.grid(True)
    plt.show()

# --- Analyze WAV Files from 'tunes' Folder ---
def analyze_wav_files():
    print("🔍 Processing .wav DTMF Files...")
    tunes_folder = "tunes"
    for file in os.listdir(tunes_folder):
        if file.endswith('.wav') and file.startswith('dtmf_'):
            filepath = os.path.join(tunes_folder, file)
            print(f"\n🎧 File: {file}")
            fs, signal_data = read(filepath)
            signal_data = signal_data.astype(np.float32)
            signal_data /= np.max(np.abs(signal_data))
            t = np.linspace(0, len(signal_data) / fs, len(signal_data), endpoint=False)

            plt.figure(figsize=(10, 4))
            plt.plot(t[:500], signal_data[:500])
            plt.title(f"Waveform - {file}")
            plt.xlabel("Time (s)")
            plt.grid(True)
            plt.show()

            xf, yf = compute_fft(signal_data, fs)
            plt.figure(figsize=(10, 4))
            plt.plot(xf, yf)
            plt.title("FFT Spectrum")
            plt.xlabel("Frequency (Hz)")
            plt.grid(True)
            plt.show()

            peak_freqs = detect_peaks(xf, yf)
            print("📈 Detected Frequencies:", peak_freqs)
            digit = decode_dtmf(*peak_freqs)
            print("📞 Decoded Digit:", digit)

# --- Test Difficult Signal ---
def test_difficult_signal():
    fs = 8000
    true_digit = random.choice(list(dtmf_freqs.keys()))
    tone, t = generate_dtmf(true_digit, duration=0.12, fs=fs, noise_level=0.4)

    print(f"\n🧪 Testing Difficult Signal for digit: {true_digit}")
    play_sound(tone)

    # FFT before filtering
    xf1, yf1 = compute_fft(tone, fs)
    raw_freqs = detect_peaks(xf1, yf1)
    raw_digit = decode_dtmf(*raw_freqs)

    # Filtering
    filtered_tone = bandpass_filter(tone, fs)
    xf2, yf2 = compute_fft(filtered_tone, fs)
    filtered_freqs = detect_peaks(xf2, yf2)
    filtered_digit = decode_dtmf(*filtered_freqs)

    # Results
    print(f"🔍 Before Filtering: {raw_freqs} => Decoded: {raw_digit}")
    print(f"✅ After Filtering : {filtered_freqs} => Decoded: {filtered_digit}")

    # Plots
    plt.figure(figsize=(10, 4))
    plt.plot(t, tone)
    plt.title(f"Raw Waveform of '{true_digit}' (Noisy & Short)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(xf1, yf1)
    plt.title("FFT Before Filtering")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(xf2, yf2)
    plt.title("FFT After Filtering")
    plt.grid(True)
    plt.show()

# --- GUI Setup ---
def launch_gui():
    root = tk.Tk()
    root.title("DTMF Dialer with Noise Control")
    root.configure(bg='#1E1E1E')

    # Noise Level Slider
    slider_label = tk.Label(root, text="Noise Level", fg='white', bg='#1E1E1E', font=('Arial', 12))
    slider_label.pack(pady=(10, 0))

    noise_scale = tk.Scale(root, from_=0.0, to=1.0, resolution=0.05, orient='horizontal',
                           length=300, bg='#1E1E1E', fg='white', troughcolor='gray')
    noise_scale.set(0.0)
    noise_scale.pack(pady=(0, 20))

    # Buttons Frame
    frame = tk.Frame(root, bg='#1E1E1E')
    frame.pack()

    buttons = [['1', '2', '3'],
               ['4', '5', '6'],
               ['7', '8', '9'],
               ['*', '0', '#']]

    for i, row in enumerate(buttons):
        for j, char in enumerate(row):
            btn = tk.Button(frame, text=char, font=('Arial', 20), width=5, height=2,
                            command=lambda c=char: on_button_click(c, noise_scale),
                            fg='white', bg='#008080', activebackground='#006666', relief='raised', bd=2)
            btn.grid(row=i, column=j, padx=10, pady=10)

    # Extra Buttons
    analyze_btn = tk.Button(root, text="🔍 Analyze WAV Files", font=('Arial', 12),
                            command=analyze_wav_files, fg='white', bg='#2196F3', relief='raised', bd=2)
    analyze_btn.pack(pady=10)

    test_btn = tk.Button(root, text="🎯 Test Difficult Signal", font=('Arial', 12),
                         command=test_difficult_signal, fg='white', bg='#F44336', relief='raised', bd=2)
    test_btn.pack(pady=10)

    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())
    root.mainloop()

# --- Run ---
if __name__ == '__main__':
    launch_gui()
