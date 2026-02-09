import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import librosa
import librosa.display
import soundfile as sf

### Config
ROW_INDEX = 10
DURATION = 3  # seconds
SR = 44100  # Sample rate for audio

# ADSR Envelope Parameters
ATTACK_TIME = 0.1  # seconds
DECAY_TIME = 0.1   # seconds
SUSTAIN_LEVEL = 0.7  # Sustain level (0 to 1)
RELEASE_TIME = 0.2  # seconds
### END Config

def create_adsr_envelope(duration, sr, a, d, s, r):
    """Create an ADSR envelope."""
    attack_samples = int(a * sr)
    decay_samples = int(d * sr)
    release_samples = int(r * sr)
    total_length = int(duration * sr)
    sustain_samples = total_length - (attack_samples + decay_samples + release_samples)

    if sustain_samples < 0:
        raise ValueError("ADSR envelope time exceeds total duration.")

    # Create the envelope
    envelope = np.concatenate([
        np.linspace(0, 1, attack_samples),  # Attack
        np.linspace(1, s, decay_samples),  # Decay
        np.full(sustain_samples, s),  # Sustain
        np.linspace(s, 0, release_samples)  # Release
    ])
    
    return envelope

### Generating Sound from Coffee Data ###
# 1. Load Data
df = pd.read_csv('Instant_Coffee_Test_Samples.csv')
sample_id = df.iloc[ROW_INDEX, 0]
raw_values = df.iloc[ROW_INDEX, 1:].values.astype(float)
wavelengths = df.columns[1:].str.replace('X', '').astype(float)

# 2. Savitzky–Golay Smoothing
smoothed_values = savgol_filter(raw_values, window_length=15, polyorder=3)

# 3. Normalise Sound (via Librosa)
audio_signal = librosa.util.normalize(smoothed_values - np.mean(smoothed_values))

# 4. Loop the audio to fill the desired duration
num_samples = int(SR * DURATION)
full_audio = np.tile(audio_signal, int(np.ceil(num_samples / len(audio_signal))))[:num_samples]

# 5. Apply ADSR Envelope
envelope = create_adsr_envelope(DURATION, SR, ATTACK_TIME, DECAY_TIME, SUSTAIN_LEVEL, RELEASE_TIME)
full_audio *= envelope

# Save audio
audio_filename = f"coffee_sample_{sample_id}.wav"
plot_filename = f"coffee_visualization_{sample_id}.png"
sf.write(audio_filename, full_audio, SR)

### Visualization of the generated sound ###
# 1. Load the generated sound
y, sr = librosa.load(audio_filename)
plt.figure(figsize=(12, 8))

#--- Plot 0: Smoothed Spectrum ---
plt.subplot(3, 1, 1)
plt.plot(wavelengths, raw_values, label="Raw", alpha=0.5)
plt.plot(wavelengths, smoothed_values, label="SavGol Smoothed")
plt.title('Coffee Sample Spectrum')
plt.xlabel('Wavelength')
plt.ylabel('Absorbance')

# --- Plot 1: Waveform ---
plt.subplot(3, 1, 2)
# Zooming in to 0.05 seconds to see the cycle detail
librosa.display.waveshow(y[:int(0.05 * sr)], sr=sr, color="saddlebrown")
plt.title('Coffee Sound Waveform (Zoomed 0.05s)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# --- Plot 2: Spectrogram ---
plt.subplot(3, 1, 3)
# Compute Short-Time Fourier Transform (STFT)
D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
# Display spectrogram
img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
plt.colorbar(img, format="%+2.0f dB")
plt.title('Coffee Sound Spectrogram')
plt.ylim(20, 10000)
plt.ylabel('Frequency (Hz)')

plt.tight_layout()
plt.savefig(plot_filename)
plt.show()