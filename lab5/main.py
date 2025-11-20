import numpy as np
import librosa
import soundfile as sf

sig, sr = librosa.load('sample.wav', sr=None)
n_fft = 2048
step = n_fft // 4
han = np.hanning(n_fft)

fft = librosa.stft(sig, n_fft=n_fft, hop_length=step, window=han)
magnitude, phase = np.abs(fft), np.angle(fft)

noise_sec = 0.5
n_noise_frames = int((noise_sec * sr - n_fft) / step) + 1
noise_frames = magnitude[:, :max(1, n_noise_frames)]
noise_spectrum = np.mean(noise_frames, axis=1, keepdims=True)

alpha = 2.5
beta = 0.02

mag_sub = magnitude - alpha * noise_spectrum
mag_sub = np.maximum(mag_sub, beta * magnitude)

fft_denoised = mag_sub * np.exp(1j * phase)
sig_denoised = librosa.istft(fft_denoised, hop_length=step, window=han)
sf.write('spectral_subtracted.wav', sig_denoised, sr)