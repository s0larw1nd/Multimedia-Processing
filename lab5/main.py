import numpy as np
import librosa
import soundfile as sf

def hanning(M):
    N = M-1
    return np.array([1/2*(1-np.cos(2*np.pi*n/N)) for n in range(M)])

def dft(y):
    res = []
    n = len(y)
    for k in range(0, n):
        temp = 0 + 0j
        for m in range(0, n):
            temp += y[m] * np.exp(-2j*np.pi*k*m/n)

        res.append(temp)

    return res

def idft(y):
    res = []
    n = len(y)
    for k in range(0, n):
        temp = 0 + 0j
        for m in range(0, n):
            temp += y[m] * np.exp(2j*np.pi*k*m/n)

        res.append(temp)

    return res

def fft(x):
    N = x.shape[0]

    if N <= 32:
        return dft(x)
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])
    
def ifft(x):
    N = x.shape[0]

    if N <= 32:
        return idft(x)
    else:
        X_even = ifft(x[::2])
        X_odd = ifft(x[1::2])
        factor = np.exp(2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])

def stft(y, n_fft, hop_length, window=None):
    if window is None: window = np.hanning(n_fft)
    
    y = np.concatenate(([0]*(n_fft//2), y, [0]*(n_fft//2)))
    
    frames = 1 + n_fft//2
    bins = int(np.floor((len(y) - n_fft) / hop_length) + 1)
    
    S = np.zeros((frames, bins),dtype=np.complex64)
    
    for bin in range(bins):        
        temp = y[bin*hop_length:bin*hop_length+n_fft]
        temp = [temp[i] * window[i] for i in range(len(temp))]
        
        temp = fft(temp)
        
        S[:, bin] = temp[:1 + n_fft//2]
        
    return S

def istft(stft_matrix, hop_length, window):
    n_bins, n_frames = stft_matrix.shape
    n_fft = 2 * (n_bins - 1)

    y = np.zeros(n_fft + hop_length * (n_frames - 1))
    win_sum = np.zeros_like(y)
    
    for frame in range(n_frames):
        spec = stft_matrix[:, frame]
        full_spectrum = np.concatenate([spec, spec[1:-1][::-1].conj()])

        frame_td = ifft(full_spectrum) / len(full_spectrum)
        frame_td = frame_td.real
        start = frame * hop_length
        y[start:start + n_fft] += frame_td * window
        win_sum[start:start + n_fft] += window ** 2
        
    nz = win_sum > 1e-8
    y[nz] /= win_sum[nz]
    y = y[n_fft // 2: n_fft // 2 + hop_length * (n_frames - 1)]

    return y

sig, sr = librosa.load('./sample.wav', sr=None)
n_fft = 2048
step = n_fft // 4
han = np.hanning(n_fft)

FFT = librosa.stft(sig, n_fft=n_fft, hop_length=step, window=han)
magnitude, phase = np.abs(FFT), np.angle(FFT)

noise_sec = 0.5
n_noise_frames = int((noise_sec * sr - n_fft) / step) + 1
noise_frames = magnitude[:, :max(1, n_noise_frames)]
noise_spectrum = np.mean(noise_frames, axis=1, keepdims=True)

alpha = 2.5
beta = 0.02

mag_sub = magnitude - alpha * noise_spectrum
mag_sub = np.maximum(mag_sub, beta * magnitude)

fft_denoised = mag_sub * np.exp(1j * phase)

sig_denoised_librosa = librosa.istft(fft_denoised, hop_length=step, window=han)
sf.write('spectral_subtracted.wav', sig_denoised_librosa, sr)

sig_denoised_my = istft(fft_denoised, hop_length=step, window=han)
sf.write('my_spectral_subtracted.wav', sig_denoised_my, sr)

print(np.allclose(sig_denoised_librosa, sig_denoised_my))