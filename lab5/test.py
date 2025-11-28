import numpy as np

def fft(x, N, s):
    if N==1:
        return x
    else:
        X0 = fft(x, N//2, 2*s)
        X1 = fft(x+s, N//2, 2*s)
        X = X0+X1
        for k in range(N//2-1):
            p = X[k]
            q = np.exp(-2*np.pi*1j/(N*k))
    
#print(np.allclose(np.fft.fft([i for i in range(1000)]), fft([i for i in range(1000)])))