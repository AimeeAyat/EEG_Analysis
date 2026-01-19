import numpy as np
import matplotlib.pyplot as plt
proc = np.load('EEGdata/sub01/process_data_1s_250Hz.npy')
i = 329
trial = proc[i]                # shape (64,250)
print('trial shape:', trial.shape)
print('min, max, mean:', trial.min(), trial.max(), trial.mean())
# channel RMS / bandpower
rms = (trial**2).mean(axis=1)**0.5
print('RMS per channel (first 8):', rms[:8])
# quick plot of a few channels and the mean
t = np.arange(trial.shape[1]) / 250.0
plt.figure(figsize=(10,4))
for ch in [0,8,16,24,32]:
    plt.plot(t, trial[ch] + ch*5, label=f'ch{ch}')   # vertical offset for visibility
plt.xlabel('Time (s)'); plt.title(f'Flat {i} label 20_earphone_06'); plt.legend(); plt.show()
# channel-averaged ERP
plt.figure(figsize=(8,3))
plt.plot(t, trial.mean(axis=0)); plt.xlabel('Time (s)'); plt.title('Channel-average ERP'); plt.show()