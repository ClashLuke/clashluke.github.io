import struct
import random
import numpy as np
import matplotlib.pyplot as plt

N = 1000
DELTA = 0.001
START = 1.0
TRIALS = 500


def to_bf16_rne(x):
    b = struct.pack('>f', x)
    hi = int.from_bytes(b[:2], 'big')
    lo = (b[2] << 8) | b[3]
    if lo > 0x8000 or (lo == 0x8000 and hi & 1):  # round to nearest, ties to even
        hi += 1
    return struct.unpack('>f', hi.to_bytes(2, 'big') + b'\x00\x00')[0]


def rne_add(val_bf16, delta):
    return to_bf16_rne(val_bf16 + delta)


def sr_add(val_bf16, delta):
    s = val_bf16 + delta
    b = struct.pack('>f', s)
    hi = struct.unpack('>f', bytes([b[0], b[1], 0, 0]))[0]
    lo_bits = (b[2] << 8) | b[3]
    if random.randint(0, 65535) < lo_bits:
        int_val = int.from_bytes(b[:2], 'big') + 1
        hi = struct.unpack('>f', int_val.to_bytes(2, 'big') + b'\x00\x00')[0]
    return hi


fp32 = [START]
for i in range(N):
    fp32.append(fp32[-1] + DELTA)

rne = [to_bf16_rne(START)]
for i in range(N):
    rne.append(rne_add(rne[-1], DELTA))

sr_all = np.zeros((TRIALS, N + 1))
for t in range(TRIALS):
    random.seed(t)
    sr_all[t, 0] = to_bf16_rne(START)
    for i in range(N):
        sr_all[t, i + 1] = sr_add(sr_all[t, i], DELTA)

steps = np.arange(N + 1)
sr_mean = sr_all.mean(axis=0)
sr_lo = np.percentile(sr_all, 5, axis=0)
sr_hi = np.percentile(sr_all, 95, axis=0)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(steps, fp32, color='#2d2d2d', linewidth=2.5, label='fp32', zorder=3)
ax.plot(steps, rne, color='#d62728', linewidth=2.5, label='round-to-nearest', zorder=3)
ax.fill_between(steps, sr_lo, sr_hi, color='#1f77b4', alpha=0.15, zorder=1)
ax.plot(steps, sr_mean, color='#1f77b4', linewidth=2, linestyle='--',
        label='stochastic rounding (mean + 90% CI)', zorder=2)

ax.set_xlabel('Step', fontsize=12)
ax.set_ylabel('Accumulated value', fontsize=12)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, which='major', alpha=0.3)

fig.tight_layout()
fig.savefig('accumulation.png', dpi=180)
