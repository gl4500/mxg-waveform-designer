"""
mxg_bin_validate.py
-------------------
Reads an N5182A-format .bin file (int16 big-endian interleaved I/Q),
validates the format, prints statistics, and plots the waveform.

Usage:
    python mxg_bin_validate.py waveform_full.bin --fs 125
    python mxg_bin_validate.py waveform_mxg_4mb.bin --fs 100 --info waveform_mxg_4mb_info.txt
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ─────────────────────────────────────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_bin(path: str) -> np.ndarray:
    """Load N5182A .bin → complex float64 array. Expects int16 big-endian I/Q."""
    raw = np.fromfile(path, dtype='>i2')          # big-endian int16
    if raw.size % 2 != 0:
        raise ValueError(f'Odd sample count ({raw.size}) — file may be truncated or wrong format.')
    I = raw[0::2].astype(np.float64)
    Q = raw[1::2].astype(np.float64)
    return I + 1j * Q


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATORS
# ─────────────────────────────────────────────────────────────────────────────

def check_endianness(path: str) -> dict:
    """
    Heuristic endianness check: compare big-endian vs little-endian peak power.
    A correctly encoded waveform should have similar I/Q power when read big-endian.
    Little-endian misread produces byte-swapped values with very different statistics.
    """
    raw_be = np.fromfile(path, dtype='>i2')   # big-endian (expected)
    raw_le = np.fromfile(path, dtype='<i2')   # little-endian

    # For a valid waveform, big-endian I and Q should have similar variance
    I_be, Q_be = raw_be[0::2].astype(float), raw_be[1::2].astype(float)
    I_le, Q_le = raw_le[0::2].astype(float), raw_le[1::2].astype(float)

    iq_balance_be = abs(np.var(I_be) - np.var(Q_be)) / max(np.var(I_be), 1)
    iq_balance_le = abs(np.var(I_le) - np.var(Q_le)) / max(np.var(I_le), 1)

    likely_be = iq_balance_be <= iq_balance_le

    return {
        'likely_big_endian': likely_be,
        'iq_imbalance_be':   iq_balance_be,
        'iq_imbalance_le':   iq_balance_le,
    }


def validate(path: str, fs_hz: float, info_path: str = None) -> dict:
    """Run all checks and return a results dict."""
    size_bytes = os.path.getsize(path)
    raw        = np.fromfile(path, dtype='>i2')
    n_samples  = raw.size // 2          # IQ pairs
    duration_s = n_samples / fs_hz if fs_hz else None

    I = raw[0::2].astype(np.float64)
    Q = raw[1::2].astype(np.float64)
    amp = np.sqrt(I**2 + Q**2)

    endian_check = check_endianness(path)

    # Read sidecar info if provided
    info_text = None
    if info_path and os.path.isfile(info_path):
        with open(info_path) as f:
            info_text = f.read()

    return {
        'path':             path,
        'size_bytes':       size_bytes,
        'size_mb':          size_bytes / 1e6,
        'n_raw_samples':    raw.size,
        'n_iq_pairs':       n_samples,
        'fs_hz':            fs_hz,
        'duration_s':       duration_s,
        'I':                I,
        'Q':                Q,
        'amp':              amp,
        'I_min':            int(I.min()),
        'I_max':            int(I.max()),
        'Q_min':            int(Q.min()),
        'Q_max':            int(Q.max()),
        'peak_amplitude':   amp.max(),
        'rms_amplitude':    np.sqrt(np.mean(amp**2)),
        'papr_db':          20 * np.log10(amp.max() / (np.sqrt(np.mean(amp**2)) + 1e-12)),
        'scale_utilisation':amp.max() / 32767.0 * 100,
        'endian':           endian_check,
        'info_text':        info_text,
        'dtype_written':    '>i2 (int16 big-endian)',
        'full_scale':       32767,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(r: dict):
    sep = '=' * 52
    print(f'\n{sep}')
    print(f'  MXG BIN VALIDATION REPORT')
    print(f'{sep}')
    print(f'  File          : {os.path.basename(r["path"])}')
    print(f'  Size          : {r["size_mb"]:.3f} MB  ({r["size_bytes"]:,} bytes)')
    print(f'  IQ pairs      : {r["n_iq_pairs"]:,}')
    print(f'  Sample rate   : {r["fs_hz"]/1e6:.3f} MSa/s')
    if r['duration_s'] is not None:
        print(f'  Duration      : {r["duration_s"]*1e3:.3f} ms')
    print(f'  Data type     : {r["dtype_written"]}')
    print()

    # Endianness
    e = r['endian']
    endian_ok = e['likely_big_endian']
    endian_sym = 'OK' if endian_ok else 'WARNING'
    print(f'  Endianness    : {"Big-endian (correct)" if endian_ok else "Looks LITTLE-endian — file may be wrong format"}  [{endian_sym}]')

    # Scaling
    util = r['scale_utilisation']
    scale_sym = 'OK' if util >= 50 else 'LOW'
    print(f'  I range       : {r["I_min"]:+6d} … {r["I_max"]:+6d}  (full scale = ±32767)')
    print(f'  Q range       : {r["Q_min"]:+6d} … {r["Q_max"]:+6d}')
    print(f'  Peak amplitude: {r["peak_amplitude"]:.1f}  ({util:.1f}% of full scale)  [{scale_sym}]')
    print(f'  RMS amplitude : {r["rms_amplitude"]:.1f}')
    print(f'  PAPR          : {r["papr_db"]:.2f} dB')
    print()

    # Warnings
    warnings = []
    if not endian_ok:
        warnings.append('File may be little-endian — N5182A requires big-endian.')
    if util < 50:
        warnings.append(f'Scale utilisation only {util:.1f}% — consider increasing peak_scale.')
    if util > 100:
        warnings.append('Peak amplitude exceeds ±32767 — clipping has occurred.')
    if r['size_mb'] > 4.0:
        warnings.append(f'File is {r["size_mb"]:.1f} MB — exceeds 4 MB MXG download limit. Use the _mxg_4mb.bin file.')

    if warnings:
        print(f'  WARNINGS ({len(warnings)}):')
        for w in warnings:
            print(f'    ⚠  {w}')
    else:
        print('  All checks passed.')

    if r['info_text']:
        print(f'\n  Sidecar info:\n')
        for line in r['info_text'].splitlines():
            print(f'    {line}')

    print(f'{sep}\n')


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_waveform(r: dict):
    I  = r['I']
    Q  = r['Q']
    fs = r['fs_hz']
    N  = len(I)
    t  = np.arange(N) / fs * 1e6     # µs

    # Cap plot samples to keep it responsive
    MAX_PLOT = 2 ** 16
    if N > MAX_PLOT:
        step = N // MAX_PLOT
        t_p, I_p, Q_p = t[::step], I[::step], Q[::step]
    else:
        t_p, I_p, Q_p = t, I, Q

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'MXG Bin Validation — {os.path.basename(r["path"])}', fontsize=12)
    gs  = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Time domain ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t_p, I_p, color='royalblue', linewidth=0.5, label='I')
    ax1.plot(t_p, Q_p, color='tomato',    linewidth=0.5, label='Q', alpha=0.75)
    ax1.axhline( 32767, color='grey', linewidth=0.6, linestyle='--', label='±32767 full scale')
    ax1.axhline(-32767, color='grey', linewidth=0.6, linestyle='--')
    ax1.set_xlabel('Time (µs)')
    ax1.set_ylabel('Amplitude (int16)')
    ax1.set_title('Time Domain I/Q')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Constellation ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(I_p, Q_p, s=0.3, alpha=0.3, color='royalblue')
    lim = 35000
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_aspect('equal')
    ax2.set_xlabel('I')
    ax2.set_ylabel('Q')
    ax2.set_title('IQ Constellation')
    ax2.grid(True, alpha=0.3)

    # ── Amplitude envelope ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    amp_p = np.sqrt(I_p**2 + Q_p**2)
    ax3.plot(t_p, amp_p, color='darkorange', linewidth=0.5)
    ax3.axhline(32767, color='grey', linewidth=0.6, linestyle='--', label='Full scale')
    ax3.set_xlabel('Time (µs)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Amplitude Envelope')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Power spectrum ───────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    iq_c  = (I + 1j * Q)
    n_fft = min(len(iq_c), 8192)
    win   = np.hanning(n_fft)
    spec  = np.fft.fftshift(np.fft.fft(iq_c[:n_fft] * win, n_fft))
    psd   = 20 * np.log10(np.abs(spec) / n_fft + 1e-12)
    freq  = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1/fs)) / 1e6

    ax4.plot(freq, psd, color='royalblue', linewidth=0.7)
    ax4.set_xlabel('Frequency (MHz, baseband)')
    ax4.set_ylabel('Power (dBFS)')
    ax4.set_title('Power Spectrum (single-window FFT)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(freq[0], freq[-1])

    plt.savefig(r['path'].replace('.bin', '_validate.png'), dpi=150, bbox_inches='tight')
    print(f'  Plot saved: {r["path"].replace(".bin", "_validate.png")}')
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Validate and plot an N5182A MXG .bin waveform file.')
    parser.add_argument('bin_file',
                        help='Path to the .bin file')
    parser.add_argument('--fs', type=float, default=125.0,
                        help='Sample rate in MSa/s (default: 125)')
    parser.add_argument('--info', type=str, default=None,
                        help='Path to sidecar _info.txt (optional)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plots, print report only')
    args = parser.parse_args()

    if not os.path.isfile(args.bin_file):
        print(f'ERROR: file not found: {args.bin_file}')
        sys.exit(1)

    # Auto-find sidecar if not specified
    info_path = args.info
    if info_path is None:
        auto = args.bin_file.replace('.bin', '_info.txt')
        if os.path.isfile(auto):
            info_path = auto

    fs_hz = args.fs * 1e6
    r     = validate(args.bin_file, fs_hz, info_path)
    print_report(r)

    if not args.no_plot:
        plot_waveform(r)


if __name__ == '__main__':
    main()
