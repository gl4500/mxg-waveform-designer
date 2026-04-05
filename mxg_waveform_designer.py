#!/usr/bin/env python3
"""
mxg_waveform_designer.py
========================
Flexible multi-channel composite I/Q waveform designer for
Agilent / Keysight MXG (N5182A / N5182B class).

Architecture
------------
  WaveformType  – enum of supported modulation types
  ChannelConfig – per-channel descriptor  (type + params dict)
  CompositeConfig – global pulse-train settings
  CompositeBuilder – assembles channels → pulse train, normalises
  WaveformPlotter  – all diagnostic plots
  WaveformExporter – saves .MAT / .CSV / .BIN

Supported waveform types (mix freely across channels)
------------------------------------------------------
  LFM     – linear FM chirp
  NLFM    – non-linear FM (tangent / cosine / hamming law)
  CW      – pure tone
  FMCW    – sawtooth or triangle FM sweep
  STEPPED – stepped-frequency (N equal-dwell CW bursts)
  BPSK    – binary phase coded (Barker-13 default, or custom)
  FRANK   – Frank polyphase code (order M → M² chips)

Quick-start
-----------
  See the __main__ block at the bottom for three ready-to-run examples:
    1.  40-channel LFM  (replicates original mxg_40ch_composite_lfm.py)
    2.  NLFM bank  (tangent law, better time-sidelobes, no amplitude window loss)
    3.  Mixed waveform  (LFM + NLFM + BPSK + Stepped-Freq in one composite)

Prerequisites
-------------
    pip install numpy scipy matplotlib pandas
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sig
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# ENUMERATIONS
# ─────────────────────────────────────────────────────────────────────────────

class WaveformType(Enum):
    LFM     = auto()   # linear FM chirp
    NLFM    = auto()   # non-linear FM
    CW      = auto()   # continuous wave (pure tone)
    FMCW    = auto()   # FM-CW (sawtooth / triangle)
    STEPPED = auto()   # stepped-frequency burst
    BPSK    = auto()   # binary phase-coded
    FRANK   = auto()   # Frank polyphase code


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChannelConfig:
    """
    Describes one channel in the composite waveform.

    center_freq_hz : baseband offset from DC [Hz]
    waveform_type  : WaveformType member
    amplitude      : relative linear amplitude (0–1)
    phase_rad      : initial phase [radians]
    params         : waveform-type-specific keyword arguments (see table below)

    params keys per waveform type
    ─────────────────────────────
    LFM:
        bandwidth_hz  (float) – chirp bandwidth               default 2e6
        up_chirp      (bool)  – True = up-chirp               default True
    NLFM:
        bandwidth_hz  (float) – chirp bandwidth               default 2e6
        up_chirp      (bool)                                   default True
        law           (str)   – 'tangent' | 'cosine' | 'hamming'  default 'tangent'
        beta          (float) – shaping factor (tangent only)  default 1.8
    CW:
        (no extra params)
    FMCW:
        bandwidth_hz  (float) – sweep bandwidth                default 2e6
        shape         (str)   – 'sawtooth' | 'triangle'        default 'sawtooth'
        up_chirp      (bool)  – sawtooth direction             default True
    STEPPED:
        bandwidth_hz  (float) – total spanned bandwidth        default 10e6
        n_steps       (int)   – number of frequency steps      default 8
    BPSK:
        code          (list)  – chip values (+1/-1)            default Barker-13
        chip_rate_hz  (float) – chip rate                      default 1e6
    FRANK:
        order         (int)   – M; total chips = M²            default 4
        chip_rate_hz  (float) – chip rate                      default 1e6
    """
    center_freq_hz: float
    waveform_type:  WaveformType        = WaveformType.LFM
    amplitude:      float               = 1.0
    phase_rad:      float               = 0.0
    params:         Dict[str, Any]      = field(default_factory=dict)


@dataclass
class CompositeConfig:
    """
    Global pulse-train parameters.

    All times in seconds, all frequencies in Hz.
    """
    fs:              float  = 150e6    # sample rate
    num_pulses:      int    = 20       # repetitions (PRIs) in the output
    pulse_width_s:   float  = 1e-3    # on-time per PRI
    pri_s:           float  = 2e-3    # PRI duration (≥ pulse_width_s)
    pulse_offset_s:  float  = 0.0     # delay of pulse from PRI start

    # Window / taper applied to every pulse envelope
    use_window:      bool   = True
    window_type:     str    = 'hann'  # 'hann' | 'hamming' | 'tukey' | 'none'
    tukey_alpha:     float  = 0.25

    # Optional cosine-taper amplitude roll-off across channel bank
    use_amp_taper:   bool   = False
    edge_amp_scale:  float  = 0.75
    center_amp_scale:float  = 1.00

    # Random phase per channel for PAPR reduction
    use_random_phase:bool   = True
    rng_seed:        int    = 12345

    # Output normalisation
    final_peak_scale:float  = 0.85

    # RF centre – metadata / documentation only
    rf_center_hz:    float  = 2.4e9

    # File output
    base_file_name:  str    = 'mxg_composite'
    save_mat:        bool   = True
    save_csv:        bool   = True
    save_bin:        bool   = True


# ─────────────────────────────────────────────────────────────────────────────
# WAVEFORM KERNEL FUNCTIONS
# Each returns a complex baseband array of length len(t).
# ─────────────────────────────────────────────────────────────────────────────

def _kernel_lfm(t: np.ndarray, pw: float, fc: float,
                bandwidth_hz: float = 2e6,
                up_chirp: bool = True, **_) -> np.ndarray:
    """Linear FM chirp – exact analytic phase."""
    half_bw = bandwidth_hz / 2
    f0 = fc - half_bw if up_chirp else fc + half_bw
    k  = bandwidth_hz / pw * (1 if up_chirp else -1)
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t ** 2)
    return np.exp(1j * phase)


def _kernel_nlfm(t: np.ndarray, pw: float, fc: float,
                 bandwidth_hz: float = 2e6,
                 up_chirp: bool = True,
                 law: str = 'tangent',
                 beta: float = 1.8, **_) -> np.ndarray:
    """
    Non-linear FM chirp.  Phase is built by numerically integrating the
    instantaneous frequency law, which concentrates dwell time at band
    edges → lower time-sidelobes without the 1–2 dB SNR loss of a window.

    law = 'tangent'  – hyperbolic-tangent frequency law; beta controls sharpness
    law = 'cosine'   – sinusoidal frequency law (gentle, -25 dB PSL)
    law = 'hamming'  – FM law derived from Hamming spectral density (≈-43 dB PSL)
    """
    dt   = t[1] - t[0]
    half = bandwidth_hz / 2
    u    = 2 * t / pw - 1          # normalised time ∈ [-1, 1]

    if law == 'tangent':
        # f_inst ranges ±half_bw, spending more time near band edges
        f_dev = half * np.tanh(beta * u) / np.tanh(beta)
    elif law == 'cosine':
        # f_dev = half * sin(π*u/2), smooth nonlinear compression
        f_dev = half * np.sin(np.pi * u / 2)
    elif law == 'hamming':
        # Spectral density ∝ Hamming window → invert to get frequency vs time
        # via normalised CDF of the Hamming profile
        tau  = np.linspace(0, 1, len(t))
        spec = 0.54 - 0.46 * np.cos(2 * np.pi * tau)
        cdf  = np.cumsum(spec) / np.sum(spec)   # ∈ [0, 1]
        f_dev = (cdf - 0.5) * bandwidth_hz
    else:
        raise ValueError(f'Unknown NLFM law: {law!r}. Use tangent/cosine/hamming.')

    if not up_chirp:
        f_dev = -f_dev

    f_inst = fc + f_dev
    phase  = 2 * np.pi * np.cumsum(f_inst) * dt
    return np.exp(1j * phase)


def _kernel_cw(t: np.ndarray, fc: float, **_) -> np.ndarray:
    """Pure continuous-wave tone at fc."""
    return np.exp(1j * 2 * np.pi * fc * t)


def _kernel_fmcw(t: np.ndarray, pw: float, fc: float,
                 bandwidth_hz: float = 2e6,
                 shape: str = 'sawtooth',
                 up_chirp: bool = True, **_) -> np.ndarray:
    """
    FM-CW sweep.
    shape = 'sawtooth' – single linear ramp across pulse width
    shape = 'triangle' – up then down ramp (half bandwidth each way)
    """
    if shape == 'sawtooth':
        # identical to LFM but the whole pulse is the sweep (no gating elsewhere)
        return _kernel_lfm(t, pw, fc,
                           bandwidth_hz=bandwidth_hz, up_chirp=up_chirp)
    elif shape == 'triangle':
        half = len(t) // 2
        s    = np.zeros(len(t), dtype=complex)
        # up ramp
        s[:half] = _kernel_lfm(t[:half], t[half - 1] - t[0],
                                fc, bandwidth_hz=bandwidth_hz, up_chirp=True)
        # down ramp (continues from where up ramp ended)
        s[half:] = _kernel_lfm(t[half:] - t[half], t[-1] - t[half],
                                fc, bandwidth_hz=bandwidth_hz, up_chirp=False)
        return s
    else:
        raise ValueError(f'Unknown FMCW shape: {shape!r}. Use sawtooth/triangle.')


def _kernel_stepped(t: np.ndarray, pw: float, fc: float,
                    bandwidth_hz: float = 10e6,
                    n_steps: int = 8, **_) -> np.ndarray:
    """
    Stepped-frequency burst: N equal-dwell CW tones uniformly spanning
    bandwidth_hz centred on fc.  Each step occupies pulse_width / n_steps seconds.
    """
    N      = len(t)
    dwell  = N // n_steps
    s      = np.zeros(N, dtype=complex)
    freqs  = np.linspace(fc - bandwidth_hz / 2,
                         fc + bandwidth_hz / 2, n_steps)
    for i, f_step in enumerate(freqs):
        lo = i * dwell
        hi = lo + dwell if i < n_steps - 1 else N
        s[lo:hi] = np.exp(1j * 2 * np.pi * f_step * t[lo:hi])
    return s


def _kernel_bpsk(t: np.ndarray, pw: float, fc: float,
                 code: Optional[List[int]] = None,
                 chip_rate_hz: float = 1e6, **_) -> np.ndarray:
    """
    Binary phase-coded (BPSK) burst.
    code     : list of +1 / -1 chip values  (default: Barker-13)
    chip_rate: chip rate in Hz
    The code is stretched to fill the available pulse time.
    """
    if code is None:
        code = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]  # Barker-13

    N          = len(t)
    n_chips    = len(code)
    chip_samps = max(1, int(round(N / n_chips)))
    s          = np.zeros(N, dtype=complex)

    for i, chip in enumerate(code):
        lo = i * chip_samps
        hi = min(lo + chip_samps, N)
        phi = 0.0 if chip > 0 else np.pi
        s[lo:hi] = np.exp(1j * (2 * np.pi * fc * t[lo:hi] + phi))
    return s


def _kernel_frank(t: np.ndarray, pw: float, fc: float,
                  order: int = 4,
                  chip_rate_hz: float = 1e6, **_) -> np.ndarray:
    """
    Frank polyphase code of order M  (M² total chips).
    Phase of chip (p, q): φ_{p,q} = 2π * p * q / M,  p,q ∈ {0…M-1}

    Frank codes have near-ideal thumbtack ambiguity at the expense of
    higher peak sidelobes vs Barker under Doppler.
    """
    M       = order
    n_chips = M * M
    phases  = np.array([2 * np.pi * p * q / M
                        for p in range(M) for q in range(M)])

    N          = len(t)
    chip_samps = max(1, int(round(N / n_chips)))
    s          = np.zeros(N, dtype=complex)

    for i, phi_chip in enumerate(phases):
        lo = i * chip_samps
        hi = min(lo + chip_samps, N)
        s[lo:hi] = np.exp(1j * (2 * np.pi * fc * t[lo:hi] + phi_chip))
    return s


# Dispatch table
_KERNELS = {
    WaveformType.LFM:     _kernel_lfm,
    WaveformType.NLFM:    _kernel_nlfm,
    WaveformType.CW:      _kernel_cw,
    WaveformType.FMCW:    _kernel_fmcw,
    WaveformType.STEPPED: _kernel_stepped,
    WaveformType.BPSK:    _kernel_bpsk,
    WaveformType.FRANK:   _kernel_frank,
}


# ─────────────────────────────────────────────────────────────────────────────
# CHANNEL BANK FACTORY HELPERS
# Convenience functions that produce lists of ChannelConfig objects.
# ─────────────────────────────────────────────────────────────────────────────

def lfm_channel_bank(n_channels: int = 40,
                     chan_spacing_hz: float = 3e6,
                     bandwidth_hz: float = 2e6,
                     up_chirp: bool = True) -> List[ChannelConfig]:
    """N uniformly spaced LFM channels centred on DC."""
    offsets = (np.arange(n_channels) - (n_channels - 1) / 2) * chan_spacing_hz
    return [
        ChannelConfig(
            center_freq_hz=float(fc),
            waveform_type=WaveformType.LFM,
            params={'bandwidth_hz': bandwidth_hz, 'up_chirp': up_chirp},
        )
        for fc in offsets
    ]


def nlfm_channel_bank(n_channels: int = 40,
                      chan_spacing_hz: float = 3e6,
                      bandwidth_hz: float = 2e6,
                      law: str = 'tangent',
                      beta: float = 1.8,
                      up_chirp: bool = True) -> List[ChannelConfig]:
    """N uniformly spaced NLFM channels."""
    offsets = (np.arange(n_channels) - (n_channels - 1) / 2) * chan_spacing_hz
    return [
        ChannelConfig(
            center_freq_hz=float(fc),
            waveform_type=WaveformType.NLFM,
            params={'bandwidth_hz': bandwidth_hz, 'law': law,
                    'beta': beta, 'up_chirp': up_chirp},
        )
        for fc in offsets
    ]


def stepped_freq_channel_bank(n_channels: int = 10,
                               chan_spacing_hz: float = 15e6,
                               total_bw_hz: float = 10e6,
                               n_steps: int = 8) -> List[ChannelConfig]:
    """N uniformly spaced stepped-frequency channels."""
    offsets = (np.arange(n_channels) - (n_channels - 1) / 2) * chan_spacing_hz
    return [
        ChannelConfig(
            center_freq_hz=float(fc),
            waveform_type=WaveformType.STEPPED,
            params={'bandwidth_hz': total_bw_hz, 'n_steps': n_steps},
        )
        for fc in offsets
    ]


def cw_channel_bank(n_channels: int = 8,
                    chan_spacing_hz: float = 5e6) -> List[ChannelConfig]:
    """N pure-tone CW channels."""
    offsets = (np.arange(n_channels) - (n_channels - 1) / 2) * chan_spacing_hz
    return [
        ChannelConfig(center_freq_hz=float(fc), waveform_type=WaveformType.CW)
        for fc in offsets
    ]


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class CompositeBuilder:
    """
    Assembles a list of ChannelConfig objects into a normalised I/Q pulse train.

    Usage
    -----
        cfg      = CompositeConfig(fs=150e6, num_pulses=20, ...)
        channels = lfm_channel_bank(40, chan_spacing_hz=3e6, bandwidth_hz=2e6)
        builder  = CompositeBuilder(cfg, channels)
        iq, table = builder.build()
    """

    def __init__(self, config: CompositeConfig, channels: List[ChannelConfig]):
        self.cfg      = config
        self.channels = channels
        self._validate()

    # ── public API ───────────────────────────────────────────────────────────

    def build(self):
        """
        Returns
        -------
        iq    : complex128 ndarray  – full normalised pulse train
        table : DataFrame           – per-channel metadata
        """
        cfg = self.cfg
        NsPulse = int(round(cfg.fs * cfg.pulse_width_s))
        NsPRI   = int(round(cfg.fs * cfg.pri_s))
        t_pulse = np.arange(NsPulse) / cfg.fs

        window        = self._make_window(NsPulse)
        amp_weights   = self._channel_weights()
        phase_offsets = self._channel_phases()

        composite_pulse = np.zeros(NsPulse, dtype=complex)
        meta_rows = []

        for ch_idx, ch in enumerate(self.channels):
            kernel = _KERNELS[ch.waveform_type]

            # generate unit-amplitude baseband sample
            raw = kernel(t_pulse, cfg.pulse_width_s, ch.center_freq_hz,
                         **ch.params)

            # apply initial phase, amplitude weight, and envelope window
            amp   = ch.amplitude * amp_weights[ch_idx]
            phi   = ch.phase_rad + phase_offsets[ch_idx]
            s     = amp * np.exp(1j * phi) * raw * window

            composite_pulse += s
            meta_rows.append(self._channel_metadata(ch_idx, ch, phi, amp))

        # insert pulse into PRI
        pulse_start = int(round(cfg.pulse_offset_s * cfg.fs))
        one_pri     = np.zeros(NsPRI, dtype=complex)
        one_pri[pulse_start:pulse_start + NsPulse] = composite_pulse

        # tile into full pulse train
        iq = np.tile(one_pri, cfg.num_pulses)

        # normalise
        iq, stats = self._normalise(iq)
        self._print_summary(iq, stats)

        table = pd.DataFrame(meta_rows)
        return iq, table

    # ── private helpers ──────────────────────────────────────────────────────

    def _validate(self):
        cfg = self.cfg
        NsPulse = int(round(cfg.fs * cfg.pulse_width_s))
        NsPRI   = int(round(cfg.fs * cfg.pri_s))
        pulse_start = int(round(cfg.pulse_offset_s * cfg.fs))
        if NsPulse > NsPRI:
            raise ValueError('pulse_width_s cannot exceed pri_s')
        if pulse_start + NsPulse > NsPRI:
            raise ValueError('Pulse extends beyond PRI end – reduce pulse_offset_s')
        if not self.channels:
            raise ValueError('Channel list is empty')
        total_span = (max(abs(c.center_freq_hz) for c in self.channels)
                      + max(c.params.get('bandwidth_hz', 0) for c in self.channels) / 2)
        if 2 * total_span > cfg.fs:
            print(f'WARNING: Estimated occupied span {2*total_span/1e6:.1f} MHz '
                  f'exceeds sample rate {cfg.fs/1e6:.1f} MHz.')

    def _make_window(self, N: int) -> np.ndarray:
        cfg = self.cfg
        if not cfg.use_window:
            return np.ones(N)
        wt = cfg.window_type.lower()
        if wt == 'hann':    return sig.windows.hann(N)
        if wt == 'hamming': return sig.windows.hamming(N)
        if wt == 'tukey':   return sig.windows.tukey(N, cfg.tukey_alpha)
        if wt == 'none':    return np.ones(N)
        raise ValueError(f'Unknown window_type: {cfg.window_type!r}')

    def _channel_weights(self) -> np.ndarray:
        cfg = self.cfg
        n   = len(self.channels)
        if not cfg.use_amp_taper:
            return np.ones(n)
        x = np.linspace(-1, 1, n)
        w = (cfg.edge_amp_scale
             + (cfg.center_amp_scale - cfg.edge_amp_scale)
             * np.cos(np.pi * x / 2) ** 2)
        return w / w.max()

    def _channel_phases(self) -> np.ndarray:
        cfg = self.cfg
        n   = len(self.channels)
        if not cfg.use_random_phase:
            return np.zeros(n)
        rng = np.random.default_rng(cfg.rng_seed)
        return 2 * np.pi * rng.random(n)

    @staticmethod
    def _channel_metadata(idx: int, ch: ChannelConfig,
                          total_phase: float, amp: float) -> dict:
        row = {
            'ChannelIndex':   idx + 1,
            'WaveformType':   ch.waveform_type.name,
            'CenterFreqHz':   ch.center_freq_hz,
            'Amplitude':      amp,
            'InitPhaseRad':   total_phase,
        }
        row.update({f'param_{k}': v for k, v in ch.params.items()
                    if not isinstance(v, (list, np.ndarray))})
        return row

    def _normalise(self, iq: np.ndarray):
        peak = np.max(np.abs(iq))
        rms  = np.sqrt(np.mean(np.abs(iq) ** 2))
        cf   = 20 * np.log10(peak / rms) if rms > 0 else 0.0
        if peak == 0:
            raise ValueError('Composite waveform is all zeros – check channel config.')
        iq_norm = iq / peak * self.cfg.final_peak_scale
        return iq_norm, {'peak_before': peak, 'rms_before': rms, 'cf_dB': cf}

    def _print_summary(self, iq: np.ndarray, stats: dict):
        cfg = self.cfg
        NsPulse = int(round(cfg.fs * cfg.pulse_width_s))
        NsPRI   = int(round(cfg.fs * cfg.pri_s))
        peak_a  = np.max(np.abs(iq))
        rms_a   = np.sqrt(np.mean(np.abs(iq) ** 2))
        cf_a    = 20 * np.log10(peak_a / rms_a) if rms_a > 0 else 0.0

        print(f'\n{"="*46}')
        print(f' WAVEFORM SUMMARY  –  {cfg.base_file_name}')
        print(f'{"="*46}')
        print(f'  Sample rate:          {cfg.fs/1e6:.3f} MSa/s')
        print(f'  Pulse width:          {cfg.pulse_width_s*1e6:.3f} µs')
        print(f'  PRI:                  {cfg.pri_s*1e6:.3f} µs')
        print(f'  Num pulses:           {cfg.num_pulses}')
        print(f'  Samples/pulse:        {NsPulse}')
        print(f'  Samples/PRI:          {NsPRI}')
        print(f'  Total samples:        {len(iq)}')
        print(f'  Channels:             {len(self.channels)}')
        types = {}
        for c in self.channels:
            types[c.waveform_type.name] = types.get(c.waveform_type.name, 0) + 1
        for k, v in types.items():
            print(f'    {k:<12} {v} ch')
        print(f'  RF centre (notional): {cfg.rf_center_hz/1e9:.6f} GHz')
        print(f'  Crest factor (before):{stats["cf_dB"]:.2f} dB')
        print(f'  Crest factor (after): {cf_a:.2f} dB')
        print(f'  Peak scale applied:   {cfg.final_peak_scale:.3f}')
        print(f'{"="*46}\n')


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTER
# ─────────────────────────────────────────────────────────────────────────────

class WaveformPlotter:
    """All diagnostic plots for a composite waveform."""

    def __init__(self, fs: float):
        self.fs = fs

    def plot_all(self, iq: np.ndarray, channels: List[ChannelConfig],
                 title_prefix: str = ''):
        """Convenience: produce all standard plots."""
        cfg     = self.fs
        NsPRI   = len(iq)   # one-PRI length not directly available; use full for spectrum
        t_all   = np.arange(len(iq)) / self.fs

        # For time-domain plots grab first PRI
        pri_samps = self._detect_pri(iq)
        one_pri   = iq[:pri_samps]
        t_pri     = np.arange(pri_samps) / self.fs

        self._plot_time_domain(one_pri, t_pri, title_prefix)
        self._plot_spectrum(iq, channels, title_prefix)
        self._plot_spectrogram(iq, channels, title_prefix)
        plt.show()

    # ── individual plots ─────────────────────────────────────────────────────

    def _plot_time_domain(self, one_pri: np.ndarray, t: np.ndarray,
                          prefix: str):
        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        fig.suptitle(f'{prefix}  –  One PRI (time domain)', fontsize=12)

        axes[0].plot(t * 1e3, np.real(one_pri), linewidth=0.7)
        axes[0].set_ylabel('I'); axes[0].grid(True)
        axes[0].set_title('In-phase (I)')

        axes[1].plot(t * 1e3, np.imag(one_pri), linewidth=0.7, color='C1')
        axes[1].set_ylabel('Q'); axes[1].grid(True)
        axes[1].set_title('Quadrature (Q)')

        axes[2].plot(t * 1e3, np.abs(one_pri), linewidth=0.7, color='C2')
        axes[2].set_xlabel('Time [ms]')
        axes[2].set_ylabel('|Envelope|'); axes[2].grid(True)
        axes[2].set_title('Envelope')

        plt.tight_layout()

    def _plot_spectrum(self, iq: np.ndarray, channels: List[ChannelConfig],
                       prefix: str):
        Nfft  = 2 ** int(np.ceil(np.log2(max(8192, min(len(iq), 2 ** 20)))))
        S     = np.fft.fftshift(np.fft.fft(iq[:Nfft], Nfft))
        f_ax  = (np.arange(Nfft) - Nfft // 2) * (self.fs / Nfft)
        dB    = 20 * np.log10(np.abs(S) + 1e-12)

        # determine sensible xlim from channel extents
        f_min, f_max = self._freq_bounds(channels)

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(f_ax / 1e6, dB, linewidth=0.7)
        ax.set_xlabel('Frequency [MHz]')
        ax.set_ylabel('Magnitude [dB]')
        ax.set_title(f'{prefix}  –  Spectrum')
        ax.set_xlim(f_min / 1e6, f_max / 1e6)
        ax.grid(True)
        plt.tight_layout()

    def _plot_spectrogram(self, iq: np.ndarray, channels: List[ChannelConfig],
                          prefix: str):
        # Cap at 3 PRIs worth of data to avoid MemoryError on long waveforms
        MAX_SPEC_SAMPLES = 2 ** 17          # ~131 k samples  ≈ 0.87 ms @ 150 MHz
        iq_spec = iq[:min(len(iq), MAX_SPEC_SAMPLES)]
        f_sg, t_sg, Sxx = sig.spectrogram(
            iq_spec, fs=self.fs,
            window=sig.windows.hann(1024),
            nperseg=1024, noverlap=768, nfft=2048,
            return_onesided=False, scaling='spectrum',
        )
        f_sg = np.fft.fftshift(f_sg)
        Sxx  = np.fft.fftshift(Sxx, axes=0)
        f_min, f_max = self._freq_bounds(channels)

        fig, ax = plt.subplots(figsize=(14, 5))
        im = ax.pcolormesh(t_sg * 1e3, f_sg / 1e6,
                           10 * np.log10(np.abs(Sxx) + 1e-12),
                           shading='gouraud', cmap='inferno')
        plt.colorbar(im, ax=ax, label='Power [dB]')
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Frequency [MHz]')
        ax.set_title(f'{prefix}  –  Spectrogram')
        ax.set_ylim(f_min / 1e6, f_max / 1e6)
        plt.tight_layout()

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_pri(iq: np.ndarray) -> int:
        """Heuristic: return length of iq (single-PRI) or a capped slice."""
        return min(len(iq), 2 ** 17)   # cap at ~0.87M samples for display

    @staticmethod
    def _freq_bounds(channels: List[ChannelConfig], margin: float = 0.1):
        """Return (f_min, f_max) in Hz with a fractional margin."""
        if not channels:
            return -75e6, 75e6
        edges = []
        for ch in channels:
            bw = ch.params.get('bandwidth_hz', 0)
            edges.append(ch.center_freq_hz - bw / 2)
            edges.append(ch.center_freq_hz + bw / 2)
        span = max(edges) - min(edges)
        mg   = max(span * margin, 1e6)
        return min(edges) - mg, max(edges) + mg


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTER
# ─────────────────────────────────────────────────────────────────────────────

class WaveformExporter:
    """
    Saves normalised IQ waveform to .MAT / .CSV / .BIN.

    Two .BIN files are always produced when save_bin is True:
      {base}_full.bin        – full resolution, original sample rate
      {base}_mxg_4mb.bin     – resampled to fit ≤ 4 MB for MXG download limit

    A companion .txt sidecar is written next to the 4 MB file recording
    the resampled sample rate to set on the MXG.
    """

    MXG_LIMIT_MB = 4.0   # MXG download size limit

    def save_all(self, iq: np.ndarray, config: CompositeConfig,
                 channel_table: pd.DataFrame, max_bin_mb: float = 0.0):
        """
        Save all enabled formats.

        max_bin_mb : override the default 4 MB MXG limit if needed (0 = use default).
        """
        I    = np.real(iq).astype(np.float64)
        Q    = np.imag(iq).astype(np.float64)
        base = config.base_file_name
        limit_mb = max_bin_mb if max_bin_mb > 0 else self.MXG_LIMIT_MB

        if config.save_mat:
            self._save_mat(base, iq, I, Q, config, channel_table)
        if config.save_csv:
            self._save_csv(base, I, Q, channel_table)
        if config.save_bin:
            # ── File 1: full resolution ──────────────────────────────────────
            self._save_bin(f'{base}_full', I, Q)
            full_mb = os.path.getsize(f'{base}_full.bin') / 1e6

            # ── File 2: resampled to MXG limit ───────────────────────────────
            iq_small, fs_small = resample_to_max_mb(iq, config.fs, limit_mb)
            I_s = np.real(iq_small).astype(np.float64)
            Q_s = np.imag(iq_small).astype(np.float64)
            self._save_bin(f'{base}_mxg_{int(limit_mb)}mb', I_s, Q_s)
            small_mb = os.path.getsize(f'{base}_mxg_{int(limit_mb)}mb.bin') / 1e6

            # ── Sidecar: MXG sample rate note ────────────────────────────────
            sidecar = f'{base}_mxg_{int(limit_mb)}mb_info.txt'
            with open(sidecar, 'w') as f:
                f.write(f'MXG waveform file info\n')
                f.write(f'======================\n')
                f.write(f'File          : {base}_mxg_{int(limit_mb)}mb.bin\n')
                f.write(f'File size     : {small_mb:.2f} MB\n')
                f.write(f'Samples       : {len(iq_small):,}\n')
                f.write(f'Sample rate   : {fs_small/1e6:.6f} MHz\n')
                f.write(f'  *** Set this sample rate on the MXG ***\n')
                f.write(f'Original fs   : {config.fs/1e6:.6f} MHz\n')
                f.write(f'Downsample    : {config.fs/fs_small:.4f}x\n')
                f.write(f'Format        : interleaved float32 I/Q (I0,Q0,I1,Q1,...)\n')

        print('\nFiles written:')
        if config.save_mat:
            mat_mb = os.path.getsize(f'{base}.mat') / 1e6
            print(f'  {base}.mat  ({mat_mb:.1f} MB)')
        if config.save_csv:
            print(f'  {base}.csv')
            print(f'  {base}_channel_table.csv')
        if config.save_bin:
            print(f'  {base}_full.bin            ({full_mb:.2f} MB)  '
                  f'@ {config.fs/1e6:.3f} MHz  – archive / full resolution')
            print(f'  {base}_mxg_{int(limit_mb)}mb.bin   ({small_mb:.2f} MB)  '
                  f'@ {fs_small/1e6:.3f} MHz  – load this onto MXG')
            print(f'  {base}_mxg_{int(limit_mb)}mb_info.txt  – MXG sample rate note')

    # ── format writers ───────────────────────────────────────────────────────

    @staticmethod
    def _save_mat(base, iq, I, Q, cfg, table):
        mat = {
            'iq': iq, 'I': I, 'Q': Q,
            'fs': cfg.fs, 'num_pulses': cfg.num_pulses,
            'pulse_width_s': cfg.pulse_width_s, 'pri_s': cfg.pri_s,
            'rf_center_hz': cfg.rf_center_hz,
            'final_peak_scale': cfg.final_peak_scale,
            'window_type': cfg.window_type,
        }
        for col in table.columns:
            safe_key = col.replace(' ', '_')
            vals = table[col].values
            if vals.dtype == object:
                vals = np.array([str(v) for v in vals])
            mat[f'chan_{safe_key}'] = vals
        sio.savemat(f'{base}.mat', mat)

    @staticmethod
    def _save_csv(base, I, Q, table):
        iq_arr = np.column_stack([I, Q])
        np.savetxt(f'{base}.csv', iq_arr, delimiter=',', header='I,Q', comments='')
        table.to_csv(f'{base}_channel_table.csv', index=False)

    @staticmethod
    def _save_bin(base, I, Q):
        buf = np.empty(2 * len(I), dtype=np.float32)
        buf[0::2] = I.astype(np.float32)
        buf[1::2] = Q.astype(np.float32)
        buf.tofile(f'{base}.bin')


# ─────────────────────────────────────────────────────────────────────────────
# RESAMPLER UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def resample_to_max_mb(iq: np.ndarray, fs: float,
                       max_mb: float = 0.0):
    """
    Downsample a complex IQ array so its float32 interleaved .BIN size stays
    within max_mb megabytes.  Returns (iq_out, new_fs).

    If max_mb is 0 or the array already fits, the original array and fs are
    returned unchanged.

    The resampler uses scipy.signal.resample (FFT-based, handles complex arrays)
    and prints the new sample rate so you can update the MXG accordingly.
    Nyquist check: warns if the new fs would alias the signal.
    """
    if max_mb <= 0:
        return iq, fs                         # no limit requested

    bytes_per_sample = 8                      # 2 × float32 (I + Q)
    max_samples = int(max_mb * 1e6 / bytes_per_sample)
    current_samples = len(iq)

    if current_samples <= max_samples:
        print(f'BIN size already within {max_mb} MB – no resampling needed.')
        return iq, fs

    ratio    = max_samples / current_samples
    new_fs   = fs * ratio
    new_n    = max_samples

    print(f'\nResampling BIN for size limit:')
    print(f'  Original : {current_samples:,} samples  @ {fs/1e6:.3f} MHz  '
          f'= {current_samples * bytes_per_sample / 1e6:.2f} MB')
    print(f'  Target   : {new_n:,} samples  @ {new_fs/1e6:.3f} MHz  '
          f'= {new_n * bytes_per_sample / 1e6:.2f} MB')
    print(f'  Ratio    : {ratio:.4f}  (downsample by {1/ratio:.2f}x)')

    # Warn if new sample rate may alias the signal
    # (rough check: assume occupied BW ≈ 80 % of original fs)
    estimated_bw = fs * 0.80
    if new_fs < estimated_bw * 1.1:
        print(f'  WARNING: new fs ({new_fs/1e6:.1f} MHz) may be close to or '
              f'below the signal bandwidth – check for aliasing.')

    # scipy.signal.resample handles complex arrays natively
    iq_out = sig.resample(iq, new_n)

    print(f'  Done.  New fs = {new_fs/1e6:.3f} MHz  '
          f'→ set this as the MXG sample rate.\n')
    return iq_out, new_fs


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE RUN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run(config: CompositeConfig, channels: List[ChannelConfig],
        plot: bool = True, max_bin_mb: float = 0.0) -> tuple:
    """
    Build, plot, and export a composite waveform in one call.

    Parameters
    ----------
    config      : CompositeConfig
    channels    : list of ChannelConfig
    plot        : whether to show diagnostic plots
    max_bin_mb  : if > 0, resample the .BIN to fit within this many MB

    Returns
    -------
    (iq, channel_table)
    """
    builder = CompositeBuilder(config, channels)
    iq, table = builder.build()

    if plot:
        plotter = WaveformPlotter(config.fs)
        plotter.plot_all(iq, channels, title_prefix=config.base_file_name)

    exporter = WaveformExporter()
    exporter.save_all(iq, config, table, max_bin_mb=max_bin_mb)

    return iq, table


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER INPUT GUI
# ─────────────────────────────────────────────────────────────────────────────

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os


class WaveformGUI:
    """
    Tkinter parameter-entry window for the MXG waveform designer.
    Opens when the script is run directly.  Fill in the fields and click
    'Build Waveform' – the waveform is generated, plots are shown, and
    files are written to the chosen output folder.
    """

    # ── Agilent / Keysight instrument profiles ───────────────────────────────
    # Keys shown in the dropdown.
    # max_bw_hz    : maximum I/Q modulation bandwidth the instrument supports
    # max_fs_hz    : maximum ARB sample rate
    # rec_fs_hz    : recommended sample rate (max_bw_hz / 0.80 guard band,
    #                capped at max_fs_hz)
    # max_mem_msamp: ARB memory in Msamples (affects max waveform length)
    # freq_range   : RF frequency coverage (documentation only)
    INSTRUMENT_PROFILES = {
        '-- Custom (manual entry) --': {
            'max_bw_hz':     None,
            'max_fs_hz':     None,
            'rec_fs_hz':     None,
            'max_mem_msamp': None,
            'freq_range':    '—',
            'notes':         'No automatic limits applied',
        },
        # ── First-generation MXG (N5182A) ───────────────────────────────────
        # Source: Keysight 5991-0192EN datasheet, Table 2
        'N5182A MXG  –  100 MHz BW  [1st gen]': {
            'max_bw_hz':     100e6,    # 100 MHz max internal modulation BW
            'max_fs_hz':     125e6,    # 125 MSa/s max ARB sample rate
            'rec_fs_hz':     125e6,    # 1.25 × 100 MHz = 125 MSa/s (capped at max_fs)
            'max_mem_msamp': 64,       # 8–64 MSa waveform memory
            'freq_range':    '250 kHz – 6 GHz',
            'notes':         '125 MSa/s max ARB | 100 MHz modulation BW | 8–64 MSa memory',
        },
        # ── EXG X-Series (N5172B) – replacement for N5182A ──────────────────
        # Source: Keysight 5991-0192EN datasheet, Table 2
        'N5172B EXG  –  120 MHz BW  [X-Series]': {
            'max_bw_hz':     120e6,    # 120 MHz max internal modulation BW
            'max_fs_hz':     150e6,    # 150 MSa/s max ARB sample rate
            'rec_fs_hz':     150e6,    # 1.25 × 120 MHz = 150 MSa/s (capped at max_fs)
            'max_mem_msamp': 512,      # 32–512 MSa waveform memory
            'freq_range':    '9 kHz – 6 GHz',
            'notes':         '150 MSa/s max ARB | 120 MHz modulation BW | up to 512 MSa memory',
        },
        'N5172B EXG  –  200 MHz BW  (external I/Q)': {
            'max_bw_hz':     200e6,    # 200 MHz nominal 3 dB BW via ext I/Q
            'max_fs_hz':     150e6,    # ARB clock max (ext I/Q bypasses internal filter)
            'rec_fs_hz':     150e6,    # capped at max_fs
            'max_mem_msamp': 512,
            'freq_range':    '9 kHz – 6 GHz',
            'notes':         'External I/Q input path; 200 MHz nominal 3 dB BW',
        },
        # ── MXG X-Series (N5182B) ────────────────────────────────────────────
        'N5182B MXG  –  80 MHz BW  (standard)': {
            'max_bw_hz':     80e6,
            'max_fs_hz':     200e6,
            'rec_fs_hz':     100e6,    # 1.25 × 80 MHz = 100 MSa/s
            'max_mem_msamp': 512,
            'freq_range':    '9 kHz – 6 GHz',
            'notes':         '200 MSa/s max ARB | 80 MHz modulation BW standard',
        },
        'N5182B MXG  –  160 MHz BW  (opt 1EL)': {
            'max_bw_hz':     160e6,
            'max_fs_hz':     200e6,
            'rec_fs_hz':     200e6,    # 1.25 × 160 MHz = 200 MSa/s (capped at max_fs)
            'max_mem_msamp': 512,
            'freq_range':    '9 kHz – 6 GHz',
            'notes':         'Option 1EL: 160 MHz wideband baseband | 200 MSa/s max ARB',
        },
        # ── PSG / ESG ────────────────────────────────────────────────────────
        'E8267D PSG  –  80 MHz BW  (standard)': {
            'max_bw_hz':     80e6,
            'max_fs_hz':     200e6,
            'rec_fs_hz':     100e6,    # 1.25 × 80 MHz = 100 MSa/s
            'max_mem_msamp': 64,
            'freq_range':    '100 kHz – 44 GHz',
            'notes':         'High-performance PSG vector; 80 MHz modulation BW',
        },
        'E8267D PSG  –  2 GHz BW  (opt H1E)': {
            'max_bw_hz':     2000e6,
            'max_fs_hz':     200e6,
            'rec_fs_hz':     200e6,    # capped at max_fs
            'max_mem_msamp': 64,
            'freq_range':    '100 kHz – 44 GHz',
            'notes':         'Option H1E: 2 GHz BB via external I/Q',
        },
        'E4438C ESG  –  80 MHz BW': {
            'max_bw_hz':     80e6,
            'max_fs_hz':     100e6,
            'rec_fs_hz':     100e6,    # 1.25 × 80 MHz = 100 MSa/s (capped at max_fs)
            'max_mem_msamp': 32,
            'freq_range':    '250 kHz – 6 GHz',
            'notes':         'Legacy ESG; 100 MSa/s max ARB | 80 MHz modulation BW',
        },
        # ── PXI ─────────────────────────────────────────────────────────────
        'M9381A PXI VSG  –  160 MHz BW': {
            'max_bw_hz':     160e6,
            'max_fs_hz':     200e6,
            'rec_fs_hz':     200e6,    # 1.25 × 160 MHz = 200 MSa/s (capped at max_fs)
            'max_mem_msamp': 256,
            'freq_range':    '1 MHz – 3 GHz',
            'notes':         'PXI modular VSG | 160 MHz modulation BW',
        },
    }

    # ── waveform-type-specific extra fields ─────────────────────────────────
    _EXTRA_FIELDS = {
        'LFM':     [],
        'NLFM':    [('NLFM law',  'law',      'str',   'tangent'),
                    ('Beta',      'beta',     'float',  1.8)],
        'CW':      [],
        'FMCW':    [('Shape',     'shape',    'str',   'sawtooth')],
        'STEPPED': [('Num steps', 'n_steps',  'int',    8)],
        'BPSK':    [('Chip rate (MHz)', 'chip_rate_mhz', 'float', 1.0)],
        'FRANK':   [('Order (M)',  'order',   'int',    4),
                    ('Chip rate (MHz)', 'chip_rate_mhz', 'float', 2.0)],
    }

    def __init__(self):
        self.root = tk.Tk()
        self.root.title('MXG Waveform Designer')
        self.root.resizable(False, False)
        self._vars: dict        = {}
        self._extra_frame       = None
        self._extra_vars: dict  = {}
        self._bw_status_label   = None   # live BW indicator (occupied vs mod BW)
        self._fs_warn_label     = None   # live sample-rate vs mod BW warning
        self._instr_notes_label = None   # instrument notes
        self._build_ui()
        self.root.mainloop()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill='both', expand=True, padx=8, pady=8)

        self._tab_timing(nb)
        self._tab_channels(nb)
        self._tab_window(nb)
        self._tab_output(nb)

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill='x', padx=8, pady=(0, 8))
        ttk.Button(btn_frame, text='Build Waveform',
                   command=self._on_build).pack(side='right', padx=4)
        ttk.Button(btn_frame, text='Cancel',
                   command=self.root.destroy).pack(side='right')

    def _tab_timing(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text='Timing & RF')

        # ── Instrument selection ─────────────────────────────────────────────
        instr_frame = ttk.LabelFrame(f, text='Instrument', padding=6)
        instr_frame.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 8))

        instr_names = list(self.INSTRUMENT_PROFILES.keys())
        self._vars['instrument'] = tk.StringVar(value=instr_names[0])
        instr_cb = ttk.Combobox(instr_frame,
                                textvariable=self._vars['instrument'],
                                values=instr_names, state='readonly', width=46)
        instr_cb.grid(row=0, column=0, columnspan=3, sticky='ew', pady=2)
        instr_cb.bind('<<ComboboxSelected>>', self._on_instrument_change)

        # Instrument info row
        ttk.Label(instr_frame, text='RF range:').grid(
            row=1, column=0, sticky='w', padx=(0, 4))
        self._instr_freq_label = ttk.Label(instr_frame, text='—', foreground='#555')
        self._instr_freq_label.grid(row=1, column=1, sticky='w')

        ttk.Label(instr_frame, text='Max fs:').grid(
            row=1, column=2, sticky='w', padx=(12, 4))
        self._instr_fs_label = ttk.Label(instr_frame, text='—', foreground='#555')
        self._instr_fs_label.grid(row=1, column=3, sticky='w')

        ttk.Label(instr_frame, text='Notes:').grid(
            row=2, column=0, sticky='w', padx=(0, 4))
        self._instr_notes_label = ttk.Label(instr_frame, text='No limits applied',
                                            foreground='grey')
        self._instr_notes_label.grid(row=2, column=1, columnspan=3, sticky='w')

        # ── Timing fields ────────────────────────────────────────────────────
        timing_fields = [
            ('Sample rate (MHz)',  'fs_mhz',    150.0),
            ('Pulse width (µs)',   'pw_us',    1000.0),
            ('PRI (µs)',           'pri_us',   2000.0),
            ('Num pulses',         'num_pulses',   20),
            ('RF centre (GHz)',    'rf_ghz',      2.4),
        ]
        self._make_fields(f, timing_fields, start_row=1)

        # Attach live BW check to fs field changes
        self._vars['fs_mhz'].trace_add('write', lambda *_: self._update_bw_status())

    def _tab_channels(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text='Channels')

        # Waveform type dropdown
        ttk.Label(f, text='Waveform type').grid(row=0, column=0, sticky='w', pady=2)
        wf_types = [t.name for t in WaveformType]
        self._vars['waveform_type'] = tk.StringVar(value='LFM')
        cb = ttk.Combobox(f, textvariable=self._vars['waveform_type'],
                          values=wf_types, state='readonly', width=14)
        cb.grid(row=0, column=1, sticky='w', pady=2)
        cb.bind('<<ComboboxSelected>>', self._on_type_change)

        fields = [
            ('Num channels',       'n_channels',  40),
            ('Chan spacing (MHz)', 'spacing_mhz',  3.0),
            ('Bandwidth/ch (MHz)', 'bw_mhz',       2.0),
        ]
        self._make_fields(f, fields, start_row=1)

        # Attach live BW check whenever channel params change
        for key in ('n_channels', 'spacing_mhz', 'bw_mhz'):
            self._vars[key].trace_add('write', lambda *_: self._update_bw_status())

        # Up/down chirp checkbox
        self._vars['up_chirp'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text='Up-chirp', variable=self._vars['up_chirp']
                        ).grid(row=4, column=0, columnspan=2, sticky='w', pady=2)

        # ── Live bandwidth status bar ────────────────────────────────────────
        bw_frame = ttk.LabelFrame(f, text='Bandwidth check', padding=6)
        bw_frame.grid(row=5, column=0, columnspan=2, sticky='ew', pady=(8, 0))

        self._bw_status_label = tk.Label(
            bw_frame,
            text='Select an instrument on the Timing & RF tab',
            font=('Segoe UI', 9, 'bold'),
            fg='grey', anchor='w', justify='left')
        self._bw_status_label.pack(fill='x')

        self._max_ch_label = tk.Label(
            bw_frame, text='', fg='#444', anchor='w')
        self._max_ch_label.pack(fill='x')

        self._fs_warn_label = tk.Label(
            bw_frame, text='', fg='red',
            font=('Segoe UI', 9, 'bold'), anchor='w', justify='left')
        self._fs_warn_label.pack(fill='x')

        # Frame for dynamic waveform-type extras
        self._extra_frame = ttk.LabelFrame(f, text='Waveform options', padding=6)
        self._extra_frame.grid(row=6, column=0, columnspan=2,
                               sticky='ew', pady=(6, 0))
        self._refresh_extra_fields('LFM')

    def _tab_window(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text='Window & Scale')

        self._vars['use_window'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text='Apply pulse window',
                        variable=self._vars['use_window']
                        ).grid(row=0, column=0, columnspan=2, sticky='w', pady=2)

        ttk.Label(f, text='Window type').grid(row=1, column=0, sticky='w', pady=2)
        self._vars['window_type'] = tk.StringVar(value='hann')
        ttk.Combobox(f, textvariable=self._vars['window_type'],
                     values=['hann', 'hamming', 'tukey', 'none'],
                     state='readonly', width=14
                     ).grid(row=1, column=1, sticky='w', pady=2)

        fields = [
            ('Tukey alpha',         'tukey_alpha',  0.25),
            ('Peak scale (0–1)',    'peak_scale',   0.85),
            ('RNG seed',            'rng_seed',     12345),
        ]
        self._make_fields(f, fields, start_row=2)

        self._vars['use_random_phase'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text='Random per-channel phase',
                        variable=self._vars['use_random_phase']
                        ).grid(row=5, column=0, columnspan=2, sticky='w', pady=2)

        self._vars['use_amp_taper'] = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text='Channel amplitude taper',
                        variable=self._vars['use_amp_taper']
                        ).grid(row=6, column=0, columnspan=2, sticky='w', pady=2)

    def _tab_output(self, nb):
        f = ttk.Frame(nb, padding=10)
        nb.add(f, text='Output')

        # Output folder
        ttk.Label(f, text='Output folder').grid(row=0, column=0, sticky='w', pady=2)
        self._vars['out_dir'] = tk.StringVar(
            value=os.path.dirname(os.path.abspath(__file__)))
        ttk.Entry(f, textvariable=self._vars['out_dir'], width=28
                  ).grid(row=0, column=1, sticky='ew', pady=2)
        ttk.Button(f, text='Browse…',
                   command=self._browse_folder).grid(row=0, column=2, padx=4)

        # File name stem
        ttk.Label(f, text='File name').grid(row=1, column=0, sticky='w', pady=2)
        self._vars['file_name'] = tk.StringVar(value='mxg_waveform')
        ttk.Entry(f, textvariable=self._vars['file_name'], width=28
                  ).grid(row=1, column=1, sticky='ew', pady=2)

        # Save format checkboxes
        self._vars['save_mat'] = tk.BooleanVar(value=True)
        self._vars['save_csv'] = tk.BooleanVar(value=True)
        self._vars['save_bin'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text='Save .MAT',
                        variable=self._vars['save_mat']
                        ).grid(row=2, column=0, sticky='w')
        ttk.Checkbutton(f, text='Save .CSV',
                        variable=self._vars['save_csv']
                        ).grid(row=3, column=0, sticky='w')
        ttk.Checkbutton(f, text='Save .BIN  (float32 I/Q for MXG)',
                        variable=self._vars['save_bin']
                        ).grid(row=4, column=0, columnspan=3, sticky='w')

        # BIN size limit
        sep = ttk.Separator(f, orient='horizontal')
        sep.grid(row=5, column=0, columnspan=3, sticky='ew', pady=8)

        ttk.Label(f, text='MXG BIN limit (MB)').grid(
            row=6, column=0, sticky='w', pady=2)
        self._vars['max_bin_mb'] = tk.StringVar(value='4')
        ttk.Entry(f, textvariable=self._vars['max_bin_mb'], width=10
                  ).grid(row=6, column=1, sticky='w', pady=2)
        ttk.Label(f, text='  always produces full + resampled pair', foreground='grey'
                  ).grid(row=6, column=2, sticky='w')

        self._vars['show_plots'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text='Show diagnostic plots',
                        variable=self._vars['show_plots']
                        ).grid(row=7, column=0, columnspan=3, sticky='w', pady=(4, 0))

    # ── helpers ──────────────────────────────────────────────────────────────

    def _make_fields(self, parent, fields, start_row=0):
        for i, (label, key, default) in enumerate(fields):
            r = start_row + i
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky='w', pady=2)
            var = tk.StringVar(value=str(default))
            self._vars[key] = var
            ttk.Entry(parent, textvariable=var, width=16
                      ).grid(row=r, column=1, sticky='w', pady=2)

    def _on_instrument_change(self, _event=None):
        """Auto-fill fs and update info labels when instrument is selected."""
        name    = self._vars['instrument'].get()
        profile = self.INSTRUMENT_PROFILES[name]

        # Update info labels
        freq = profile['freq_range']
        max_fs = profile['max_fs_hz']
        notes  = profile['notes']
        self._instr_freq_label.config(text=freq)
        self._instr_fs_label.config(
            text=f"{max_fs/1e6:.0f} MSa/s" if max_fs else '—')
        self._instr_notes_label.config(text=notes)

        # Auto-fill sample rate with recommended value
        if profile['rec_fs_hz']:
            self._vars['fs_mhz'].set(str(profile['rec_fs_hz'] / 1e6))

        self._update_bw_status()

    def _update_bw_status(self):
        """Recompute occupied BW and update the status label on the Channels tab."""
        if self._bw_status_label is None:
            return

        # Safely read current field values
        try:
            n   = int(float(self._vars['n_channels'].get()))
            sp  = float(self._vars['spacing_mhz'].get()) * 1e6
            bw  = float(self._vars['bw_mhz'].get()) * 1e6
            fs  = float(self._vars['fs_mhz'].get()) * 1e6
        except (ValueError, KeyError):
            return

        occupied = max(0, (n - 1) * sp + bw)

        # Get instrument limit
        name    = self._vars.get('instrument', tk.StringVar()).get()
        profile = self.INSTRUMENT_PROFILES.get(name, {})
        max_bw  = profile.get('max_bw_hz')
        max_fs  = profile.get('max_fs_hz')

        # Compute max channels that fit
        if max_bw and sp > 0:
            max_ch = max(1, int((max_bw - bw) / sp) + 1)
        else:
            max_ch = None

        # Build status text
        if max_bw is None:
            status = f'Occupied span: {occupied/1e6:.1f} MHz   (no instrument limit set)'
            color  = 'grey'
        elif occupied <= max_bw:
            pct    = occupied / max_bw * 100
            status = (f'✓  Occupied: {occupied/1e6:.1f} MHz  /  '
                      f'Limit: {max_bw/1e6:.0f} MHz  ({pct:.0f}% used)')
            color  = 'green'
        else:
            over   = (occupied - max_bw) / 1e6
            status = (f'⚠  EXCEEDS LIMIT by {over:.1f} MHz   '
                      f'Occupied: {occupied/1e6:.1f} MHz  /  '
                      f'Limit: {max_bw/1e6:.0f} MHz')
            color  = 'red'

        self._bw_status_label.config(text=status, fg=color)

        # Sample rate vs ARB clock max check
        fs_warn = ''
        fs_color = 'red'
        if max_fs and fs > max_fs:
            fs_warn = (f'⚠  Sample rate {fs/1e6:.0f} MHz > ARB clock max '
                       f'{max_fs/1e6:.0f} MHz — reduce sample rate.')

        # Max channels hint
        if max_ch is not None:
            self._max_ch_label.config(
                text=f'Max channels at {sp/1e6:.1f} MHz spacing: {max_ch}',
                fg='#444')

        if hasattr(self, '_fs_warn_label') and self._fs_warn_label:
            self._fs_warn_label.config(text=fs_warn, fg=fs_color if fs_warn else '#444')

    def _on_type_change(self, _event=None):
        self._refresh_extra_fields(self._vars['waveform_type'].get())

    def _refresh_extra_fields(self, wf_type: str):
        for w in self._extra_frame.winfo_children():
            w.destroy()
        self._extra_vars.clear()
        extras = self._EXTRA_FIELDS.get(wf_type, [])
        if not extras:
            ttk.Label(self._extra_frame, text='(no extra options)',
                      foreground='grey').grid(row=0, column=0, columnspan=2)
            return
        for i, (label, key, dtype, default) in enumerate(extras):
            ttk.Label(self._extra_frame, text=label).grid(
                row=i, column=0, sticky='w', pady=2, padx=(0, 8))
            var = tk.StringVar(value=str(default))
            self._extra_vars[key] = (var, dtype)
            ttk.Entry(self._extra_frame, textvariable=var, width=14
                      ).grid(row=i, column=1, sticky='w', pady=2)

    def _browse_folder(self):
        folder = filedialog.askdirectory(
            initialdir=self._vars['out_dir'].get(),
            title='Select output folder')
        if folder:
            self._vars['out_dir'].set(folder)

    # ── build action ─────────────────────────────────────────────────────────

    def _on_build(self):
        try:
            p = self._collect()
        except ValueError as e:
            messagebox.showerror('Input error', str(e))
            return

        self.root.destroy()

        cfg = CompositeConfig(
            fs               = p['fs_mhz'] * 1e6,
            num_pulses       = p['num_pulses'],
            pulse_width_s    = p['pw_us'] * 1e-6,
            pri_s            = p['pri_us'] * 1e-6,
            rf_center_hz     = p['rf_ghz'] * 1e9,
            use_window       = p['use_window'],
            window_type      = p['window_type'],
            tukey_alpha      = p['tukey_alpha'],
            use_random_phase = p['use_random_phase'],
            rng_seed         = p['rng_seed'],
            use_amp_taper    = p['use_amp_taper'],
            final_peak_scale = p['peak_scale'],
            base_file_name   = os.path.join(p['out_dir'], p['file_name']),
            save_mat         = p['save_mat'],
            save_csv         = p['save_csv'],
            save_bin         = p['save_bin'],
        )

        channels = self._build_channels(p)
        run(cfg, channels, plot=p['show_plots'], max_bin_mb=p['max_bin_mb'])

    def _collect(self) -> dict:
        """Read and validate all widget values."""
        def flt(key, label):
            try:    return float(self._vars[key].get())
            except: raise ValueError(f'"{label}" must be a number.')
        def integer(key, label):
            try:    return int(self._vars[key].get())
            except: raise ValueError(f'"{label}" must be an integer.')

        p = {
            'fs_mhz':          flt('fs_mhz',      'Sample rate'),
            'pw_us':           flt('pw_us',        'Pulse width'),
            'pri_us':          flt('pri_us',       'PRI'),
            'num_pulses':      integer('num_pulses','Num pulses'),
            'rf_ghz':          flt('rf_ghz',       'RF centre'),
            'n_channels':      integer('n_channels','Num channels'),
            'spacing_mhz':     flt('spacing_mhz', 'Chan spacing'),
            'bw_mhz':          flt('bw_mhz',      'Bandwidth'),
            'tukey_alpha':     flt('tukey_alpha',  'Tukey alpha'),
            'peak_scale':      flt('peak_scale',   'Peak scale'),
            'rng_seed':        integer('rng_seed', 'RNG seed'),
            'waveform_type':   self._vars['waveform_type'].get(),
            'window_type':     self._vars['window_type'].get(),
            'up_chirp':        self._vars['up_chirp'].get(),
            'use_window':      self._vars['use_window'].get(),
            'use_random_phase':self._vars['use_random_phase'].get(),
            'use_amp_taper':   self._vars['use_amp_taper'].get(),
            'save_mat':        self._vars['save_mat'].get(),
            'save_csv':        self._vars['save_csv'].get(),
            'save_bin':        self._vars['save_bin'].get(),
            'show_plots':      self._vars['show_plots'].get(),
            'out_dir':         self._vars['out_dir'].get(),
            'file_name':       self._vars['file_name'].get(),
            'max_bin_mb':      flt('max_bin_mb', 'Max BIN size'),
        }

        if p['pw_us'] >= p['pri_us']:
            raise ValueError('Pulse width must be less than PRI.')
        if p['peak_scale'] <= 0 or p['peak_scale'] > 1:
            raise ValueError('Peak scale must be between 0 and 1.')
        if not p['file_name'].strip():
            raise ValueError('File name cannot be empty.')
        if p['max_bin_mb'] < 0:
            raise ValueError('Max BIN size must be 0 (no limit) or a positive number.')

        # Instrument bandwidth check
        instr_name = self._vars['instrument'].get()
        profile    = self.INSTRUMENT_PROFILES.get(instr_name, {})
        max_bw     = profile.get('max_bw_hz')
        max_fs     = profile.get('max_fs_hz')
        occupied   = max(0, (p['n_channels'] - 1) * p['spacing_mhz'] * 1e6
                           + p['bw_mhz'] * 1e6)
        p['instrument']  = instr_name
        p['max_bw_hz']   = max_bw
        p['occupied_hz'] = occupied

        if max_bw and occupied > max_bw:
            over = (occupied - max_bw) / 1e6
            raise ValueError(
                f'Occupied bandwidth {occupied/1e6:.1f} MHz exceeds '
                f'{instr_name} limit of {max_bw/1e6:.0f} MHz '
                f'(by {over:.1f} MHz).\n\n'
                f'Reduce channels, spacing, or per-channel bandwidth.')
        if max_fs and p['fs_mhz'] * 1e6 > max_fs:
            raise ValueError(
                f'Sample rate {p["fs_mhz"]:.1f} MHz exceeds '
                f'{instr_name} ARB clock maximum of {max_fs/1e6:.0f} MHz.')

        # Extra type-specific fields
        p['extra'] = {}
        for key, (var, dtype) in self._extra_vars.items():
            raw = var.get()
            try:
                if dtype == 'int':   p['extra'][key] = int(raw)
                elif dtype == 'float': p['extra'][key] = float(raw)
                else:                p['extra'][key] = raw
            except ValueError:
                raise ValueError(f'Extra field "{key}" has invalid value.')
        return p

    def _build_channels(self, p: dict) -> List[ChannelConfig]:
        """Create a uniform channel bank from GUI parameters."""
        wf   = WaveformType[p['waveform_type']]
        n    = p['n_channels']
        sp   = p['spacing_mhz'] * 1e6
        bw   = p['bw_mhz'] * 1e6
        extra = p['extra']

        offsets = (np.arange(n) - (n - 1) / 2) * sp

        # Build base params dict
        params: dict = {'bandwidth_hz': bw, 'up_chirp': p['up_chirp']}

        if wf == WaveformType.NLFM:
            params['law']  = extra.get('law', 'tangent')
            params['beta'] = extra.get('beta', 1.8)

        elif wf == WaveformType.FMCW:
            params['shape'] = extra.get('shape', 'sawtooth')

        elif wf == WaveformType.STEPPED:
            params['n_steps'] = extra.get('n_steps', 8)
            # bandwidth_hz here means total stepped span
        elif wf == WaveformType.BPSK:
            cr = extra.get('chip_rate_mhz', 1.0) * 1e6
            params = {'code': [1,1,1,1,1,-1,-1,1,1,-1,1,-1,1],
                      'chip_rate_hz': cr}
        elif wf == WaveformType.FRANK:
            cr = extra.get('chip_rate_mhz', 2.0) * 1e6
            params = {'order': extra.get('order', 4), 'chip_rate_hz': cr}

        elif wf == WaveformType.CW:
            params = {}

        return [
            ChannelConfig(center_freq_hz=float(fc),
                          waveform_type=wf,
                          params=dict(params))
            for fc in offsets
        ]


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    WaveformGUI()
