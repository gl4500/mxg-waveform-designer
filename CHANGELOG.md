# MXG Waveform Designer — Changelog & Roadmap

## Format
`[vX.Y] YYYY-MM-DD — Summary`

---

## Released

### [v1.4] 2026-04-05 — Fix binary format: int16 big-endian
- **N5182A expects int16 big-endian, NOT float32.**
  Corrected `_save_bin()` to write `>i2` (big-endian int16) scaled to ±32767.
- Fixed `resample_to_max_mb()` bytes-per-sample from 8 → 4 (int16 = 2 B × 2 channels).
- Updated sidecar `_info.txt` and GUI checkbox label to reflect correct format.

### [v1.3] 2026-04-05 — Sample rate = 1.25 × BW (Nyquist oversampling)
- `rec_fs_hz` set to `min(1.25 × max_bw_hz, max_fs_hz)` across all instrument profiles.
- Removed erroneous `fs > modulation BW` hard block; only `fs > ARB clock max` is enforced.
- Live red warning fires only when `fs > max_fs_hz`.

### [v1.2] 2026-04-05 — Instrument profile corrections from Keysight 5991-0192EN
- **N5182A MXG:** 100 MHz modulation BW / 125 MSa/s max ARB (was 80 MHz / 100 MSa/s).
- **N5172B EXG:** 120 MHz modulation BW / 150 MSa/s max ARB (was 40 MHz / 200 MSa/s).
- Source: Keysight datasheet 5991-0192EN, Table 2.

### [v1.1] 2026-04-05 — Instrument dropdown + live BW checker
- Added `INSTRUMENT_PROFILES` dict covering N5182A, N5172B (×2), N5182B (×2),
  E8267D (×2), E4438C, M9381A.
- Selecting an instrument auto-fills sample rate and shows RF frequency range / notes.
- Live `_update_bw_status()` computes occupied span vs modulation BW limit:
  green ✓ when within limit, red ⚠ when exceeded.
- Hard block in `_collect()` prevents build if occupied BW or sample rate exceeds limits.

### [v1.0] 2026-04-05 — Initial release
- Multi-waveform IQ designer: LFM, NLFM, CW, FMCW, Stepped-Frequency, BPSK, Frank.
- Class-based `ChannelConfig` / `CompositeConfig` / `CompositeBuilder` architecture.
- `WaveformPlotter` with spectrogram capped at 2¹⁷ samples (avoids MemoryError).
- `WaveformExporter` always writes two `.bin` files per build:
  - `_full.bin` — full resolution archive.
  - `_mxg_4mb.bin` — FFT-resampled to ≤4 MB for MXG download limit.
  - `_info.txt` sidecar with resampled sample rate to set on instrument.
- tkinter GUI with tabs: Timing & RF, Channels, Window & Scale, Output.
- Uploaded to `gl4500/mxg-waveform-designer` on GitHub.

---

## Known Issues / Future Work

| # | Priority | Description |
|---|----------|-------------|
| 1 | High | Validate `.bin` file load on physical N5182A hardware |
| 2 | Medium | Add SCPI auto-download via `pyvisa` (GPIB/LAN) so user doesn't manually copy file |
| 3 | Medium | Add per-channel amplitude taper control (e.g., Taylor weighting across channels) |
| 4 | Medium | Support N5182A waveform header (file catalog name, marker data) if required |
| 5 | Low | Add pulse-Doppler mode: stepped PRI across pulses |
| 6 | Low | Export to Keysight 89600 VSA compatible format for offline analysis |
| 7 | Low | Add MATLAB `.m` script export via `py2mat.py` integration |

---

## Instrument Reference

| Instrument | Mod BW | Max ARB fs | rec fs (1.25×) | Memory |
|---|---|---|---|---|
| N5182A MXG 1st gen | 100 MHz | 125 MSa/s | 125 MSa/s | 8–64 MSa |
| N5172B EXG X-Series | 120 MHz | 150 MSa/s | 150 MSa/s | 32–512 MSa |
| N5172B EXG ext I/Q | 200 MHz | 150 MSa/s | 150 MSa/s | 32–512 MSa |
| N5182B MXG std | 80 MHz | 200 MSa/s | 100 MSa/s | up to 512 MSa |
| N5182B MXG opt 1EL | 160 MHz | 200 MSa/s | 200 MSa/s | up to 512 MSa |
| E8267D PSG std | 80 MHz | 200 MSa/s | 100 MSa/s | up to 64 MSa |
| E4438C ESG | 80 MHz | 100 MSa/s | 100 MSa/s | up to 32 MSa |
| M9381A PXI VSG | 160 MHz | 200 MSa/s | 200 MSa/s | up to 256 MSa |

---

## Binary Format Specification (N5182A)

```
Encoding  : int16, big-endian (MSB first)
Interleave: I₀, Q₀, I₁, Q₁, ... (sample-interleaved)
Scale     : ±32767 full scale
File ext  : .bin
```

**Python write:**
```python
buf = np.empty(2 * N, dtype='>i2')
buf[0::2] = np.clip(I * scale, -32767, 32767).astype('>i2')
buf[1::2] = np.clip(Q * scale, -32767, 32767).astype('>i2')
buf.tofile('waveform.bin')
```
