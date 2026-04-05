# MXG Waveform Designer — Changelog & Roadmap

## Format
`[vX.Y] YYYY-MM-DD — Summary`

---

## Released

### [v1.16] 2026-04-05 — All output files saved directly into params/ folder
- `base_file_name` now points into `<output_dir>/params/<stem>` so every file
  produced by a build (`.bin`, `.mat`, `.csv`, `_info.txt`, `_scpi.txt`,
  `_params.json`, `_vsa89600.mat`, `_load.m`) lands in the same `params/` folder.
- No more scattered files in the root output directory.
- `save_all()` calls `os.makedirs(..., exist_ok=True)` before the first write.
- `_save_params()` no longer creates a nested `params/params/` layer.

### [v1.15] 2026-04-05 — Auto-generated file names from waveform parameters
- Output tab now has an **Auto-name** button next to the file name field.
- Clicking it populates the field with a descriptive stem:
  `{types}_{N}ch_{fs}MHz_{pw}us_{npri}pri_{YYYYMMDD_HHMMSS}`
  e.g. `LFM_40ch_125MHz_1000us_20pri_20260405_143211`
  or   `LFM+BPSK_44ch_100MHz_500us_10pri_20260405_150022`
- If the file name is still the bare default (`mxg_waveform`) when Build is clicked,
  the name is auto-generated automatically — no silent overwrites.
- NLFM banks append the law suffix: `NLFM-tangent`, `NLFM-cosine`, `NLFM-hamming`.
- Mixed-bank composites join type names with `+`: `LFM+STEPPED+FRANK`.
- Timestamp in the stem guarantees every build produces a unique set of files.

### [v1.14] 2026-04-05 — Fix: figure windows frozen/unmoveable
- `plt.show()` → `plt.show(block=False)` in `WaveformPlotter.plot_all()`.
- Root cause: blocking `plt.show()` called inside tkinter's mainloop via `root.after()`
  caused matplotlib (TkAgg backend) and tkinter to fight over the same Tk event loop.
  Figure windows appeared but received no move/resize events — locked in place.
- Fix hands event processing back to tkinter's mainloop so all figure windows are
  fully interactive (move, resize, zoom, pan) alongside the designer GUI.

### [v1.13] 2026-04-05 — Fix: GUI no longer closes after build
- Removed `self.root.destroy()` from `_on_build()`.
- Build + export now run in a `daemon` background thread (`threading.Thread`).
- Build button disables and shows "Building…" during the run; re-enables on completion.
- Status bar added below buttons: shows progress → green "Done — \<file\> (\<samples\>, \<ms\>)
  Files saved to: \<dir\>" on success, red error message on failure.
- Plots scheduled back onto the main thread after worker finishes (`root.after(0, ...)`).
- "Cancel" renamed "Close" — window persists between builds.
- 48 tests still pass.

### [v1.12] 2026-04-05 — Auto-save waveform parameters to params/ subfolder
- `WaveformExporter.save_all()` writes `params/<stem>_params.json` automatically
  on every build — no checkbox, always on.
- JSON contains: full `CompositeConfig`, all channel configs (enum as string),
  derived stats (PAPR, total samples, duration, channel type counts), build timestamp,
  output directory.
- `load_params(path)` helper reconstructs `(CompositeConfig, [ChannelConfig])`
  from any params file for exact waveform reproduction on another machine.
- 6 new tests added (`TestParamsSave`) — 48 total, all pass.

### [v1.11] 2026-04-05 — Automated pytest test suite (42 tests)
- `test_mxg_waveform_designer.py` added covering all 7 kernel functions, `CompositeBuilder`,
  Taylor/Chebwin tapers, pulse-Doppler modes, binary format roundtrip, `resample_to_max_mb`,
  PRI auto-detect, `validate()` / endianness checks, VSA/89600/.m/SCPI export helpers.
- All 42 tests pass with `pytest -v` under radioconda (Python 3.12 + scipy/numpy/pandas).

### [v1.10] 2026-04-05 — MATLAB .m script export
- `WaveformExporter._save_m_script()` writes `{base}_load.m` when `save_m_script=True`.
- Script loads the .mat archive, runs `spectrogram()`, and shows a complete
  `tcpclient`-based SCPI download sequence for MATLAB Instrument Control Toolbox.
- GUI Output tab checkbox: "Save MATLAB .m script".

### [v1.9] 2026-04-05 — Keysight 89600 VSA .mat export
- `WaveformExporter._save_vsa89600()` writes `{base}_vsa89600.mat` when `save_vsa89600=True`.
- Variables: `Y` (complex column vector), `XDelta`, `XStart`, `XUnit='sec'`,
  `YUnit='V'`, `InputCenter` (RF centre Hz) — compatible with 89600 VSA "Load Data File".
- GUI Output tab checkbox: "Save 89600 VSA .mat".

### [v1.8] 2026-04-05 — N5182A SCPI sidecar commands file
- `WaveformExporter._save_scpi_sidecar()` writes `{base}_mxg_4mb_scpi.txt` alongside
  every .bin build.
- Contains exact `MMEM:DATA`, `WGEN:ARB:WAVEFORM`, `WGEN:ARB:SRAT`, `OUTPUT:STATE`
  commands with Python/PyVISA and raw SCPI forms.

### [v1.7] 2026-04-05 — pyvisa SCPI auto-download tab
- New "SCPI" tab in GUI with VISA resource field, waveform name, sample rate, and bin
  file browser.
- "Connect & Download to N5182A" transfers the .bin via `MMEM:DATA` block data and
  configures the instrument in one click.
- Graceful fallback if `pyvisa` is not installed (displays install instructions).

### [v1.6] 2026-04-05 — Pulse-Doppler staggered / jittered PRI
- `CompositeConfig` gains `use_pulse_doppler`, `pd_pri_step_s`, `pd_mode` ('stagger'|'jitter'),
  `pd_jitter_seed`.
- `stagger` mode: PRI grows linearly by `pd_pri_step_s` each pulse — resolves Doppler
  ambiguity at the cost of variable waveform length.
- `jitter` mode: PRI is randomly perturbed ±`pd_pri_step_s` — reduces range-Doppler
  coupling in dense emitter environments.

### [v1.5] 2026-04-05 — Mixed-type channel banks + Taylor/Chebwin taper + PRI detect + resample_poly
**Channel groups (GUI):**
- Channels tab redesigned with a bank list (Treeview). Each bank is an independent uniform
  group with its own waveform type + parameters.
- "Add bank" / "Remove selected" buttons. Default seeds one LFM bank on launch.
- Mixed composites (e.g. LFM + BPSK + FRANK) are now fully supported from the GUI.

**Taylor / Chebwin amplitude taper:**
- `CompositeConfig.taper_type` selects `'cosine'` (legacy) | `'taylor'` | `'chebwin'`.
- `taylor_nbar` (int, default 4) and `taylor_sll` (dB, default −30) control the Taylor
  window shape. Chebwin uses `|taylor_sll|` as its equiripple sidelobe attenuation.
- GUI Window & Scale tab exposes taper type dropdown plus nbar / SLL fields.

**PRI auto-detect:**
- `WaveformPlotter._detect_pri()` now uses envelope autocorrelation to find the true PRI
  period; falls back to 2^17-sample cap only when no pulse structure is detected (e.g. CW).
- `run()` passes `pri_samples=NsPRI` to `plot_all()` so the builder's known PRI takes
  precedence over detection when available.

**Polyphase resampler:**
- `resample_to_max_mb()` replaced `scipy.signal.resample` (FFT-based, Gibbs ringing at
  pulse edges) with `scipy.signal.resample_poly` + Kaiser FIR anti-alias filter.
- Rational p/q ratio computed via `Fraction(...).limit_denominator(1000)`.
- Output trimmed/padded by at most 1 sample to hit the exact 4 MB byte target.

**Welch PSD in mxg_bin_validate.py:**
- Power spectrum plot replaced with a Welch-averaged PSD (`scipy.signal.welch`).
- nperseg auto-scaled to `min(8192, N/8)`; number of averages shown in plot title.
- Provides a much cleaner spectral view for long waveform files (was single-window FFT).

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

| # | Priority | Description | Status |
|---|----------|-------------|--------|
| 1 | High | Validate `.bin` file load on physical N5182A hardware | **Open** — requires lab access |
| 2 | Medium | SCPI auto-download via `pyvisa` (GPIB/LAN) | **Done** v1.7 |
| 3 | Medium | Per-channel amplitude taper (Taylor / Chebwin) | **Done** v1.5 |
| 4 | Medium | N5182A waveform header / SCPI sidecar commands | **Done** v1.8 |
| 5 | Low | Pulse-Doppler mode: stagger / jitter PRI | **Done** v1.6 |
| 6 | Low | Export to Keysight 89600 VSA `.mat` format | **Done** v1.9 |
| 7 | Low | MATLAB `.m` script export | **Done** v1.10 |
| 8 | Medium | Mixed-type channel banks in GUI | **Done** v1.5 |
| 9 | Medium | Polyphase resampler (avoid FFT Gibbs ringing) | **Done** v1.5 |
| 10 | Medium | Real PRI auto-detect from waveform structure | **Done** v1.5 |
| 11 | Low | Automated pytest test suite | **Done** v1.11 (42 tests) |
| 12 | Low | Welch PSD in `mxg_bin_validate.py` | **Done** v1.5 |

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
