# MXG Waveform Designer

Multi-channel composite I/Q waveform designer for Agilent / Keysight MXG signal generators (N5182A / N5182B and compatible).  Supports LFM, NLFM, CW, FMCW, Stepped-Frequency, BPSK, and Frank polyphase waveforms — mix freely across channels.

---

## Quick Start

### Windows
Double-click **`install_and_run.bat`**

Creates a Python virtual environment, installs all dependencies, and launches the GUI automatically.  No Anaconda or pre-installed packages required — just Python 3.8+.

### macOS / Linux
```bash
bash install_and_run.sh
```

> **Note — tkinter on macOS:** if the GUI fails to open, install the Tk binding for your Python:
> - Homebrew Python: `brew install python-tk`
> - Debian/Ubuntu: `sudo apt install python3-tk`

### Manual launch (if you already have the dependencies)
```bash
python mxg_waveform_designer.py
```

### Dependencies
```
numpy >= 1.24
scipy >= 1.11
matplotlib >= 3.7
pandas >= 2.0
pyvisa >= 1.13   # optional — only needed for the SCPI auto-download tab
```

---

## GUI Walkthrough

The designer window has five tabs.

---

### Tab 1 — Timing & RF

| Field | Description |
|-------|-------------|
| **Instrument** | Select your instrument from the dropdown. Auto-fills the recommended sample rate and shows the modulation BW limit. Choose *Custom* to enter values manually. |
| **Sample rate (MHz)** | ARB sample rate. Must not exceed the instrument's ARB clock maximum. |
| **Pulse width (µs)** | On-time of the pulse within each PRI. Must be less than PRI. |
| **PRI (µs)** | Pulse Repetition Interval. Total length of one pulse + dead-time slot. |
| **Num pulses** | How many PRIs to concatenate in the output waveform. |
| **RF centre (GHz)** | Notional RF carrier — stored as metadata in the .mat and params files, not used in baseband generation. |

---

### Tab 2 — Channels

Defines the channel banks that make up the composite waveform.  Each bank is a uniform group of one waveform type.  Multiple banks of different types can be combined (e.g. LFM + BPSK + FRANK in one composite).

#### Bank list
Shows all currently defined banks.  At launch, one default LFM bank (40 channels) is pre-loaded.

| Button | Action |
|--------|--------|
| **Add bank** | Reads the entry form below and appends a new bank to the list. |
| **Remove selected** | Deletes the highlighted bank from the list. |

#### New bank entry form

| Field | Description |
|-------|-------------|
| **Waveform type** | LFM, NLFM, CW, FMCW, STEPPED, BPSK, or FRANK. |
| **Num channels** | Number of uniformly-spaced channels in this bank. |
| **Chan spacing (MHz)** | Frequency separation between adjacent channel centres. |
| **Bandwidth/ch (MHz)** | Per-channel chirp / sweep bandwidth (not used for CW). |
| **Up-chirp** | Chirp direction for LFM / NLFM / FMCW. |
| **Waveform options** | Type-specific extra fields (NLFM law, FMCW shape, step count, BPSK/Frank code params). |

#### Bandwidth check bar
Updates live as you change fields.  Shows occupied span vs the selected instrument's modulation BW limit — green when within limit, red when exceeded.  Build is blocked if the limit is exceeded.

---

### Tab 3 — Window & Scale

| Field | Description |
|-------|-------------|
| **Apply pulse window** | Enable/disable a time-domain envelope taper on each pulse. |
| **Window type** | `hann` (default), `hamming`, `tukey`, or `none`. |
| **Tukey alpha** | Shape parameter for the Tukey window (0 = rectangular, 1 = Hann). |
| **Peak scale (0–1)** | Final normalisation — peak IQ amplitude as a fraction of full scale. Default 0.85 leaves 1.4 dB headroom. |
| **RNG seed** | Seed for the random per-channel phase dither (for PAPR reduction). |
| **Random per-channel phase** | Enable random phase offsets across channels to reduce crest factor. |
| **Channel amplitude taper** | Apply a cross-channel amplitude weighting to the bank. |
| **Taper type** | `cosine` (legacy), `taylor` (true Taylor window), or `chebwin` (equiripple Chebyshev). |
| **Taylor nbar** | Number of constant-level inner sidelobes (Taylor / Chebwin). Default 4. |
| **SLL (dB, neg.)** | Desired sidelobe level, e.g. `-30`.  Used by both Taylor and Chebwin. |

---

### Tab 4 — Output

| Field | Action |
|-------|--------|
| **Output folder** | Directory where the `params/` subfolder will be created. Click **Browse…** to choose. |
| **File name** | Base stem for all output files.  Click **Auto-name** to generate a descriptive name from the waveform parameters automatically (format: `{types}_{N}ch_{fs}MHz_{pw}us_{npri}pri_{timestamp}`). |
| **Save .MAT** | MATLAB archive containing IQ data, config, and channel table. |
| **Save .CSV** | Raw I/Q columns as comma-separated text. |
| **Save .BIN** | Binary waveform files for the MXG (int16 big-endian, interleaved I/Q, ±32767 full scale). Always produces two files: `_full.bin` (full resolution) and `_mxg_4mb.bin` (resampled to fit the 4 MB MXG download limit). |
| **MXG BIN limit (MB)** | Maximum size of the resampled .bin file. Default 4 MB matches the N5182A download limit. |
| **Save 89600 VSA .mat** | Keysight 89600 VSA-compatible .mat file (`Y`, `XDelta`, `XStart`, `XUnit`, `YUnit`, `InputCenter`). Load via *File → Load Data File* in the 89600 VSA software. |
| **Save MATLAB .m script** | Generates a `_load.m` script that loads the .mat, plots a spectrogram, and shows `tcpclient`-based SCPI download commands for MATLAB Instrument Control Toolbox. |
| **Show diagnostic plots** | Display time-domain, spectrum, and spectrogram plots after the build. |

> **Auto-save:** A `_params.json` file is always written to `params/` regardless of the checkboxes above.  It captures the full configuration and channel layout so any build can be reproduced exactly.

---

### Tab 5 — SCPI

Direct waveform download to a connected N5182A over LAN or GPIB using PyVISA.

> Requires `pyvisa` and a backend:  `pip install pyvisa pyvisa-py`

| Field | Description |
|-------|-------------|
| **VISA resource** | Instrument address, e.g. `TCPIP0::192.168.1.100::5025::SOCKET` or `GPIB0::19::INSTR`. |
| **Waveform name** | Name stored in the instrument's waveform catalog (no spaces). |
| **Sample rate (MHz)** | Sample rate to set on the instrument after transfer. Must match the `_mxg_4mb_info.txt` value. |
| **Bin file path** | Path to the `_mxg_4mb.bin` file to transfer. Click **Browse…** to select. |
| **Connect & Download** | Transfers the .bin via `MMEM:DATA`, selects the waveform, sets the sample rate, and enables ARB output. |

---

## Building a Waveform — Step by Step

1. **Timing & RF tab** — select your instrument, set sample rate, pulse width, PRI, and number of pulses.
2. **Channels tab** — configure the default bank or add multiple banks.  Watch the bandwidth check bar.
3. **Window & Scale tab** — choose window type and taper settings.
4. **Output tab** — select output folder, click **Auto-name**, tick the formats you need.
5. Click **Build Waveform**.  The status bar shows progress.  When complete it shows the file path and sample count in green.
6. Load `_mxg_4mb.bin` onto the instrument manually or use the **SCPI** tab.

---

## Output Files

All files are written to `<output folder>/params/`.

| File | Description |
|------|-------------|
| `<stem>_full.bin` | Full-resolution binary waveform — archive copy. |
| `<stem>_mxg_4mb.bin` | Polyphase-resampled binary ≤ 4 MB — load this onto the MXG. |
| `<stem>_mxg_4mb_info.txt` | Sample rate to set on the instrument. |
| `<stem>_mxg_4mb_scpi.txt` | Ready-to-use SCPI command sequence for loading via PyVISA or IO Libraries Suite. |
| `<stem>.mat` | MATLAB archive (IQ, config, channel table). |
| `<stem>.csv` / `<stem>_channel_table.csv` | Raw I/Q data and per-channel metadata. |
| `<stem>_vsa89600.mat` | 89600 VSA import file (optional). |
| `<stem>_load.m` | MATLAB reconstruction and SCPI download script (optional). |
| `<stem>_params.json` | **Always saved.** Full config + channel layout + build stats. |

### Binary Format (N5182A)
```
Encoding  : int16, big-endian (MSB first)
Interleave: I₀, Q₀, I₁, Q₁, …  (sample-interleaved)
Scale     : ±32767 full scale
```

---

## Validate a .bin File

```bash
python mxg_bin_validate.py params/LFM_40ch_125MHz_1000us_20pri_<ts>_mxg_4mb.bin --fs 125
```

Prints a report (file size, IQ range, PAPR, endianness check, scale utilisation) and opens a four-panel diagnostic plot: time-domain I/Q, IQ constellation, amplitude envelope, and Welch PSD.

Optional flags:
```
--fs 125        sample rate in MSa/s (default 125)
--info <file>   path to the _mxg_4mb_info.txt sidecar (auto-detected if omitted)
--no-plot       print report only, skip plots
```

---

## Reproduce a Build from a params.json

```python
from mxg_waveform_designer import load_params, CompositeBuilder, WaveformExporter

cfg, channels = load_params('params/LFM_40ch_125MHz_1000us_20pri_<ts>_params.json')
iq, table = CompositeBuilder(cfg, channels).build()
WaveformExporter().save_all(iq, cfg, channels, table)
```

---

## Run Tests

```bash
# Windows (radioconda)
C:\Users\gl450\radioconda\python -m pytest tests/ -v

# Standard Python / venv
pytest tests/ -v
```

48 tests covering all waveform kernels, composite builder, binary format, resampler, PRI detection, and all export helpers.  No hardware required.

---

## Supported Instruments

| Instrument | Mod BW | Max ARB fs | Memory |
|---|---|---|---|
| N5182A MXG 1st gen | 100 MHz | 125 MSa/s | 8–64 MSa |
| N5172B EXG X-Series | 120 MHz | 150 MSa/s | 32–512 MSa |
| N5182B MXG std | 80 MHz | 200 MSa/s | up to 512 MSa |
| N5182B MXG opt 1EL | 160 MHz | 200 MSa/s | up to 512 MSa |
| E8267D PSG std | 80 MHz | 200 MSa/s | up to 64 MSa |
| E4438C ESG | 80 MHz | 100 MSa/s | up to 32 MSa |
| M9381A PXI VSG | 160 MHz | 200 MSa/s | up to 256 MSa |

---

## Version History

See [CHANGELOG.md](CHANGELOG.md) for the full release history.  Current version: **v1.16**.
