# MXG Waveform Designer

Multi-channel composite I/Q waveform builder for Keysight N5182A/N5182B MXG signal generators.

## Quick Start

**Windows**
```
install_and_run.bat
```

**macOS / Linux**
```
./install_and_run.sh
```

Both scripts create an isolated `.venv`, install dependencies from `requirements.txt`, and launch the GUI.

## Documentation

| Document | Location |
|----------|----------|
| User Guide (HTML) | [`docs/MXG_Waveform_Designer_User_Guide.html`](docs/MXG_Waveform_Designer_User_Guide.html) |
| Full README | [`docs/README.md`](docs/README.md) |
| Changelog | [`docs/CHANGELOG.md`](docs/CHANGELOG.md) |

## Repository Layout

```
mxg-waveform-designer/
├── mxg_waveform_designer.py   — Main GUI application
├── install_and_run.bat        — Windows one-click launcher
├── install_and_run.sh         — macOS/Linux one-click launcher
├── requirements.txt           — Python dependencies
├── docs/                      — User guide, README, changelog
├── tests/                     — pytest suite (48 tests)
└── tools/                     — Standalone utilities
    └── mxg_bin_validate.py    — .bin file readback / validation tool
```

## Validate a .bin File

```bash
python tools/mxg_bin_validate.py params/<stem>_mxg_4mb.bin --fs 125
```
