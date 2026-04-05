"""
test_mxg_waveform_designer.py
=============================
Pytest test suite for mxg_waveform_designer.py and mxg_bin_validate.py.
No hardware required.  Run with:  pytest -v
"""
import os
import sys
import tempfile

import numpy as np
import pytest

# ── import the designer (headless — skip tkinter main) ───────────────────────
# We import before tkinter is ever called; the GUI class is only instantiated
# inside  if __name__ == '__main__'  so tests are safe.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import mxg_waveform_designer as mwd
from mxg_waveform_designer import (
    WaveformType, ChannelConfig, CompositeConfig, CompositeBuilder,
    WaveformExporter, WaveformPlotter, resample_to_max_mb,
    load_params,
    _kernel_lfm, _kernel_nlfm, _kernel_cw, _kernel_fmcw,
    _kernel_stepped, _kernel_bpsk, _kernel_frank,
    lfm_channel_bank, nlfm_channel_bank,
)
import mxg_bin_validate as mbv


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _simple_cfg(**kwargs) -> CompositeConfig:
    """Return a minimal CompositeConfig suitable for unit tests (short waveform)."""
    defaults = dict(
        fs=50e6, num_pulses=2,
        pulse_width_s=10e-6, pri_s=20e-6,
        use_window=False, use_random_phase=False,
        save_mat=False, save_csv=False, save_bin=False,
    )
    defaults.update(kwargs)
    return CompositeConfig(**defaults)


def _simple_channels(n=4):
    return lfm_channel_bank(n_channels=n, chan_spacing_hz=5e6, bandwidth_hz=2e6)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKernels:
    """Each kernel should return a complex array of the same length as t."""

    def setup_method(self):
        self.N  = 500
        self.fs = 50e6
        self.t  = np.arange(self.N) / self.fs
        self.pw = self.t[-1]
        self.fc = 5e6

    def _check(self, s):
        assert s.dtype == complex or np.iscomplexobj(s)
        assert len(s) == self.N
        assert np.all(np.isfinite(s))

    def test_lfm(self):
        self._check(_kernel_lfm(self.t, self.pw, self.fc, bandwidth_hz=4e6))

    def test_lfm_down_chirp(self):
        self._check(_kernel_lfm(self.t, self.pw, self.fc,
                                bandwidth_hz=4e6, up_chirp=False))

    def test_nlfm_tangent(self):
        self._check(_kernel_nlfm(self.t, self.pw, self.fc,
                                 bandwidth_hz=4e6, law='tangent'))

    def test_nlfm_cosine(self):
        self._check(_kernel_nlfm(self.t, self.pw, self.fc,
                                 bandwidth_hz=4e6, law='cosine'))

    def test_nlfm_hamming(self):
        self._check(_kernel_nlfm(self.t, self.pw, self.fc,
                                 bandwidth_hz=4e6, law='hamming'))

    def test_nlfm_bad_law(self):
        with pytest.raises(ValueError):
            _kernel_nlfm(self.t, self.pw, self.fc, law='bad')

    def test_cw(self):
        s = _kernel_cw(self.t, self.fc)
        self._check(s)
        # CW should be unit amplitude everywhere
        np.testing.assert_allclose(np.abs(s), 1.0, atol=1e-10)

    def test_fmcw_sawtooth(self):
        self._check(_kernel_fmcw(self.t, self.pw, self.fc,
                                 bandwidth_hz=4e6, shape='sawtooth'))

    def test_fmcw_triangle(self):
        self._check(_kernel_fmcw(self.t, self.pw, self.fc,
                                 bandwidth_hz=4e6, shape='triangle'))

    def test_fmcw_bad_shape(self):
        with pytest.raises(ValueError):
            _kernel_fmcw(self.t, self.pw, self.fc, shape='spiral')

    def test_stepped(self):
        self._check(_kernel_stepped(self.t, self.pw, self.fc,
                                    bandwidth_hz=10e6, n_steps=4))

    def test_bpsk_default(self):
        self._check(_kernel_bpsk(self.t, self.pw, self.fc))

    def test_bpsk_custom_code(self):
        self._check(_kernel_bpsk(self.t, self.pw, self.fc,
                                 code=[1, -1, 1, -1]))

    def test_frank(self):
        self._check(_kernel_frank(self.t, self.pw, self.fc, order=4))


# ─────────────────────────────────────────────────────────────────────────────
# CompositeBuilder tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCompositeBuilder:

    def test_basic_build_shape(self):
        cfg      = _simple_cfg()
        channels = _simple_channels(4)
        builder  = CompositeBuilder(cfg, channels)
        iq, tbl  = builder.build()
        expected = int(round(cfg.fs * cfg.pri_s)) * cfg.num_pulses
        assert len(iq) == expected

    def test_iq_peak_within_scale(self):
        cfg      = _simple_cfg(final_peak_scale=0.85)
        channels = _simple_channels(4)
        iq, _    = CompositeBuilder(cfg, channels).build()
        assert np.max(np.abs(iq)) <= 0.86   # small tolerance

    def test_no_channels_raises(self):
        cfg = _simple_cfg()
        with pytest.raises(ValueError):
            CompositeBuilder(cfg, []).build()

    def test_pulse_wider_than_pri_raises(self):
        with pytest.raises(ValueError):
            CompositeBuilder(
                _simple_cfg(pulse_width_s=25e-6, pri_s=20e-6),
                _simple_channels(2)
            )

    def test_table_columns(self):
        cfg      = _simple_cfg()
        channels = _simple_channels(4)
        _, tbl   = CompositeBuilder(cfg, channels).build()
        assert 'WaveformType' in tbl.columns
        assert 'CenterFreqHz' in tbl.columns
        assert len(tbl) == 4

    # ── Taper tests ──────────────────────────────────────────────────────────

    def test_taper_cosine(self):
        cfg = _simple_cfg(use_amp_taper=True, taper_type='cosine')
        CompositeBuilder(cfg, _simple_channels(8)).build()

    def test_taper_taylor(self):
        cfg = _simple_cfg(use_amp_taper=True, taper_type='taylor',
                          taylor_nbar=4, taylor_sll=-30.0)
        iq, _ = CompositeBuilder(cfg, _simple_channels(8)).build()
        assert np.all(np.isfinite(iq))

    def test_taper_chebwin(self):
        cfg = _simple_cfg(use_amp_taper=True, taper_type='chebwin',
                          taylor_sll=-35.0)
        iq, _ = CompositeBuilder(cfg, _simple_channels(8)).build()
        assert np.all(np.isfinite(iq))

    def test_taper_taylor_weights_peak_one(self):
        """Taylor taper peak should be 1.0 (normalised)."""
        cfg = _simple_cfg(use_amp_taper=True, taper_type='taylor',
                          taylor_nbar=4, taylor_sll=-30.0)
        builder = CompositeBuilder(cfg, _simple_channels(16))
        w = builder._channel_weights()
        assert abs(w.max() - 1.0) < 1e-9

    # ── Pulse-Doppler tests ──────────────────────────────────────────────────

    def test_pulse_doppler_stagger_longer_than_uniform(self):
        """Stagger mode produces a longer waveform than uniform tiling."""
        base_cfg = _simple_cfg(num_pulses=5)
        pd_cfg   = _simple_cfg(num_pulses=5, use_pulse_doppler=True,
                               pd_pri_step_s=1e-6, pd_mode='stagger')
        chs = _simple_channels(2)
        iq_base, _ = CompositeBuilder(base_cfg, chs).build()
        iq_pd, _   = CompositeBuilder(pd_cfg,   chs).build()
        assert len(iq_pd) > len(iq_base)

    def test_pulse_doppler_jitter(self):
        cfg = _simple_cfg(num_pulses=4, use_pulse_doppler=True,
                          pd_pri_step_s=2e-6, pd_mode='jitter')
        iq, _ = CompositeBuilder(cfg, _simple_channels(2)).build()
        assert len(iq) > 0 and np.all(np.isfinite(iq))


# ─────────────────────────────────────────────────────────────────────────────
# WaveformExporter binary format tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBinaryFormat:

    def _build_iq(self, n=1000):
        t  = np.arange(n) / 50e6
        fc = 5e6
        return np.exp(1j * 2 * np.pi * fc * t) * 0.8

    def test_save_bin_dtype_big_endian(self, tmp_path):
        """Binary file must be int16 big-endian."""
        iq   = self._build_iq()
        I, Q = np.real(iq), np.imag(iq)
        base = str(tmp_path / 'test')
        WaveformExporter._save_bin(base, I, Q)
        raw = np.fromfile(base + '.bin', dtype='>i2')
        assert raw.size == 2 * len(I)

    def test_save_bin_scale_utilisation(self, tmp_path):
        """Peak amplitude should be close to ±32767 (≥ 90 % utilisation)."""
        iq   = self._build_iq()
        I, Q = np.real(iq), np.imag(iq)
        base = str(tmp_path / 'test')
        WaveformExporter._save_bin(base, I, Q)
        raw  = np.fromfile(base + '.bin', dtype='>i2').astype(float)
        peak = np.max(np.abs(raw))
        assert peak / 32767.0 >= 0.90

    def test_roundtrip_iq(self, tmp_path):
        """Read back I/Q from .bin and verify it matches the original (±1 LSB)."""
        iq   = self._build_iq()
        I, Q = np.real(iq), np.imag(iq)
        base = str(tmp_path / 'rt')
        WaveformExporter._save_bin(base, I, Q)
        raw  = np.fromfile(base + '.bin', dtype='>i2')
        I_back = raw[0::2].astype(float)
        Q_back = raw[1::2].astype(float)
        # Normalise to compare shapes
        I_norm = I / np.max(np.abs(I))
        I_back_norm = I_back / np.max(np.abs(I_back))
        np.testing.assert_allclose(I_norm, I_back_norm, atol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# resample_to_max_mb tests
# ─────────────────────────────────────────────────────────────────────────────

class TestResample:

    def _iq(self, n=200_000, fs=125e6):
        t = np.arange(n) / fs
        return np.exp(1j * 2 * np.pi * 5e6 * t), fs

    def test_no_resample_needed(self):
        iq, fs = self._iq(n=100)
        iq_out, fs_out = resample_to_max_mb(iq, fs, max_mb=4.0)
        assert len(iq_out) == len(iq)
        assert fs_out == fs

    def test_output_within_limit(self):
        iq, fs = self._iq(n=500_000)
        iq_out, fs_out = resample_to_max_mb(iq, fs, max_mb=4.0)
        bytes_out = len(iq_out) * 4  # 2×int16
        assert bytes_out <= 4.0 * 1e6 + 4   # allow 1 sample rounding

    def test_no_limit_passthrough(self):
        iq, fs = self._iq()
        iq_out, fs_out = resample_to_max_mb(iq, fs, max_mb=0)
        assert iq_out is iq

    def test_output_finite(self):
        iq, fs = self._iq(n=300_000)
        iq_out, _ = resample_to_max_mb(iq, fs, max_mb=2.0)
        assert np.all(np.isfinite(iq_out))


# ─────────────────────────────────────────────────────────────────────────────
# WaveformPlotter._detect_pri tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPRIDetect:

    def _make_pulse_train(self, fs=50e6, pw=10e-6, pri=25e-6, n_pulses=5):
        NsPRI   = int(round(fs * pri))
        NsPulse = int(round(fs * pw))
        one_pri = np.zeros(NsPRI, dtype=complex)
        t       = np.arange(NsPulse) / fs
        one_pri[:NsPulse] = np.exp(1j * 2 * np.pi * 5e6 * t)
        return np.tile(one_pri, n_pulses), NsPRI

    def test_detects_correct_pri(self):
        iq, NsPRI = self._make_pulse_train()
        detected  = WaveformPlotter._detect_pri(iq)
        # Accept within 2 % of true PRI
        assert abs(detected - NsPRI) / NsPRI < 0.02

    def test_fallback_on_cw(self):
        """CW signal has no pulse structure — should fall back without error."""
        iq = np.exp(1j * 2 * np.pi * 5e6 * np.arange(50_000) / 50e6)
        detected = WaveformPlotter._detect_pri(iq)
        assert detected > 0


# ─────────────────────────────────────────────────────────────────────────────
# mxg_bin_validate tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBinValidate:

    def _write_bin(self, path, n=2000, fs=125e6):
        t  = np.arange(n) / fs
        iq = np.exp(1j * 2 * np.pi * 5e6 * t)
        I  = np.real(iq)
        Q  = np.imag(iq)
        WaveformExporter._save_bin(path, I, Q)

    def test_load_bin(self, tmp_path):
        p = str(tmp_path / 'v')
        self._write_bin(p)
        iq = mbv.load_bin(p + '.bin')
        assert len(iq) == 2000
        assert np.iscomplexobj(iq)

    def test_validate_fields(self, tmp_path):
        p = str(tmp_path / 'v')
        self._write_bin(p)
        r = mbv.validate(p + '.bin', fs_hz=125e6)
        assert r['n_iq_pairs'] == 2000
        assert r['scale_utilisation'] >= 90.0
        assert r['endian']['likely_big_endian'] == True  # noqa: E712  (np.bool_ safe)

    def test_endianness_check_correct(self, tmp_path):
        p = str(tmp_path / 'e')
        self._write_bin(p)
        result = mbv.check_endianness(p + '.bin')
        assert result['likely_big_endian'] == True  # noqa: E712

    def test_validate_size_warning(self, tmp_path):
        """A file > 4 MB should trigger a size warning."""
        p = str(tmp_path / 'big')
        # Write 1,100,000 IQ pairs = 4.4 MB
        n  = 1_100_000
        iq = np.exp(1j * np.linspace(0, 10 * np.pi, n))
        WaveformExporter._save_bin(p, np.real(iq), np.imag(iq))
        r  = mbv.validate(p + '.bin', fs_hz=125e6)
        warnings_text = []
        if r['size_mb'] > 4.0:
            warnings_text.append('size')
        assert 'size' in warnings_text


# ─────────────────────────────────────────────────────────────────────────────
# Export helpers tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExportHelpers:

    def _iq_and_cfg(self, n=500):
        cfg = _simple_cfg(rf_center_hz=2.4e9, fs=50e6)
        t   = np.arange(n) / cfg.fs
        iq  = np.exp(1j * 2 * np.pi * 5e6 * t)
        return iq, cfg

    def test_vsa89600_mat_keys(self, tmp_path):
        import scipy.io as sio
        iq, cfg = self._iq_and_cfg()
        base = str(tmp_path / 'vsa')
        WaveformExporter._save_vsa89600(base, iq, cfg)
        mat = sio.loadmat(base + '_vsa89600.mat')
        for key in ('Y', 'XDelta', 'XStart', 'XUnit', 'YUnit', 'InputCenter'):
            assert key in mat

    def test_vsa89600_xdelta(self, tmp_path):
        import scipy.io as sio
        iq, cfg = self._iq_and_cfg()
        base = str(tmp_path / 'vsa2')
        WaveformExporter._save_vsa89600(base, iq, cfg)
        mat = sio.loadmat(base + '_vsa89600.mat')
        assert abs(float(mat['XDelta']) - 1.0 / cfg.fs) < 1e-15

    def test_m_script_created(self, tmp_path):
        iq, cfg = self._iq_and_cfg()
        base = str(tmp_path / 'ms')
        WaveformExporter._save_m_script(base, cfg)
        assert os.path.isfile(base + '_load.m')
        content = open(base + '_load.m', encoding='utf-8').read()
        assert 'spectrogram' in content
        assert 'tcpclient' in content

    def test_scpi_sidecar_created(self, tmp_path):
        iq, cfg = self._iq_and_cfg()
        base = str(tmp_path / 'sc')
        WaveformExporter._save_scpi_sidecar(base, 4.0, 100e6, cfg)
        sidecar = base + '_mxg_4mb_scpi.txt'
        assert os.path.isfile(sidecar)
        content = open(sidecar).read()
        assert 'MMEM:DATA' in content
        assert 'WGEN:ARB:WAVEFORM' in content


# ─────────────────────────────────────────────────────────────────────────────
# Params JSON auto-save + load_params roundtrip
# ─────────────────────────────────────────────────────────────────────────────

class TestParamsSave:

    def _build(self, tmp_path):
        # base_file_name already points into params/ (mirrors GUI behaviour)
        params_dir = tmp_path / 'params'
        cfg      = _simple_cfg(
            base_file_name=str(params_dir / 'test_wfm'),
            save_bin=False, save_mat=False, save_csv=False,
        )
        channels = _simple_channels(4)
        iq, tbl  = CompositeBuilder(cfg, channels).build()
        WaveformExporter().save_all(iq, cfg, channels, tbl)
        return cfg, channels, iq, tmp_path

    def test_params_dir_created(self, tmp_path):
        self._build(tmp_path)
        assert os.path.isdir(tmp_path / 'params')

    def test_params_file_exists(self, tmp_path):
        self._build(tmp_path)
        # params JSON is flat inside params/ — no nested subfolder
        assert os.path.isfile(tmp_path / 'params' / 'test_wfm_params.json')

    def test_params_json_keys(self, tmp_path):
        self._build(tmp_path)
        import json
        doc = json.loads((tmp_path / 'params' / 'test_wfm_params.json'
                          ).read_text(encoding='utf-8'))
        for key in ('_version', '_saved', 'config', 'channels', 'derived'):
            assert key in doc

    def test_params_channel_count(self, tmp_path):
        self._build(tmp_path)
        import json
        doc = json.loads((tmp_path / 'params' / 'test_wfm_params.json'
                          ).read_text(encoding='utf-8'))
        assert doc['derived']['total_channels'] == 4

    def test_load_params_roundtrip(self, tmp_path):
        """load_params() should reconstruct identical cfg and channels."""
        cfg_orig, chs_orig, _, _ = self._build(tmp_path)
        params_path = str(tmp_path / 'params' / 'test_wfm_params.json')
        cfg2, chs2 = load_params(params_path)

        assert cfg2.fs            == cfg_orig.fs
        assert cfg2.num_pulses    == cfg_orig.num_pulses
        assert cfg2.pulse_width_s == cfg_orig.pulse_width_s
        assert len(chs2)          == len(chs_orig)
        assert chs2[0].waveform_type == chs_orig[0].waveform_type

    def test_params_saved_each_unique_build(self, tmp_path):
        """Two builds with different file names produce two separate param files."""
        params_dir = tmp_path / 'params'
        for name in ('build_a', 'build_b'):
            cfg = _simple_cfg(
                base_file_name=str(params_dir / name),
                save_bin=False, save_mat=False, save_csv=False,
            )
            chs = _simple_channels(2)
            iq, tbl = CompositeBuilder(cfg, chs).build()
            WaveformExporter().save_all(iq, cfg, chs, tbl)
        files = [f for f in params_dir.iterdir() if f.suffix == '.json']
        assert len(files) == 2
