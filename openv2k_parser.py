#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenV2K IQ Parser -- offline analyser / validator for OpenV2K recordings
=======================================================================

A self-contained Tkinter + NumPy + Matplotlib desktop application that parses
the raw IQ recordings written by the OpenV2K application
(https://github.com/OpenV2K, "Record IQ" button) and reproduces the project's
own analysis chain offline, plus pulse-level measurements and export.

What OpenV2K writes
-------------------
* ``~/OpenV2K_<YYYYmmdd_HHMMSS>.iq`` -- raw IQ, dtype ``complex64``
  (interleaved float32 I/Q), sample rate = HackRF rate = 2 000 000 Sps.
  Compatible with GNU Radio / inspectrum / GQRX / SDR#.
* Optionally a waterfall/spectrogram PNG generated *inside* the app
  (``_generate_waterfall``), not needed here because this tool rebuilds it.

Analysis reproduced here (same constants as OpenV2K160.py)
----------------------------------------------------------
* active-region threshold = 5 % of peak |IQ|            (``thresh_factor``)
* +/- 1 ms of pre/post padding around the active region  (``pre/post_ms``)
* active window capped at 3.0 s before FFT               (``max_active_s``)
* spectrogram: fft_size = 64, Hann window, hop = 32 samples => 16.0 us/frame,
  fftshift, power in dB, extent = [ms, +/-rate/2 MHz]
* duration "checksum": detected active duration vs the source audio's own
  active duration, tolerance ``abs(d) <= max(30 ms, 0.3 * src_ms)``
  (OpenV2K skips this for live-microphone input, since there is no
  fixed-length source file -- load a WAV here to get the same check).

Extra measurements added by this tool
-------------------------------------
* per-pulse table: start time, width (us), peak amplitude -> CSV
* pulse count, mean/median/min/max width, PRF (Hz), duty cycle (%)
* peak / RMS level (dBFS), PAPR, DC offset, clipping count, median level
* averaged spectrum (Welch-style, NumPy only) and I/Q constellation
* OpenV2K event-log parser (pulls Pulse width / HPF / LPF / Filter / Checksum
  lines out of a pasted or saved log)
* export: TXT report, JSON metrics, pulse CSV, waterfall PNG, per-tab PNG

Usage
-----
    python3 openv2k_parser.py                      # empty GUI, use File > Open
    python3 openv2k_parser.py ~/OpenV2K_20260912_143005.iq
    python3 openv2k_parser.py rec.iq --report out.txt --pulses out.csv
    python3 openv2k_parser.py --generate-sample sample.iq --generate-wav sample.wav
    Xvfb :99 & DISPLAY=:99 python3 openv2k_parser.py rec.iq --selftest OUTDIR

Requires: Python 3.8+, numpy, matplotlib, tkinter (python3-tk).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import sys
import threading
import time
import traceback
import wave
from collections import namedtuple
from dataclasses import dataclass, field, asdict

import numpy as np

# ----------------------------------------------------------------------------
# GUI / plotting toolkit -- fail with an actionable message, not a traceback
# ----------------------------------------------------------------------------
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as exc:  # pragma: no cover
    sys.stderr.write(
        "ERROR: tkinter is not available ({0}).\n"
        "  Debian/Ubuntu : sudo apt install python3-tk\n"
        "  Fedora        : sudo dnf install python3-tkinter\n"
        "  macOS/Windows : reinstall Python with Tcl/Tk support\n".format(exc))
    raise SystemExit(2)

try:
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk)
except Exception as exc:  # pragma: no cover
    sys.stderr.write(
        "ERROR: matplotlib is not available ({0}).\n"
        "  pip install matplotlib\n".format(exc))
    raise SystemExit(2)

VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Constants taken from the OpenV2K application
# ---------------------------------------------------------------------------
OPENV2K_RATE = 2_000_000       # HACKRF_RATE in OpenV2K160.py
SIDEBAR_W = 380               # fixed width of the control sidebar (px)
DEFAULT_FFT = 64               # fft_size used by _generate_waterfall
DEFAULT_HOP = 32               # hop  samples -> 16.0 us per waterfall frame
DEFAULT_THRESH_FACTOR = 0.05   # mag.max() * 0.05
DEFAULT_PRE_MS = 1.0           # sr * 0.001
DEFAULT_POST_MS = 1.0
DEFAULT_MAX_ACTIVE_S = 3.0     # MAX_S in _generate_waterfall
EPS = 1e-10

SAMPLE_RE = re.compile(r"^(OpenV2K|openv2k)[_\-].*\.(iq|cfile|bin|dat|raw)$",
                       re.IGNORECASE)


# ---------------------------------------------------------------------------
# Sample formats
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FormatSpec:
    key: str
    label: str
    dtype: str          # numpy dtype of the on-disk words
    words_per_sample: int
    interleaved: bool
    scale: float
    complex_out: bool


FORMATS = {
    "complex64": FormatSpec(
        "complex64", "complex64 IQ - OpenV2K / GNU Radio (interleaved f32)",
        "complex64", 1, False, 1.0, True),
    "int16_iq": FormatSpec(
        "int16_iq", "int16 interleaved IQ - HackRF raw / .cs16",
        "int16", 2, True, 1.0 / 32768.0, True),
    "int8_iq": FormatSpec(
        "int8_iq", "int8 interleaved IQ - HackRF raw 8-bit",
        "int8", 2, True, 1.0 / 128.0, True),
    "f32_real": FormatSpec(
        "f32_real", "float32 mono real - audio / baseband envelope",
        "float32", 1, False, 1.0, False),
    "i16_real": FormatSpec(
        "i16_real", "int16 mono real - audio capture",
        "int16", 1, False, 1.0 / 32768.0, False),
}
FORMAT_ORDER = ["complex64", "int16_iq", "int8_iq", "f32_real", "i16_real"]


class IQReader:
    """Memory-mapped, chunk-readable view over an OpenV2K / SDR IQ file."""

    def __init__(self, path: str, fmt_key: str, rate: float):
        self.path = os.path.abspath(os.path.expanduser(path))
        self.fmt = FORMATS[fmt_key]
        self.rate = float(rate)
        self.size_bytes = os.path.getsize(self.path)
        item = np.dtype(self.fmt.dtype).itemsize
        words = self.size_bytes // item
        self.samples = words // self.fmt.words_per_sample
        self.trailing_bytes = self.size_bytes - words * item
        if self.samples <= 0:
            raise ValueError(
                "file too small for the selected format ({} bytes)".format(
                    self.size_bytes))
        # shape= keeps the memmap aligned even if the file has trailing bytes
        self._raw = np.memmap(
            self.path, dtype=self.fmt.dtype, mode="r",
            shape=(words,))
        self.duration = self.samples / self.rate

    # -- reading ----------------------------------------------------------
    def read(self, start: int, count: int) -> np.ndarray:
        """Return `count` samples from `start` (clipped to file bounds)."""
        start = max(0, int(start))
        stop = min(self.samples, start + int(count))
        if stop <= start:
            return np.zeros(0, dtype=np.complex64)
        w0 = start * self.fmt.words_per_sample
        w1 = stop * self.fmt.words_per_sample
        chunk = np.asarray(self._raw[w0:w1])
        return self._convert(chunk)

    def _convert(self, chunk: np.ndarray) -> np.ndarray:
        f = self.fmt
        if f.interleaved:
            pairs = chunk.reshape(-1, 2)
            i = pairs[:, 0].astype(np.float32)
            q = pairs[:, 1].astype(np.float32)
            return ((i + 1j * q) * np.float32(f.scale)).astype(np.complex64)
        if f.complex_out:
            return chunk.astype(np.complex64)
        return chunk.astype(np.float32)

    def iter_chunks(self, chunk_samples: int = 1 << 20):
        pos = 0
        while pos < self.samples:
            n = min(chunk_samples, self.samples - pos)
            yield pos, self.read(pos, n)
            pos += n


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class PulseStats:
    count: int = 0
    raw_count: int = 0
    starts_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    widths_us: np.ndarray = field(default_factory=lambda: np.zeros(0))
    peaks: np.ndarray = field(default_factory=lambda: np.zeros(0))
    prf_hz: float = 0.0
    jitter_us: float = 0.0
    duty_pct: float = 0.0
    mean_width_us: float = 0.0
    median_width_us: float = 0.0
    min_width_us: float = 0.0
    max_width_us: float = 0.0


@dataclass
class CheckSum:
    src_ms: float = 0.0
    active_ms: float = 0.0
    ratio: float = 0.0
    ok: bool = False
    tol_ok_ms: float = 0.0
    note: str = ""
    source: str = ""


@dataclass
class AnalysisResult:
    path: str = ""
    fmt_key: str = "complex64"
    fmt_label: str = ""
    rate: float = 0.0
    samples: int = 0
    size_bytes: int = 0
    duration_s: float = 0.0
    trailing_bytes: int = 0
    real_input: bool = False
    # global metrics
    peak_lin: float = 0.0
    peak_dbfs: float = 0.0
    rms_dbfs: float = 0.0
    papr_db: float = 0.0
    median_dbfs: float = 0.0
    dc_i: float = 0.0
    dc_q: float = 0.0
    clip_count: int = 0
    # active region
    thresh_lin: float = 0.0
    thresh_factor: float = 0.0
    active_first: int = 0
    active_last: int = 0
    win_start: int = 0
    win_end: int = 0
    active_ms: float = 0.0
    truncated: bool = False
    # waveforms / plots
    env_full_t_ms: np.ndarray = field(default_factory=lambda: np.zeros(0))
    env_full_mag: np.ndarray = field(default_factory=lambda: np.zeros(0))
    env_t_ms: np.ndarray = field(default_factory=lambda: np.zeros(0))
    env_mag: np.ndarray = field(default_factory=lambda: np.zeros(0))
    env_t0_ms: float = 0.0
    env_t1_ms: float = 0.0
    spec_db: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    spec_extent: tuple = (0.0, 1.0, -1.0, 1.0)
    spec_step_us: float = 0.0
    spec_cols: int = 0
    psd_freq_mhz: np.ndarray = field(default_factory=lambda: np.zeros(0))
    psd_db: np.ndarray = field(default_factory=lambda: np.zeros(0))
    psd_peak_mhz: float = 0.0
    iq_i: np.ndarray = field(default_factory=lambda: np.zeros(0))
    iq_q: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # pulse analysis
    pulses: PulseStats = field(default_factory=PulseStats)
    checksum: CheckSum = field(default_factory=CheckSum)
    params: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    elapsed_s: float = 0.0
    waveform_source: str = ""

    def metrics_dict(self) -> dict:
        d = asdict(self)
        for k in ("env_full_t_ms", "env_full_mag", "env_t_ms", "env_mag",
                  "spec_db", "psd_freq_mhz", "psd_db", "iq_i", "iq_q"):
            d.pop(k, None)
        p = d["pulses"]
        p["starts_s"] = (self.pulses.starts_s[:50].round(6).tolist())
        p["widths_us"] = (self.pulses.widths_us[:50].round(3).tolist())
        p["peaks"] = (self.pulses.peaks[:50].round(6).tolist())
        p["starts_s_truncated_to"] = int(min(50, self.pulses.count))
        return d


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def bucket_max(a: np.ndarray, g0: int, step: int, n_out: int,
               out: np.ndarray) -> None:
    """Decimate `out[k] = max(|a|)` for global samples g0.. with width `step`.

    Keeps narrow pulses visible in the full-file envelope plot.
    """
    if a.size == 0 or step <= 0:
        return
    k0 = g0 // step
    k1 = (g0 + a.size - 1) // step
    if k1 >= n_out:
        k1 = n_out - 1
    if k1 < k0:
        return
    idx = np.arange(k0, k1 + 1)
    offs = (idx * step - g0).astype(np.int64)
    np.clip(offs, 0, a.size - 1, out=offs)
    vals = np.maximum.reduceat(a, offs)
    # a bucket that starts before this chunk only sees part of its samples;
    # keep the running maximum instead of overwriting it.
    old = out[k0:k1 + 1]
    np.maximum(old, vals, out=old)


def _db(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, EPS))


def _db_amp(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(x, EPS))


def spectrogram(x: np.ndarray, fft_size: int, hop: int, max_cols: int = 2048):
    """Power spectrogram in dB, matching OpenV2K's _generate_waterfall math.

    Uses exactly the OpenV2K recipe: Hann window, |FFT|^2, fftshift,
    10*log10(power + 1e-10), one frame every `hop` samples (16 us at 2 MSps).

    Two deliberate differences, so a GUI stays responsive on long captures:
      * frames are *never* widened - pulse time resolution is preserved;
      * consecutive frames are averaged (in linear power) down to at most
        `max_cols` display columns, and processed in blocks so peak memory is
        bounded rather than n_frames x fft_size.

    Returns (spec_db[n_cols, fft_size] float32, frames_per_column) or (None, 0).
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n = x.size
    if n < fft_size:
        return None, 0
    n_frames = (n - fft_size) // hop + 1
    if n_frames <= 0:
        return None, 0

    c = max(1, int(math.ceil(n_frames / float(max_cols))))
    n_cols = int(math.ceil(n_frames / float(c)))
    win = np.hanning(fft_size).astype(np.float32)
    acc = np.zeros((n_cols, fft_size), dtype=np.float64)
    cnt = np.zeros(n_cols, dtype=np.int32)

    block_cols = 512                     # columns accumulated per FFT block
    f = 0                                # current frame index
    col = 0                              # current output column index
    while f < n_frames and col < n_cols:
        take = min(block_cols * c, n_frames - f)
        s0 = f * hop
        s1 = s0 + (take - 1) * hop + fft_size
        seg = x[s0:s1]
        got = (seg.size - fft_size) // hop + 1
        if got < take:
            take = got
        if take <= 0:
            break
        frames = sliding_window_view(seg, fft_size)[::hop][:take]
        p = np.abs(np.fft.fft(frames * win, axis=1)) ** 2
        p = np.fft.fftshift(p, axes=1)
        groups = take // c
        if groups:
            acc[col:col + groups] = p[:groups * c].reshape(
                groups, c, fft_size).sum(axis=1)
            cnt[col:col + groups] = c
        col += groups
        rest = take - groups * c
        last_block = (f + take) >= n_frames
        if rest and last_block:
            # trailing partial group -> final (narrower) column
            acc[col] = p[groups * c:groups * c + rest].sum(axis=0)
            cnt[col] = rest
            col += 1
        # frames that did not fill a whole column are re-read next block,
        # unless they were the final block (handled above)
        f += groups * c if not last_block else take
    cnt[cnt == 0] = 1
    acc = acc[:max(1, col)]
    cnt = cnt[:max(1, col)]
    spec = (_db(acc / cnt[:, None])).astype(np.float32)
    return spec, c


def detect_pulses(mag: np.ndarray, rate: float, thresh: float,
                  min_width_us: float, merge_gap_us: float) -> PulseStats:
    """Threshold/edge detection on the magnitude envelope -> pulse table."""
    st = PulseStats()
    if mag.size < 3:
        return st
    mask = mag > thresh
    if not mask.any():
        return st
    d = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mag.size]
    n = min(starts.size, ends.size)
    starts, ends = starts[:n], ends[:n]
    keep = ends > starts
    starts, ends = starts[keep], ends[keep]
    st.raw_count = int(starts.size)
    if st.raw_count == 0:
        return st

    # merge lobes separated by less than merge_gap (Schmitt-trigger ripple,
    # resampler ringing) and drop slivers shorter than min_width
    merge_samples = max(1, int(round(merge_gap_us * 1e-6 * rate)))
    min_samples = max(1, int(round(min_width_us * 1e-6 * rate)))
    m_start = [int(starts[0])]
    m_end = [int(ends[0])]
    for s, e in zip(starts[1:], ends[1:]):
        if s - m_end[-1] <= merge_samples:
            m_end[-1] = int(e)
        else:
            m_start.append(int(s))
            m_end.append(int(e))
    s_arr = np.asarray(m_start, dtype=np.int64)
    e_arr = np.asarray(m_end, dtype=np.int64)
    w = e_arr - s_arr
    good = w >= min_samples
    s_arr, e_arr, w = s_arr[good], e_arr[good], w[good]

    st.count = int(s_arr.size)
    if st.count == 0:
        return st
    st.starts_s = s_arr / rate
    st.widths_us = w / rate * 1e6
    st.peaks = np.array([float(mag[s:e].max()) for s, e in zip(s_arr, e_arr)])
    st.mean_width_us = float(st.widths_us.mean())
    st.median_width_us = float(np.median(st.widths_us))
    st.min_width_us = float(st.widths_us.min())
    st.max_width_us = float(st.widths_us.max())
    if st.count >= 2:
        gaps = np.diff(st.starts_s)
        st.prf_hz = float(1.0 / np.median(gaps)) if np.median(gaps) > 0 else 0.0
        st.jitter_us = float(np.std(gaps) * 1e6)
    else:
        st.prf_hz = 0.0
    span = float(st.starts_s[-1] + st.widths_us[-1] * 1e-6 - st.starts_s[0])
    if span > 0:
        st.duty_pct = float(st.widths_us.sum() * 1e-6 / span * 100.0)
    else:
        st.duty_pct = 0.0
    return st


def psd_welch(x: np.ndarray, fft_size: int = 1024, overlap: float = 0.5):
    """Simple Welch PSD (NumPy only) -> (freqs_Hz, dB)."""
    if x.size < fft_size:
        fft_size = int(2 ** int(math.floor(math.log2(max(8, x.size)))))
        if fft_size < 8:
            return np.zeros(0), np.zeros(0)
    hop = max(1, int(fft_size * (1.0 - overlap)))
    win = np.hanning(fft_size).astype(np.float32)
    wpow = float((win ** 2).sum())
    n_frames = (x.size - fft_size) // hop + 1
    if n_frames <= 0:
        return np.zeros(0), np.zeros(0)
    acc = np.zeros(fft_size, dtype=np.float64)
    count = 0
    xs = x.astype(np.complex64) if np.iscomplexobj(x) else x.astype(np.complex64)
    for f in range(min(n_frames, 4000)):
        seg = xs[f * hop:f * hop + fft_size]
        acc += (np.abs(np.fft.fft(seg * win)) ** 2)
        count += 1
    count = max(1, count)
    p = acc / (count * wpow)
    p = np.fft.fftshift(p)
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0)) * 1.0
    p_db = _db(p) + 10.0 * math.log10(2.0 / fft_size)
    return freqs, p_db.astype(np.float32)


def wav_active_ms(path: str, thresh_factor: float = DEFAULT_THRESH_FACTOR) -> tuple:
    """Active (non-silence) duration of a WAV, using OpenV2K's 5% rule."""
    with wave.open(path, "rb") as w:
        nch = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        nframes = w.getnframes()
        raw = w.readframes(nframes)
    if width == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    elif width == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError("unsupported WAV sample width: {} bytes".format(width))
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    peak = float(np.abs(a).max()) if a.size else 0.0
    total_ms = nframes / float(rate) * 1000.0
    if peak <= 0:
        return 0.0, total_ms, rate
    idx = np.flatnonzero(np.abs(a) > peak * thresh_factor)
    if idx.size == 0:
        return 0.0, total_ms, rate
    active_ms = (idx[-1] - idx[0] + 1) / float(rate) * 1000.0
    return float(active_ms), float(total_ms), float(rate)


LOG_PATTERNS = [
    ("Pulse width", re.compile(r"Pulse width:\s*([0-9.]+)\s*(?:u|\u00b5|µ)s", re.I)),
    ("HPF", re.compile(r"HPF:\s*([0-9.]+)\s*Hz", re.I)),
    ("LPF", re.compile(r"LPF:\s*([0-9.]+)\s*Hz", re.I)),
    ("Filter", re.compile(r"Filter\s+(ON|off)\s*[-–]+\s*(.+)$", re.I)),
    ("Checksum", re.compile(r"Checksum:\s*(.+)$", re.I)),
    ("Recording", re.compile(r"(Recording (?:IQ to disk|stopped).*)$", re.I)),
    ("Waterfall", re.compile(r"Waterfall:\s*(.+)$", re.I)),
    ("Error", re.compile(r"ERROR[^:]*:\s*(.+)$", re.I)),
    ("Power", re.compile(r"Power Calculation[^:]*:?\s*(.*)$", re.I)),
]


def parse_log_text(text: str) -> dict:
    """Pull OpenV2K event-log parameter/checksum lines out of pasted text."""
    found = {name: [] for name, _ in LOG_PATTERNS}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for name, rx in LOG_PATTERNS:
            m = rx.search(line)
            if m:
                found[name].append((m.group(1).strip(), line))
    checksum = {"active_ms": None, "src_ms": None, "ratio": None, "verdict": ""}
    for _, line in found.get("Checksum", []):
        m = re.search(r"active=\s*([0-9.]+)\s*ms", line)
        if m:
            checksum["active_ms"] = float(m.group(1))
        m = re.search(r"active-region=\s*([0-9.]+)\s*ms", line)
        if m:
            checksum["src_ms"] = float(m.group(1))
        m = re.search(r"ratio=\s*([0-9.]+)x", line)
        if m:
            checksum["ratio"] = float(m.group(1))
        if "MISMATCH" in line.upper():
            checksum["verdict"] = "MISMATCH"
        elif "OK" in line.upper():
            checksum["verdict"] = "OK"
    return {"params": {k: v for k, v in found.items() if k != "Checksum"},
            "checksum": checksum}


# ---------------------------------------------------------------------------
# Main analysis entry point (GUI-free, unit-testable)
# ---------------------------------------------------------------------------
def analyze_iq(path: str, fmt_key: str = "complex64", rate: float = OPENV2K_RATE,
               fft_size: int = DEFAULT_FFT, hop: int = DEFAULT_HOP,
               thresh_factor: float = DEFAULT_THRESH_FACTOR,
               pre_ms: float = DEFAULT_PRE_MS, post_ms: float = DEFAULT_POST_MS,
               max_active_s: float = DEFAULT_MAX_ACTIVE_S,
               max_display_cols: int = 2048,
               min_width_us: float = 1.0, merge_gap_us: float = 5.0,
               src_wav: str = None, src_ms_manual: float = None,
               progress=None) -> AnalysisResult:
    """Parse + analyse one recording. Pure function (no Tk objects)."""
    t0 = time.time()
    res = AnalysisResult(path=os.path.abspath(os.path.expanduser(path)))
    res.fmt_key = fmt_key
    res.fmt_label = FORMATS[fmt_key].label
    res.rate = float(rate)
    res.params = dict(fmt=fmt_key, rate=rate, fft_size=fft_size, hop=hop,
                      thresh_factor=thresh_factor, pre_ms=pre_ms,
                      post_ms=post_ms, max_active_s=max_active_s,
                      min_width_us=min_width_us, merge_gap_us=merge_gap_us)

    def say(pct, msg):
        if progress:
            progress(pct, msg)

    say(2, "opening {}".format(os.path.basename(res.path)))
    reader = IQReader(res.path, fmt_key, rate)
    res.samples = reader.samples
    res.size_bytes = reader.size_bytes
    res.duration_s = reader.duration
    res.trailing_bytes = reader.trailing_bytes
    res.real_input = not FORMATS[fmt_key].complex_out
    if res.trailing_bytes:
        res.warnings.append(
            "{} trailing byte(s) ignored (not a whole sample)".format(
                res.trailing_bytes))

    # ---- pass 1: global statistics ------------------------------------
    say(10, "measuring {}".format(os.path.basename(res.path)))
    peak = 0.0
    sumsq = 0.0
    sum_i = 0.0
    sum_q = 0.0
    clip = 0
    n_done = 0
    dec_step = max(1, int(math.ceil(reader.samples / 4096.0)))
    n_env = int(math.ceil(reader.samples / dec_step))
    env_full = np.zeros(n_env, dtype=np.float32)
    for pos, x in reader.iter_chunks(1 << 21):
        a = np.abs(x)
        if a.size:
            peak = max(peak, float(a.max()))
            sumsq += float(np.dot(a, a))
            clip += int((a >= 0.999).sum())
        if np.iscomplexobj(x):
            sum_i += float(x.real.sum())
            sum_q += float(x.imag.sum())
        bucket_max(a, pos, dec_step, n_env, env_full)
        n_done += x.size
        if reader.samples:
            say(10 + 25.0 * n_done / reader.samples, "measuring samples")
    n = max(1, reader.samples)
    res.peak_lin = peak
    res.peak_dbfs = _db_amp(np.array([peak]))[0]
    rms_amp = math.sqrt(sumsq / n)
    res.rms_dbfs = _db_amp(np.array([rms_amp]))[0]
    res.papr_db = float(res.peak_dbfs - res.rms_dbfs)
    res.dc_i = sum_i / n
    res.dc_q = sum_q / n
    res.clip_count = clip
    nz = env_full[env_full > 0]
    res.median_dbfs = _db_amp(np.array([np.median(nz) if nz.size else 0.0]))[0]
    res.env_full_mag = env_full
    res.env_full_t_ms = (np.arange(n_env) * dec_step) / res.rate * 1000.0

    thresh = max(peak * thresh_factor, 1e-12)
    res.thresh_lin = thresh
    res.thresh_factor = thresh_factor

    # ---- pass 2: locate active region (OpenV2K rule) -------------------
    say(40, "locating active region")
    first = None
    last = None
    for pos, x in reader.iter_chunks(1 << 21):
        idx = np.flatnonzero(np.abs(x) > thresh)
        if idx.size:
            if first is None:
                first = pos + int(idx[0])
            last = pos + int(idx[-1])
    if first is None or last is None:
        res.warnings.append(
            "no sample exceeded {:.3g} (peak*{:.3g}) - check format/rate".format(
                thresh, thresh_factor))
        res.elapsed_s = time.time() - t0
        return res
    res.active_first, res.active_last = int(first), int(last)
    pre = int(res.rate * pre_ms / 1000.0)
    post = int(res.rate * post_ms / 1000.0)
    win_start = max(0, first - pre)
    win_end = min(reader.samples, last + 1 + post)
    cap = int(res.rate * max_active_s)
    if win_end - win_start > cap:
        res.truncated = True
        win_end = win_start + cap
        res.warnings.append(
            "active region > {:.1f}s - analysis capped at first {:.1f}s "
            "(same cap as OpenV2K's waterfall)".format(max_active_s,
                                                       max_active_s))
    res.win_start, res.win_end = int(win_start), int(win_end)
    res.active_ms = (win_end - win_start) / res.rate * 1000.0
    chunk = reader.read(win_start, win_end - win_start)
    mag = np.abs(chunk)
    if not np.iscomplexobj(chunk):
        res.waveform_source = "real-valued input (I/Q = signal/0)"
    else:
        res.waveform_source = "complex IQ"

    # ---- envelope (decimated, peak preserving) -------------------------
    step = max(1, int(math.ceil(mag.size / 4096.0)))
    if step > 1:
        pad = (-mag.size) % step
        m2 = np.concatenate([mag, np.zeros(pad, dtype=mag.dtype)])
        env = m2.reshape(-1, step).max(axis=1)
    else:
        env = mag
    res.env_mag = env.astype(np.float32)
    res.env_t_ms = ((np.arange(env.size) * step) + win_start) / res.rate * 1000.0
    res.env_t0_ms = win_start / res.rate * 1000.0
    res.env_t1_ms = win_end / res.rate * 1000.0

    # ---- spectrogram (OpenV2K math) ------------------------------------
    say(55, "building spectrogram")
    spec, cols = spectrogram(chunk, fft_size, hop, max_display_cols)
    if spec is None:
        res.warnings.append("active window too short for a {} point FFT".format(
            fft_size))
    else:
        res.spec_db = spec
        res.spec_cols = int(spec.shape[0])
        res.spec_step_us = hop / res.rate * 1e6
        t0_ms = win_start / res.rate * 1000.0
        step_ms = (hop * spec.shape[0]) / float(chunk.size) if chunk.size else 0
        t1_ms = win_end / res.rate * 1000.0
        res.spec_extent = (t0_ms, t1_ms, -res.rate / 2 / 1e6,
                           res.rate / 2 / 1e6)

    # ---- spectrum ------------------------------------------------------
    say(70, "averaging spectrum")
    freqs, psd = psd_welch(chunk, 1024, 0.5)
    if freqs.size:
        res.psd_freq_mhz = (freqs * res.rate / 1e6).astype(np.float32)
        res.psd_db = psd
        # ignore the DC bin when naming the peak
        k = int(np.argmax(psd[1:-1])) + 1 if psd.size > 2 else int(np.argmax(psd))
        res.psd_peak_mhz = float(res.psd_freq_mhz[k])

    # ---- constellation -------------------------------------------------
    k = min(20000, mag.size)
    if k > 1:
        sel = np.linspace(0, mag.size - 1, k).astype(np.int64)
        res.iq_i = chunk.real[sel].astype(np.float32)
        res.iq_q = (chunk.imag[sel].astype(np.float32) if np.iscomplexobj(chunk)
                    else np.zeros(k, dtype=np.float32))

    # ---- pulse analysis ------------------------------------------------
    say(82, "detecting pulses")
    res.pulses = detect_pulses(mag, res.rate, thresh, min_width_us, merge_gap_us)
    if res.pulses.raw_count and res.pulses.count != res.pulses.raw_count:
        res.warnings.append(
            "{} raw threshold crossings merged/filtered -> {} pulses "
            "(merge gap {:.1f}us, min width {:.1f}us)".format(
                res.pulses.raw_count, res.pulses.count, merge_gap_us,
                min_width_us))

    # ---- duration checksum (OpenV2K validation tooling) ----------------
    src_ms = None
    src_label = ""
    if src_ms_manual is not None and src_ms_manual > 0:
        src_ms, src_label = float(src_ms_manual), "manual entry"
    elif src_wav:
        try:
            a_ms, tot_ms, wrate = wav_active_ms(src_wav)
            src_ms = a_ms
            src_label = "{} ({} Hz, {:.0f} ms file, {:.0f} ms active)".format(
                os.path.basename(src_wav), wrate, tot_ms, a_ms)
        except Exception as exc:
            res.warnings.append("could not read WAV: {}".format(exc))
    if src_ms:
        tol = max(30.0, 0.3 * src_ms)
        d = abs(res.active_ms - src_ms)
        res.checksum = CheckSum(
            src_ms=src_ms, active_ms=res.active_ms,
            ratio=res.active_ms / src_ms if src_ms else 0.0,
            ok=d <= tol, tol_ok_ms=d, source=src_label)
        res.checksum.note = (
            "active={:.0f}ms vs source={:.0f}ms (ratio={:.2f}x, delta={:.0f}ms, "
            "tolerance +/-{:.0f}ms) -- {}".format(
                res.active_ms, src_ms, res.checksum.ratio, d, tol,
                "OK" if res.checksum.ok else
                "MISMATCH, check signal chain for ringing/gating"))

    say(95, "done")
    res.elapsed_s = time.time() - t0
    return res


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------
def build_report(res: AnalysisResult, tab_px: int = 0) -> str:
    L = []
    add = L.append
    add("=" * 78)
    add("OpenV2K IQ PARSER REPORT")
    add("=" * 78)
    add("File                : {}".format(res.path))
    add("Format              : {}".format(res.fmt_label))
    add("Sample rate         : {:.0f} Sps".format(res.rate))
    add("Samples             : {:,}".format(res.samples))
    add("File size           : {:,} bytes".format(res.size_bytes))
    add("Duration            : {:.6f} s".format(res.duration_s))
    add("")
    add("-" * 78)
    add("LEVELS / GLOBAL STATISTICS")
    add("-" * 78)
    add("Peak level          : {:.2f} dBFS   ({:.6f})".format(
        res.peak_dbfs, res.peak_lin))
    add("RMS level           : {:.2f} dBFS".format(res.rms_dbfs))
    add("PAPR (peak/RMS)     : {:.2f} dB".format(res.papr_db))
    add("Median bucket peak: {:.2f} dBFS (decimated envelope)".format(
        res.median_dbfs))
    add("DC offset           : I={:+.3e}  Q={:+.3e}".format(res.dc_i, res.dc_q))
    add("Clipped samples     : {:,} (|IQ| >= 0.999)".format(res.clip_count))
    add("")
    add("-" * 78)
    add("ACTIVE REGION  (threshold = peak * {:.3g})".format(res.thresh_factor))
    add("-" * 78)
    add("Threshold           : {:.3e}".format(res.thresh_lin))
    add("First active sample : {:,}  ({:.3f} ms)".format(
        res.active_first, res.active_first / res.rate * 1000.0))
    add("Last  active sample : {:,}  ({:.3f} ms)".format(
        res.active_last, res.active_last / res.rate * 1000.0))
    add("Analysis window     : {:,} .. {:,}".format(res.win_start, res.win_end))
    add("Active window       : {:.3f} ms".format(res.active_ms))
    add("Truncated at cap    : {}".format("yes" if res.truncated else "no"))
    add("")
    add("-" * 78)
    add("PULSE MEASUREMENTS")
    add("-" * 78)
    p = res.pulses
    add("Raw crossings       : {}".format(p.raw_count))
    add("Pulses detected     : {}".format(p.count))
    add("Mean pulse width    : {:.3f} us".format(p.mean_width_us))
    add("Median pulse width  : {:.3f} us".format(p.median_width_us))
    add("Min / Max width     : {:.3f} / {:.3f} us".format(
        p.min_width_us, p.max_width_us))
    add("PRF (median-based)  : {:.3f} Hz".format(p.prf_hz))
    add("Pulse-to-pulse jitter: {:.3f} us (1 sigma)".format(p.jitter_us))
    add("Duty cycle          : {:.3f} %".format(p.duty_pct))
    add("")
    add("-" * 78)
    add("SPECTROGRAM / SPECTRUM")
    add("-" * 78)
    add("FFT size            : {} (Hann, {:.1f} us/frame)".format(
        res.params.get("fft_size"), res.spec_step_us))
    add("Display columns     : {}".format(res.spec_cols))
    add("Extent (ms)         : {:.3f} .. {:.3f}".format(
        res.spec_extent[0], res.spec_extent[1]))
    add("Freq span           : +/- {:.3f} MHz".format(res.rate / 2 / 1e6))
    add("Spectral peak       : {:+.4f} MHz (offset from centre)".format(
        res.psd_peak_mhz))
    add("")
    add("-" * 78)
    add("DURATION CHECKSUM  (OpenV2K validation tooling)")
    add("-" * 78)
    if res.checksum.src_ms > 0:
        add("Source active ms    : {:.1f}   [{}]".format(
            res.checksum.src_ms, res.checksum.source))
        add("Detected active ms  : {:.1f}".format(res.checksum.active_ms))
        add("Ratio               : {:.2f}x".format(res.checksum.ratio))
        add("Delta / tolerance   : {:.1f} ms / +/-{:.1f} ms".format(
            res.checksum.tol_ok_ms, max(30.0, 0.3 * res.checksum.src_ms)))
        add("Verdict             : {}".format(
            "OK" if res.checksum.ok else "MISMATCH"))
        add("Detail              : {}".format(res.checksum.note))
    else:
        add("Not evaluated -- load the source WAV or type the source speech")
        add("duration (OpenV2K skips this check for live-microphone input).")
    add("")
    if res.warnings:
        add("-" * 78)
        add("NOTES / WARNINGS")
        add("-" * 78)
        for w in res.warnings:
            add(" * {}".format(w))
        add("")
    add("-" * 78)
    add("Analysis time       : {:.2f} s".format(res.elapsed_s))
    add("Parser version      : OpenV2K IQ Parser {}".format(VERSION))
    add("=" * 78)
    if tab_px:
        add("Figure width at export: {} px".format(tab_px))
    return "\n".join(L)


def pulses_csv(res: AnalysisResult) -> str:
    rows = ["index,start_s,start_ms,width_us,peak_amp"]
    for i, (s, w, pk) in enumerate(zip(res.pulses.starts_s, res.pulses.widths_us,
                                       res.pulses.peaks)):
        rows.append("{},{:.9f},{:.6f},{:.3f},{:.6f}".format(i, s, s * 1e3, w, pk))
    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# Sample fixture generator (used by --generate-sample and --selftest)
# ---------------------------------------------------------------------------
def generate_sample_iq(path: str, rate: float = OPENV2K_RATE, pulse_us: float = 100.0,
                       prf_hz: float = 500.0, active_s: float = 1.0,
                       lead_s: float = 0.05, amp: float = 0.6,
                       noise: float = 5e-4, seed: int = 7):
    """Write a synthetic OpenV2K-style complex64 IQ file: a zero-crossing
    derived boxcar pulse train at `prf_hz`, with 3 speech-like bursts."""
    rng = np.random.default_rng(seed)
    total = int(round((active_s + 2 * lead_s) * rate))
    x = (rng.normal(0, noise, total) + 1j * rng.normal(0, noise, total)).astype(
        np.complex64)
    period = 1.0 / prf_hz
    w_pulse = int(round(pulse_us * 1e-6 * rate))
    bursts = ((0.00, 0.30), (0.34, 0.62), (0.66, 1.00))
    t = 0.0
    starts = []
    while t < active_s:
        # speech-like gating: 3 bursts separated by two short pauses
        phase = t / active_s
        if any(a <= phase < b for a, b in bursts):
            i0 = int(round((lead_s + t) * rate))
            i1 = min(total, i0 + w_pulse)
            phase_cycle = rng.uniform(0, 2 * math.pi)
            j = np.arange(i1 - i0)
            carrier = np.exp(1j * 0.15 * j)  # slow intra-pulse phase ramp
            x[i0:i1] = (amp * carrier * np.exp(1j * phase_cycle)).astype(
                np.complex64)
            starts.append(i0)
        t += period
    x.tofile(path)
    span_ms = ((starts[-1] + w_pulse) - starts[0]) / rate * 1000.0 if starts \
        else 0.0
    return {"path": path, "samples": int(total), "rate": rate,
            "pulses": len(starts), "pulse_us": pulse_us, "prf_hz": prf_hz,
            "active_s": active_s, "lead_s": lead_s,
            "expected_active_span_ms": span_ms,
            "expected_duty_pct": (pulse_us * 1e-6 * len(starts) /
                                  (span_ms / 1000.0) * 100.0) if span_ms else 0.0,
            "expected_prf_hz": prf_hz}


def generate_sample_wav(path: str, active_ms: float, rate: int = 48000,
                        f0: float = 220.0):
    """Write a WAV whose active (above 5% of peak) duration is `active_ms`."""
    n = int(round(active_ms / 1000.0 * rate))
    t = np.arange(n) / rate
    a = 0.5 * np.sin(2 * math.pi * f0 * t)
    pcm = np.clip(a * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return {"path": path, "active_ms": active_ms, "rate": rate}


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class ParserApp(tk.Tk):
    TABS = ["Waterfall", "Envelope && Pulses", "Pulse Zoom", "Spectrum",
            "Constellation", "Report", "Event Log"]

    def __init__(self, initial_file: str = None, rate: float = OPENV2K_RATE,
                 fmt_key: str = "complex64", **kw):
        super().__init__(**kw)
        self.title("OpenV2K IQ Parser {}  -  offline waveform inspector".format(
            VERSION))
        self.geometry("1500x950")
        self.minsize(1050, 700)
        self.configure(bg="#20242b")

        # state
        self.res: AnalysisResult = None
        self.q = queue.Queue()
        self.worker = None
        self.busy = False
        self.recent = []
        self._toolbars = {}
        self._canvases = {}
        self._figs = {}

        self.var_fmt = tk.StringVar(value=FORMATS[fmt_key].label)
        self.var_rate = tk.StringVar(value="{:g}".format(rate))
        self.var_fft = tk.StringVar(value=str(DEFAULT_FFT))
        self.var_hop = tk.StringVar(value=str(DEFAULT_HOP))
        self.var_thresh = tk.DoubleVar(value=DEFAULT_THRESH_FACTOR * 100.0)
        self.var_pre = tk.StringVar(value="{:g}".format(DEFAULT_PRE_MS))
        self.var_post = tk.StringVar(value="{:g}".format(DEFAULT_POST_MS))
        self.var_maxact = tk.StringVar(value="{:g}".format(DEFAULT_MAX_ACTIVE_S))
        self.var_minw = tk.StringVar(value="1.0")
        self.var_gap = tk.StringVar(value="5.0")
        self.var_cols = tk.StringVar(value="2048")
        self.var_srcms = tk.StringVar(value="")
        self.var_wav = tk.StringVar(value="(none)")
        self.var_full = tk.BooleanVar(value=False)
        self.var_zoom_start = tk.StringVar(value="0")
        self.var_zoom_span = tk.StringVar(value="25")
        self.status = tk.StringVar(value="Ready.  File > Open IQ recording "
                                         "(Ctrl+O), or drop a path on the command line.")

        self._build_menu()
        self._build_layout()
        self._build_sidebar()
        self._build_tabs()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(80, self._poll)

        if initial_file:
            self.after(200, lambda: self.load_file(initial_file))

    # -- chrome ---------------------------------------------------------
    def _build_menu(self):
        m = tk.Menu(self)
        f = tk.Menu(m, tearoff=0)
        f.add_command(label="Open IQ recording\u2026", accelerator="Ctrl+O",
                      command=self.on_open)
        f.add_command(label="Load source WAV (for checksum)\u2026",
                      command=self.on_open_wav)
        f.add_command(label="Load OpenV2K event log\u2026", command=self.on_open_log)
        f.add_separator()
        f.add_command(label="Scan home for OpenV2K_*.iq", command=self.on_scan_home)
        self.menu_recent = tk.Menu(f, tearoff=0)
        f.add_cascade(label="Recent", menu=self.menu_recent)
        f.add_separator()
        f.add_command(label="Export report (TXT)\u2026", command=self.export_txt)
        f.add_command(label="Export metrics (JSON)\u2026", command=self.export_json)
        f.add_command(label="Export pulse table (CSV)\u2026", command=self.export_csv)
        f.add_command(label="Export waterfall PNG\u2026",
                      command=lambda: self.export_png("Waterfall"))
        f.add_command(label="Export current tab PNG\u2026",
                      command=lambda: self.export_png("__current__"))
        f.add_separator()
        f.add_command(label="Quit", accelerator="Ctrl+Q", command=self._on_close)
        m.add_cascade(label="File", menu=f)

        v = tk.Menu(m, tearoff=0)
        v.add_checkbutton(label="Fit whole file (off = active region only)",
                          variable=self.var_full, command=self.render_all)
        v.add_command(label="Redraw active tab", command=self.render_all)
        m.add_cascade(label="View", menu=v)

        h = tk.Menu(m, tearoff=0)
        h.add_command(label="Show README / about this parser", command=self.show_about)
        m.add_cascade(label="Help", menu=h)
        self.config(menu=m)

    def _build_layout(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook.Tab", padding=(10, 5))
        style.configure("Mono.TLabel", font=("DejaVu Sans Mono", 9))
        style.configure("Head.TLabel", font=("DejaVu Sans", 10, "bold"))

        bar = ttk.Frame(self, padding=(6, 5))
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(bar, text="Open IQ\u2026", command=self.on_open).pack(side=tk.LEFT)
        ttk.Button(bar, text="Source WAV\u2026", command=self.on_open_wav).pack(
            side=tk.LEFT, padx=(4, 0))
        ttk.Button(bar, text="Event Log\u2026", command=self.on_open_log).pack(
            side=tk.LEFT, padx=(4, 0))
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y,
                                                    padx=8)
        ttk.Button(bar, text="Analyse / Reload (F5)",
                   command=self.on_reanalyze).pack(side=tk.LEFT)
        ttk.Checkbutton(bar, text="Fit whole file", variable=self.var_full,
                        command=self.render_all).pack(side=tk.LEFT, padx=10)

        self.pb = ttk.Progressbar(self, mode="determinate", maximum=100)
        self.pb.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(self, textvariable=self.status, anchor="w",
                  padding=(8, 3)).pack(side=tk.BOTTOM, fill=tk.X)

        panes = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True)
        self.panes = panes
        side_holder = ttk.Frame(panes, width=SIDEBAR_W)
        self.right = ttk.Frame(panes, padding=(0, 4))
        panes.add(side_holder, weight=0)
        panes.add(self.right, weight=1)
        self.side_holder = side_holder

        # The sidebar is fixed width and scrolls vertically when the window is
        # short, so the notebook always keeps the remaining space.
        try:
            bg = style.lookup("TFrame", "background") or "#dcdad5"
        except tk.TclError:
            bg = "#dcdad5"
        self.side_canvas = tk.Canvas(side_holder, width=SIDEBAR_W - 16,
                                     highlightthickness=0, bd=0, bg=bg)
        self.side_sb = ttk.Scrollbar(side_holder, orient="vertical",
                                     command=self.side_canvas.yview)
        self.side_canvas.configure(yscrollcommand=self.side_sb.set)
        self.side_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.side_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.side = ttk.Frame(self.side_canvas, padding=6)
        self._side_win = self.side_canvas.create_window(
            (0, 0), window=self.side, anchor="nw")

        def _side_sync(_e=None):
            self.side_canvas.configure(scrollregion=self.side_canvas.bbox("all"))
            self.side_canvas.itemconfigure(self._side_win,
                                           width=self.side_canvas.winfo_width())

        self.side.bind("<Configure>", _side_sync)
        self.side_canvas.bind("<Configure>", _side_sync)

        def _side_wheel(e):
            if getattr(e, "num", None) == 4:
                step = -1
            elif getattr(e, "num", None) == 5:
                step = 1
            else:
                step = -1 if getattr(e, "delta", 0) > 0 else 1
            self.side_canvas.yview_scroll(step, "units")

        for w in (self.side_canvas, self.side):
            for ev in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
                w.bind(ev, _side_wheel)

        self.after(120, self._claim_sidebar_width)

    def _claim_sidebar_width(self):
        """Keep the sidebar at a fixed width regardless of window size."""
        try:
            self.panes.sashpos(0, SIDEBAR_W)
        except (tk.TclError, AttributeError):
            pass

    # -- sidebar --------------------------------------------------------
    def _build_sidebar(self):
        s = self.side

        def head(txt):
            ttk.Label(s, text=txt, style="Head.TLabel").pack(
                anchor="w", pady=(10, 2))

        def row(label, widget, width=12):
            fr = ttk.Frame(s)
            fr.pack(fill=tk.X, pady=1)
            ttk.Label(fr, text=label, width=19, anchor="w").pack(side=tk.LEFT)
            widget.pack(in_=fr, side=tk.LEFT, fill=tk.X, expand=True)

        head("RECORDING")
        self.cb_fmt = ttk.Combobox(
            s, state="readonly", values=[FORMATS[k].label for k in FORMAT_ORDER])
        _init = next((i for i, k in enumerate(FORMAT_ORDER)
                      if FORMATS[k].label == self.var_fmt.get()), 0)
        self.cb_fmt.current(_init)
        row("Sample format", self.cb_fmt)
        row("Sample rate (Sps)", ttk.Entry(s, textvariable=self.var_rate))

        head("ANALYSIS (OpenV2K defaults)")
        row("FFT size", ttk.Combobox(s, state="readonly", width=8,
                                     values=["64", "128", "256", "512", "1024",
                                             "2048"],
                                     textvariable=self.var_fft))
        row("Hop (samples)", ttk.Entry(s, textvariable=self.var_hop))
        fr = ttk.Frame(s)
        fr.pack(fill=tk.X, pady=1)
        ttk.Label(fr, text="Threshold (% of peak)", width=20,
                  anchor="w").pack(side=tk.LEFT)
        self.lbl_thr = ttk.Label(fr, text="5.0 %", width=7)
        self.lbl_thr.pack(side=tk.RIGHT)
        sc = ttk.Scale(s, from_=0.5, to=25.0, variable=self.var_thresh,
                       command=lambda v: self.lbl_thr.config(
                           text="{:.1f} %".format(float(v))))
        sc.pack(fill=tk.X)
        row("Pre / post pad (ms)", ttk.Entry(s, textvariable=self.var_pre))
        row("Max active (s)", ttk.Entry(s, textvariable=self.var_maxact))
        row("Min pulse width (us)", ttk.Entry(s, textvariable=self.var_minw))
        row("Merge gap (us)", ttk.Entry(s, textvariable=self.var_gap))
        row("Max display columns", ttk.Entry(s, textvariable=self.var_cols))

        head("DURATION CHECKSUM")
        row("Source active (ms)", ttk.Entry(s, textvariable=self.var_srcms))
        wl = ttk.Label(s, textvariable=self.var_wav, wraplength=330,
                       justify="left", foreground="#8fb6d9")
        wl.pack(anchor="w", pady=(2, 0))

        head("FILE INFO")
        self.info = tk.Text(s, height=7, width=40, bg="#171a1f", fg="#d7dde5",
                            relief="flat", font=("DejaVu Sans Mono", 9),
                            wrap="word")
        self.info.pack(fill=tk.X, pady=2)
        self.info.insert("1.0", "No file loaded.")
        self.info.config(state="disabled")

        head("METRICS")
        self.metrics_txt = tk.Text(s, height=10, width=40, bg="#171a1f",
                                   fg="#c6f0c6", relief="flat",
                                   font=("DejaVu Sans Mono", 9), wrap="word")
        self.metrics_txt.pack(fill=tk.BOTH, expand=True, pady=2)
        self.metrics_txt.insert("1.0", "-")
        self.metrics_txt.config(state="disabled")

        ttk.Button(s, text="Copy report to clipboard",
                   command=self.copy_report).pack(fill=tk.X, pady=(6, 0))
        ttk.Button(s, text="Export everything\u2026",
                   command=self.export_all).pack(fill=tk.X, pady=2)

    # -- tabs -----------------------------------------------------------
    def _new_fig(self, name):
        fig = Figure(figsize=(9, 6), dpi=100, facecolor="#20242b")
        canvas = FigureCanvasTkAgg(fig, master=self.nb_frames[name])
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        tb = NavigationToolbar2Tk(canvas, self.nb_frames[name], pack_toolbar=False)
        self._toolbars[name] = tb
        self._canvases[name] = canvas
        self._figs[name] = fig
        return fig

    def _build_tabs(self):
        self.nb = ttk.Notebook(self.right)
        self.nb.pack(fill=tk.BOTH, expand=True)
        self.nb_frames = {}

        for name in ("Waterfall", "Envelope && Pulses", "Pulse Zoom", "Spectrum",
                     "Constellation"):
            fr = ttk.Frame(self.nb)
            self.nb.add(fr, text=name.replace("&&", "&"))
            self.nb_frames[name] = fr

        # Pulse zoom controls must be packed BEFORE that tab's canvas so the
        # canvas lays out underneath them.
        zf = ttk.Frame(self.nb_frames["Pulse Zoom"], padding=(4, 2))
        zf.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(zf, text="Window start (ms):").pack(side=tk.LEFT)
        ttk.Entry(zf, textvariable=self.var_zoom_start, width=8).pack(side=tk.LEFT)
        ttk.Label(zf, text="  Span (ms):").pack(side=tk.LEFT)
        ttk.Entry(zf, textvariable=self.var_zoom_span, width=8).pack(side=tk.LEFT)
        ttk.Button(zf, text="Redraw", command=self.render_all).pack(
            side=tk.LEFT, padx=8)
        ttk.Label(zf, text="(pulse edges marked from the detected pulse table)",
                  foreground="#8fb6d9").pack(side=tk.LEFT)

        for name in ("Waterfall", "Envelope && Pulses", "Pulse Zoom", "Spectrum",
                     "Constellation"):
            self._new_fig(name)

        # Report tab
        rf = ttk.Frame(self.nb)
        self.nb.add(rf, text="Report")
        self.nb_frames["Report"] = rf
        self.report = tk.Text(rf, bg="#171a1f", fg="#dbe6f3", relief="flat",
                              font=("DejaVu Sans Mono", 9), wrap="none")
        sb = ttk.Scrollbar(rf, orient=tk.VERTICAL, command=self.report.yview)
        self.report.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.report.pack(fill=tk.BOTH, expand=True)

        # Event log tab
        lf = ttk.Frame(self.nb)
        self.nb.add(lf, text="Event Log")
        self.nb_frames["Event Log"] = lf
        ctl = ttk.Frame(lf, padding=(4, 2))
        ctl.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(ctl, text="Parse log", command=self.parse_log).pack(side=tk.LEFT)
        ttk.Button(ctl, text="Clear", command=lambda: self.logtxt.delete(
            "1.0", tk.END)).pack(side=tk.LEFT, padx=4)
        ttk.Label(ctl, text="  paste an OpenV2K event log here (or File > Load "
                            "event log)", foreground="#8fb6d9").pack(side=tk.LEFT)
        self.logtxt = tk.Text(lf, bg="#171a1f", fg="#dbe6f3", relief="flat",
                              font=("DejaVu Sans Mono", 9), wrap="none")
        self.logtxt.pack(fill=tk.BOTH, expand=True)

        self.nb.bind("<<NotebookTabChanged>>", lambda e: self._sync_toolbar())
        self._sync_toolbar()

    def _sync_toolbar(self):
        try:
            idx = self.nb.index(self.nb.select())
            label = self.nb.tab(idx, "text").replace("&", "&&")
        except Exception:
            return
        key = {"Waterfall": "Waterfall", "Envelope & Pulses": "Envelope && Pulses",
               "Pulse Zoom": "Pulse Zoom", "Spectrum": "Spectrum",
               "Constellation": "Constellation"}.get(label)
        for k, tb in self._toolbars.items():
            if k == key:
                tb.pack(side=tk.BOTTOM, fill=tk.X)
            else:
                tb.pack_forget()

    def _bind_keys(self):
        self.bind("<Control-o>", lambda e: self.on_open())
        self.bind("<Control-w>", lambda e: self.on_open_wav())
        self.bind("<F5>", lambda e: self.on_reanalyze())
        self.bind("<Control-s>", lambda e: self.export_txt())
        self.bind("<Control-q>", lambda e: self._on_close())

    # -- helpers --------------------------------------------------------
    def _set_text(self, widget, text):
        widget.config(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.config(state="disabled")

    def _fmt_key(self):
        label = self.cb_fmt.get()
        for k, spec in FORMATS.items():
            if spec.label == label:
                return k
        return "complex64"

    def _params(self):
        def f(var, default):
            try:
                return float(str(var.get()).strip())
            except Exception:
                return default
        return dict(
            fmt_key=self._fmt_key(),
            rate=f(self.var_rate, OPENV2K_RATE),
            fft_size=int(f(self.var_fft, DEFAULT_FFT)),
            hop=max(1, int(f(self.var_hop, DEFAULT_HOP))),
            thresh_factor=max(1e-4, f(self.var_thresh, 5.0) / 100.0),
            pre_ms=f(self.var_pre, DEFAULT_PRE_MS),
            post_ms=f(self.var_post, DEFAULT_POST_MS),
            max_active_s=max(0.01, f(self.var_maxact, DEFAULT_MAX_ACTIVE_S)),
            max_display_cols=max(128, int(f(self.var_cols, 2048))),
            min_width_us=f(self.var_minw, 1.0),
            merge_gap_us=f(self.var_gap, 5.0),
            src_wav=None if self.var_wav.get().startswith("(none") else
            self.var_wav.get(),
            src_ms_manual=(f(self.var_srcms, 0.0) or None),
        )

    # -- file actions ---------------------------------------------------
    def on_open(self):
        p = filedialog.askopenfilename(
            title="Open OpenV2K IQ recording",
            initialdir=os.path.expanduser("~"),
            filetypes=[("OpenV2K / raw IQ", "*.iq *.cfile *.bin *.dat *.raw *.cs16"),
                       ("All files", "*.*")])
        if p:
            self.load_file(p)

    def on_open_wav(self):
        p = filedialog.askopenfilename(
            title="Open source audio used to generate the recording",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")])
        if p:
            self.var_wav.set(p)
            self.status.set("Source WAV: {}".format(p))
            self.on_reanalyze()

    def on_open_log(self):
        p = filedialog.askopenfilename(
            title="Open OpenV2K event log",
            filetypes=[("Text/log", "*.txt *.log *.md"), ("All files", "*.*")])
        if p:
            try:
                with open(p, "r", errors="replace") as fh:
                    self.logtxt.delete("1.0", tk.END)
                    self.logtxt.insert("1.0", fh.read())
                self.nb.select(self.nb_frames["Event Log"])
                self.parse_log()
            except Exception as exc:
                messagebox.showerror("Event log", str(exc))

    def on_scan_home(self):
        home = os.path.expanduser("~")
        found = []
        try:
            for name in sorted(os.listdir(home)):
                if SAMPLE_RE.match(name):
                    found.append(os.path.join(home, name))
        except Exception as exc:
            messagebox.showerror("Scan", str(exc))
            return
        if not found:
            messagebox.showinfo(
                "Scan home",
                "No OpenV2K_*.iq recordings found in {}.\n\nUse Record IQ in "
                "the OpenV2K app, or File > Open.".format(home))
            return
        win = tk.Toplevel(self)
        win.title("OpenV2K recordings in {}".format(home))
        win.geometry("760x420")
        ttk.Label(win, text="{} recording(s) found - double-click to load".format(
            len(found))).pack(anchor="w", padx=6, pady=4)
        lb = tk.Listbox(win, bg="#171a1f", fg="#dbe6f3", font=("DejaVu Sans Mono", 9))
        lb.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        for p in found:
            try:
                sz = os.path.getsize(p) / 1e6
                dur = sz * 1e6 / 8 / OPENV2K_RATE
                lb.insert(tk.END, "{:<52} {:8.1f} MB  ~{:6.2f} s".format(
                    os.path.basename(p), sz, dur))
            except Exception:
                lb.insert(tk.END, p)

        def choose(_e=None):
            sel = lb.curselection()
            if sel:
                win.destroy()
                self.load_file(found[sel[0]])
        lb.bind("<Double-Button-1>", choose)
        ttk.Button(win, text="Load selected", command=choose).pack(pady=4)

    def load_file(self, path):
        self.recent.insert(0, path)
        self.recent = self.recent[:8]
        self._refresh_recent()
        self._start_worker(path)

    def _refresh_recent(self):
        self.menu_recent.delete(0, tk.END)
        for p in self.recent:
            self.menu_recent.add_command(
                label=os.path.basename(p),
                command=lambda p=p: self.load_file(p))

    def on_reanalyze(self):
        if self.res and not self.busy:
            self._start_worker(self.res.path)

    def _start_worker(self, path):
        if self.busy:
            return
        if not os.path.isfile(path):
            messagebox.showerror("Open", "Not a file:\n{}".format(path))
            return
        self.busy = True
        self.pb.configure(mode="determinate", value=0)
        self.status.set("Analysing {} ...".format(os.path.basename(path)))
        params = self._params()
        fmt = FORMATS[params["fmt_key"]]
        guess = "complex64 IQ" if fmt.complex_out else "real mono"
        self._set_text(self.info,
                       "file   : {}\nsize   : {:,} bytes\nformat : {}\n"
                       "rate   : {:g} Sps\ndur    : {:.3f} s (if {})\n".format(
                           os.path.basename(path), os.path.getsize(path),
                           fmt.label, params["rate"],
                           os.path.getsize(path) /
                           (np.dtype(fmt.dtype).itemsize * fmt.words_per_sample)
                           / params["rate"], guess))

        def work():
            try:
                res = analyze_iq(
                    path, progress=lambda p, m: self.q.put(("progress", (p, m))),
                    **params)
                self.q.put(("done", res))
            except Exception:
                self.q.put(("error", traceback.format_exc()))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "progress":
                    pct, msg = payload
                    self.pb.configure(value=pct)
                    self.status.set("{} ... {:.0f}%".format(msg, pct))
                elif kind == "done":
                    self.busy = False
                    self.pb.configure(value=100)
                    self._on_result(payload)
                elif kind == "error":
                    self.busy = False
                    self.status.set("Analysis failed.")
                    messagebox.showerror("Analysis failed", payload)
        except queue.Empty:
            pass
        self.after(80, self._poll)

    # -- results --------------------------------------------------------
    def _on_result(self, res: AnalysisResult):
        self.res = res
        self._set_text(self.info,
                       "file   : {}\nsize   : {:,} bytes\nformat : {}\n"
                       "rate   : {:.0f} Sps\nsamples: {:,}\ndur    : "
                       "{:.3f} s\nwindow : {:,} .. {:,}\nactive : {:.3f} ms\n"
                       "warn   : {}".format(
                           os.path.basename(res.path), res.size_bytes,
                           res.fmt_label.split(" - ")[0], res.rate, res.samples,
                           res.duration_s, res.win_start, res.win_end,
                           res.active_ms,
                           len(res.warnings)))
        p = res.pulses
        ck = res.checksum
        ck_txt = ("not evaluated (need source WAV / manual ms)"
                  if ck.src_ms <= 0 else
                  "{}  [{:.0f} ms vs {:.0f} ms, ratio {:.2f}x]".format(
                      "OK" if ck.ok else "MISMATCH", ck.active_ms, ck.src_ms,
                      ck.ratio))
        self._set_text(self.metrics_txt,
                       "active region : {:.3f} ms\n"
                       "pulses        : {}\n"
                       "mean width    : {:.2f} us\n"
                       "median width  : {:.2f} us\n"
                       "width range   : {:.2f} .. {:.2f} us\n"
                       "PRF           : {:.2f} Hz\n"
                       "jitter        : {:.2f} us rms\n"
                       "duty cycle    : {:.2f} %\n"
                       "peak / RMS    : {:.2f} / {:.2f} dBFS\n"
                       "PAPR          : {:.2f} dB\n"
                       "median peak   : {:.2f} dBFS\n"
                       "DC offset     : {:+.2e} / {:+.2e}\n"
                       "clipped       : {}\n"
                       "spectral peak : {:+.3f} MHz\n"
                       "checksum      : {}".format(
                           res.active_ms, p.count, p.mean_width_us,
                           p.median_width_us, p.min_width_us, p.max_width_us,
                           p.prf_hz, p.jitter_us, p.duty_pct, res.peak_dbfs,
                           res.rms_dbfs, res.papr_db, res.median_dbfs, res.dc_i,
                           res.dc_q, res.clip_count, res.psd_peak_mhz, ck_txt))

        self._set_text(self.report, build_report(res))
        if res.checksum.src_ms > 0:
            self.logtxt.insert(tk.END, (
                "Checksum: active={:.0f}ms vs source active-region={:.0f}ms "
                "(ratio={:.2f}x) -- {}\n").format(
                    res.checksum.active_ms, res.checksum.src_ms,
                    res.checksum.ratio,
                    "OK" if res.checksum.ok else
                    "MISMATCH, check signal chain for ringing/gating"))
        self.render_all()
        msg = "{}  |  {} pulses, {:.1f} us mean, {:.1f}% duty, {} checksum".format(
            os.path.basename(res.path), p.count, p.mean_width_us, p.duty_pct,
            "OK" if res.checksum.ok else
            ("MISMATCH" if res.checksum.src_ms > 0 else "n/a"))
        if res.warnings:
            msg += "  |  {} warning(s), see Report tab".format(len(res.warnings))
        self.status.set(msg)

    # -- plotting -------------------------------------------------------
    def render_all(self):
        self.render_waterfall()
        self.render_envelope()
        self.render_zoom()
        self.render_spectrum()
        self.render_constellation()

    def _clear(self, name):
        fig = self._figs[name]
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor("#161a1f")
        for sp in ax.spines.values():
            sp.set_color("#5b6472")
        ax.tick_params(colors="#c9d3e0", labelsize=8)
        return ax

    def render_waterfall(self):
        fig = self._figs["Waterfall"]
        ax = self._clear("Waterfall")
        ax.title.set_color("#e8eef7")
        res = self.res
        if res is None or res.spec_db.size == 0:
            ax.text(0.5, 0.5, "No spectrogram yet - open an IQ recording",
                    ha="center", va="center", color="#8fb6d9")
            ax.set_axis_off()
            self._canvases["Waterfall"].draw_idle()
            return
        t0, t1, f0, f1 = res.spec_extent
        if not self.var_full.get():
            t0 = max(t0, res.win_start / res.rate * 1000.0)
        vmax = float(np.percentile(res.spec_db, 99.5))
        im = ax.imshow(res.spec_db.T, aspect="auto", origin="lower",
                       extent=[t0, t1, f0, f1], cmap="inferno",
                       interpolation="nearest", vmin=vmax - 55.0, vmax=vmax)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Freq offset (MHz)")
        ax.set_title("OpenV2K pulse waterfall  |  {}  |  {:.0f} ms active  |  "
                     "hop {:.1f} us/frame  |  {} cols  |  {} x {}  |  "
                     "FFT {}".format(
                         os.path.basename(res.path), res.active_ms,
                         res.spec_step_us, res.spec_cols,
                         res.spec_db.shape[0], res.spec_db.shape[1],
                         res.params.get("fft_size")), fontsize=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label("Power (dB)", color="#c9d3e0", fontsize=8)
        cb.ax.tick_params(colors="#c9d3e0", labelsize=7)
        ax.set_xlim(t0, t1)
        fig.tight_layout()
        self._canvases["Waterfall"].draw_idle()

    def render_envelope(self):
        fig = self._figs["Envelope && Pulses"]
        ax = self._clear("Envelope && Pulses")
        ax.title.set_color("#e8eef7")
        res = self.res
        if res is None or res.env_mag.size == 0:
            ax.text(0.5, 0.5, "No signal analysed yet", ha="center", va="center",
                    color="#8fb6d9")
            ax.set_axis_off()
            self._canvases["Envelope && Pulses"].draw_idle()
            return
        full = self.var_full.get()
        if full and res.env_full_mag.size:
            ax.plot(res.env_full_t_ms, res.env_full_mag, color="#4ea1ff",
                    linewidth=0.6)
            ax.set_xlabel("Time (ms)")
            ax.set_title("Full-file magnitude envelope ({} px, peak-preserving "
                         "decimation)  |  {} samples  |  {:.3f} s".format(
                             res.env_full_mag.size, res.samples,
                             res.duration_s), fontsize=9)
        else:
            ax.plot(res.env_t_ms, res.env_mag, color="#4ea1ff", linewidth=0.6)
            ax.axhline(res.thresh_lin, color="#ff7a59", linewidth=0.9,
                       linestyle="--", label="threshold = peak x {:.3g}".format(
                           res.thresh_factor))
            show = res.pulses.starts_s
            if show.size and show.size <= 3000:
                ax.vlines(show * 1e3, 0, res.peak_lin, color="#ffd166",
                          linewidth=0.4, alpha=0.75,
                          label="{} detected pulse starts".format(show.size))
            ax.legend(loc="upper right", fontsize=7, facecolor="#20242b",
                      edgecolor="#5b6472", labelcolor="#c9d3e0")
            ax.set_title("Active-region envelope  |  {:.0f} ms  |  {} pulses  |  "
                         "{:.1f} us mean width  |  {:.2f} Hz PRF  |  "
                         "{:.2f} % duty".format(
                             res.active_ms, res.pulses.count,
                             res.pulses.mean_width_us, res.pulses.prf_hz,
                             res.pulses.duty_pct), fontsize=9)
            ax.set_xlabel("Time (ms)")
        ax.set_ylabel("|IQ|")
        fig.tight_layout()
        self._canvases["Envelope && Pulses"].draw_idle()

    def render_zoom(self):
        fig = self._figs["Pulse Zoom"]
        ax = self._clear("Pulse Zoom")
        ax.title.set_color("#e8eef7")
        res = self.res
        if res is None or res.env_mag.size == 0:
            ax.set_axis_off()
            self._canvases["Pulse Zoom"].draw_idle()
            return
        try:
            start_ms = float(self.var_zoom_start.get())
        except Exception:
            start_ms = 0.0
        try:
            span_ms = max(0.01, float(self.var_zoom_span.get()))
        except Exception:
            span_ms = 25.0
        t0_abs = res.win_start / res.rate * 1000.0
        x0 = t0_abs + start_ms
        x1 = x0 + span_ms
        ax.plot(res.env_t_ms, res.env_mag, color="#4ea1ff", linewidth=0.9)
        ax.axhline(res.thresh_lin, color="#ff7a59", linewidth=0.8,
                   linestyle="--")
        k = 0
        for s, w, pk in zip(res.pulses.starts_s, res.pulses.widths_us,
                            res.pulses.peaks):
            s_ms = s * 1e3
            w_ms = w * 1e-3
            if s_ms + w_ms < x0 or s_ms > x1:
                continue
            ax.axvspan(s_ms, s_ms + w_ms, color="#ffd166", alpha=0.35)
            ax.annotate("{:.1f} us".format(w),
                        xy=(s_ms + w_ms / 2.0, min(1.0, pk * 1.02)),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", fontsize=6, color="#ffe9a8")
            k += 1
            if k >= 500:
                break
        ax.set_xlim(x0, x1)
        ax.set_ylim(0, max(res.peak_lin * 1.15, 1e-6))
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("|IQ|")
        ax.set_title("Pulse zoom  |  {} pulse(s) in {:.3f} - {:.3f} ms  |  "
                     "threshold {:.3g}".format(k, x0, x1, res.thresh_lin),
                     fontsize=9)
        fig.tight_layout()
        self._canvases["Pulse Zoom"].draw_idle()

    def render_spectrum(self):
        fig = self._figs["Spectrum"]
        ax = self._clear("Spectrum")
        ax.title.set_color("#e8eef7")
        res = self.res
        if res is None or res.psd_db.size == 0:
            ax.text(0.5, 0.5, "No spectrum yet", ha="center", va="center",
                    color="#8fb6d9")
            ax.set_axis_off()
            self._canvases["Spectrum"].draw_idle()
            return
        ax.plot(res.psd_freq_mhz, res.psd_db, color="#7ee787", linewidth=0.7)
        ax.axvline(res.psd_peak_mhz, color="#ff7a59", linewidth=0.8,
                   linestyle="--",
                   label="peak {:+.4f} MHz".format(res.psd_peak_mhz))
        ax.set_xlabel("Freq offset (MHz)")
        ax.set_ylabel("Power (dB/Hz)")
        ax.set_title("Averaged spectrum of active region (Welch, 1024 pt, 50% "
                     "overlap)", fontsize=9)
        ax.legend(loc="upper right", fontsize=7, facecolor="#20242b",
                  edgecolor="#5b6472", labelcolor="#c9d3e0")
        fig.tight_layout()
        self._canvases["Spectrum"].draw_idle()

    def render_constellation(self):
        fig = self._figs["Constellation"]
        ax = self._clear("Constellation")
        ax.title.set_color("#e8eef7")
        res = self.res
        if res is None or res.iq_i.size == 0:
            ax.set_axis_off()
            self._canvases["Constellation"].draw_idle()
            return
        ax.scatter(res.iq_i, res.iq_q, s=3, c=np.arange(res.iq_i.size),
                   cmap="viridis", alpha=0.5, linewidths=0)
        ax.axhline(0, color="#5b6472", linewidth=0.6)
        ax.axvline(0, color="#5b6472", linewidth=0.6)
        m = max(1e-3, float(max(np.abs(res.iq_i).max(), np.abs(res.iq_q).max())))
        ax.set_xlim(-m * 1.1, m * 1.1)
        ax.set_ylim(-m * 1.1, m * 1.1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_title("I/Q constellation, active region ({} points, colour = time, "
                     "DC {:.2e}/{:.2e})".format(
                         res.iq_i.size, res.dc_i, res.dc_q), fontsize=9)
        fig.tight_layout()
        self._canvases["Constellation"].draw_idle()

    # -- misc actions ---------------------------------------------------
    def parse_log(self):
        text = self.logtxt.get("1.0", tk.END)
        info = parse_log_text(text)
        lines = ["Parsed OpenV2K event log", "-" * 60]
        any_found = False
        for name, hits in info["params"].items():
            for val, raw in hits:
                lines.append("{:<12}: {}".format(name, raw))
                any_found = True
        ck = info["checksum"]
        if ck["active_ms"] is not None or ck["src_ms"] is not None:
            any_found = True
            lines += ["", "Checksum line found in log:", "  active={} ms  "
                      "source={} ms  ratio={}  verdict={}".format(
                          ck["active_ms"], ck["src_ms"], ck["ratio"],
                          ck["verdict"] or "-")]
            if self.res and self.res.checksum.src_ms > 0:
                lines.append("  parser measured active={:.0f} ms  -> {}".format(
                    self.res.active_ms,
                    "agrees with log" if ck["active_ms"] and abs(
                        ck["active_ms"] - self.res.active_ms) <=
                    max(30.0, 0.05 * ck["active_ms"]) else
                    "differs from log (different window/threshold?)"))
        if not any_found:
            lines.append("No OpenV2K parameter/checksum lines recognised."
                         " Expected lines look like:")
            lines.append('  "Pulse width: 100 us"')
            lines.append('  "Checksum: active=1200ms vs eSpeak active-region='
                         '1180ms (ratio=1.02x) -- OK"')
            lines.append('  "Filter ON -- 50/60 Hz Notch"')
        self.logtxt.insert(tk.END, "\n\n" + "\n".join(lines) + "\n")
        self.nb.select(self.nb_frames["Event Log"])

    def copy_report(self):
        if not self.res:
            return
        self.clipboard_clear()
        self.clipboard_append(build_report(self.res))
        self.status.set("Report copied to clipboard.")

    def _ask_path(self, defname, types):
        return filedialog.asksaveasfilename(initialfile=defname, filetypes=types)

    def export_txt(self):
        if not self.res:
            messagebox.showinfo("Export", "Nothing to export yet.")
            return
        p = self._ask_path(
            os.path.splitext(os.path.basename(self.res.path))[0] + "_report.txt",
            [("Text report", "*.txt"), ("All files", "*.*")])
        if not p:
            return
        with open(p, "w") as fh:
            fh.write(build_report(self.res))
        self.status.set("Report written: {}".format(p))

    def export_json(self):
        if not self.res:
            messagebox.showinfo("Export", "Nothing to export yet.")
            return
        p = self._ask_path(
            os.path.splitext(os.path.basename(self.res.path))[0] + "_metrics.json",
            [("JSON", "*.json"), ("All files", "*.*")])
        if not p:
            return
        d = self.res.metrics_dict()
        d["parser"] = {"name": "openv2k_parser", "version": VERSION}
        with open(p, "w") as fh:
            json.dump(d, fh, indent=2, default=float)
        self.status.set("Metrics written: {}".format(p))

    def export_csv(self):
        if not self.res:
            messagebox.showinfo("Export", "Nothing to export yet.")
            return
        p = self._ask_path(
            os.path.splitext(os.path.basename(self.res.path))[0] + "_pulses.csv",
            [("CSV", "*.csv"), ("All files", "*.*")])
        if not p:
            return
        with open(p, "w") as fh:
            fh.write(pulses_csv(self.res))
        self.status.set("Pulse table written: {}".format(p))

    def export_png(self, which):
        if not self.res:
            messagebox.showinfo("Export", "Nothing to analyse yet.")
            return
        if which == "__current__":
            idx = self.nb.index(self.nb.select())
            label = self.nb.tab(idx, "text").replace("&", "&&")
            key = {"Waterfall": "Waterfall", "Envelope & Pulses":
                   "Envelope && Pulses", "Pulse Zoom": "Pulse Zoom",
                   "Spectrum": "Spectrum", "Constellation": "Constellation"}.get(
                       label)
            if key is None:
                messagebox.showinfo("Export", "This tab has no figure.")
                return
        else:
            key = which
        p = self._ask_path(
            os.path.splitext(os.path.basename(self.res.path))[0] + "_" +
            key.replace(" & ", "_").replace(" ", "_").replace("&&", "&") + ".png",
            [("PNG image", "*.png"), ("All files", "*.*")])
        if not p:
            return
        self._figs[key].savefig(p, dpi=150, facecolor=self._figs[key].get_facecolor())
        self.status.set("Figure written: {}".format(p))

    def export_all(self):
        if not self.res:
            messagebox.showinfo("Export", "Nothing to analyse yet.")
            return
        d = filedialog.askdirectory(title="Choose an output directory")
        if not d:
            return
        base = os.path.splitext(os.path.basename(self.res.path))[0]
        outs = []
        rp = os.path.join(d, base + "_report.txt")
        with open(rp, "w") as fh:
            fh.write(build_report(self.res))
        outs.append(rp)
        jp = os.path.join(d, base + "_metrics.json")
        with open(jp, "w") as fh:
            json.dump(self.res.metrics_dict(), fh, indent=2, default=float)
        outs.append(jp)
        cp = os.path.join(d, base + "_pulses.csv")
        with open(cp, "w") as fh:
            fh.write(pulses_csv(self.res))
        outs.append(cp)
        for key in ("Waterfall", "Envelope && Pulses", "Pulse Zoom", "Spectrum",
                    "Constellation"):
            if key not in self._figs:
                continue
            fp = os.path.join(d, base + "_" + key.replace("&&", "&").replace(
                " & ", "_").replace(" ", "_") + ".png")
            self._figs[key].savefig(fp, dpi=150,
                                    facecolor=self._figs[key].get_facecolor())
            outs.append(fp)
        self.status.set("Exported {} file(s) to {}".format(len(outs), d))
        messagebox.showinfo("Export complete",
                            "Written:\n" + "\n".join(os.path.basename(o)
                                                     for o in outs))

    def show_about(self):
        messagebox.showinfo(
            "About OpenV2K IQ Parser",
            "OpenV2K IQ Parser {}\n\n"
            "Offline parser/validator for OpenV2K IQ recordings.\n"
            "Reproduces OpenV2K's analysis constants: fft_size=64, Hann window, "
            "hop=32 samples (16 us/frame), threshold = 5% of peak |IQ|, +/-1 ms "
            "active-region padding, 3 s active cap, and the active-duration "
            "checksum against the source audio.\n\n"
            "Adds per-pulse measurement (count, width, PRF, duty cycle, "
            "jitter), Welch spectrum, constellation, an event-log parser and "
            "TXT/JSON/CSV/PNG export.".format(VERSION))

    def _on_close(self):
        if self.busy:
            if not messagebox.askyesno("Quit", "Analysis is still running. "
                                       "Quit anyway?"):
                return
        self.destroy()

    # -- programmatic entry point for --selftest -------------------------
    def selftest_load(self, path, timeout=120.0):
        self.load_file(path)
        t0 = time.time()
        while self.busy and time.time() - t0 < timeout:
            self.update()
            time.sleep(0.02)
        for _ in range(40):
            self.update()
            time.sleep(0.01)
        return self.res

    def save_all_figures(self, outdir, prefix="tab"):
        os.makedirs(outdir, exist_ok=True)
        paths = []
        for key in ("Waterfall", "Envelope && Pulses", "Pulse Zoom", "Spectrum",
                    "Constellation"):
            if key not in self._figs:
                continue
            p = os.path.join(outdir, "{}_{}.png".format(
                prefix, key.replace("&&", "&").replace(" & ", "_").replace(
                    " ", "_")))
            self._figs[key].savefig(p, dpi=130,
                                    facecolor=self._figs[key].get_facecolor())
            paths.append(p)
        return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def selftest_main(outdir: str, quiet: bool = False) -> int:
    os.makedirs(outdir, exist_ok=True)
    iq = os.path.join(outdir, "OpenV2K_selftest.iq")
    wav_ok = os.path.join(outdir, "selftest_source_ok.wav")
    wav_bad = os.path.join(outdir, "selftest_source_bad.wav")

    meta = generate_sample_iq(iq, pulse_us=100.0, prf_hz=500.0, active_s=1.0)
    generate_sample_wav(wav_ok, meta["active_s"] * 1000.0 + 2.0)
    generate_sample_wav(wav_bad, 300.0)
    print("[fixture] {}  samples={:,} pulses={} pulse_us={} prf={} Hz".format(
        iq, meta["samples"], meta["pulses"], meta["pulse_us"], meta["prf_hz"]))

    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print("  [{}] {:<44} {}".format("PASS" if ok else "FAIL", name, detail))

    print("\n[1] headless analysis (numpy only)")
    res = analyze_iq(iq, src_wav=wav_ok,
                     progress=(lambda p, m: None) if quiet else None)
    exp_pulses = meta["pulses"]
    check("pulse count within 10%",
          abs(res.pulses.count - exp_pulses) <= max(2, 0.1 * exp_pulses),
          "found {} vs expected {}".format(res.pulses.count, exp_pulses))
    check("mean width within 15%",
          abs(res.pulses.mean_width_us - 100.0) <= 15.0,
          "{:.2f} us".format(res.pulses.mean_width_us))
    check("PRF within 5%",
          abs(res.pulses.prf_hz - 500.0) <= 25.0,
          "{:.2f} Hz".format(res.pulses.prf_hz))
    check("duty cycle within 1 pt of 5%",
          abs(res.pulses.duty_pct - 5.0) <= 1.0,
          "{:.2f} %".format(res.pulses.duty_pct))
    check("active region ~1000 ms",
          abs(res.active_ms - 1002.0) <= 60.0, "{:.1f} ms".format(res.active_ms))
    check("spectrogram built (fft 64, hop 32)",
          res.spec_db.size > 0 and res.spec_db.shape[1] == 64,
          "shape {}".format(res.spec_db.shape))
    check("streamed spectrum present", res.psd_db.size > 0,
          "{} bins, peak {:+.3f} MHz".format(res.psd_db.size, res.psd_peak_mhz))
    check("checksum OK with matching WAV", res.checksum.ok,
          res.checksum.note)
    check("peak level sane", -20 < res.peak_dbfs < 3,
          "{:.2f} dBFS".format(res.peak_dbfs))

    print("\n[2] checksum mismatch detection")
    res_bad = analyze_iq(iq, src_wav=wav_bad)
    check("checksum flags 300 ms source as MISMATCH",
          not res_bad.checksum.ok,
          "active={:.0f} ms vs src={:.0f} ms ratio={:.2f}x".format(
              res_bad.checksum.active_ms, res_bad.checksum.src_ms,
              res_bad.checksum.ratio))

    print("\n[3] format handling (int16 interleaved IQ of same data)")
    i16 = os.path.join(outdir, "OpenV2K_selftest_int16.iq")
    d = np.fromfile(iq, dtype=np.complex64)
    iq_i16 = np.empty(d.size * 2, dtype=np.int16)
    iq_i16[0::2] = np.clip(d.real * 32767.0, -32768, 32767).astype(np.int16)
    iq_i16[1::2] = np.clip(d.imag * 32767.0, -32768, 32767).astype(np.int16)
    iq_i16.tofile(i16)
    res_i16 = analyze_iq(i16, fmt_key="int16_iq")
    check("int16 IQ parses with same pulse count",
          abs(res_i16.pulses.count - res.pulses.count) <= 2,
          "{} vs {}".format(res_i16.pulses.count, res.pulses.count))
    check("int16 mean width within 15%",
          abs(res_i16.pulses.mean_width_us - 100.0) <= 15.0,
          "{:.2f} us".format(res_i16.pulses.mean_width_us))

    print("\n[4] log parser")
    logdemo = (
        "Pulse width: 100 \u00b5s\n"
        "HPF: 300 Hz\n"
        "LPF: 3000 Hz\n"
        "Filter ON -- 50/60 Hz Notch\n"
        "Waterfall: 1000ms active  |  40000x900 px  |  40.0px/ms\n"
        "Checksum: active=1002ms vs eSpeak active-region=1000ms "
        "(ratio=1.00x) -- OK\n")
    info = parse_log_text(logdemo)
    check("log parser extracts pulse width",
          any("100" in v for v, _ in info["params"].get("Pulse width", [])),
          str([v for v, _ in info["params"].get("Pulse width", [])]))
    check("log parser extracts checksum verdict",
          info["checksum"]["verdict"] == "OK" and
          info["checksum"]["active_ms"] == 1002.0,
          str(info["checksum"]))

    print("\n[5] exports")
    rp = os.path.join(outdir, "selftest_report.txt")
    with open(rp, "w") as fh:
        fh.write(build_report(res))
    cp = os.path.join(outdir, "selftest_pulses.csv")
    with open(cp, "w") as fh:
        fh.write(pulses_csv(res))
    jp = os.path.join(outdir, "selftest_metrics.json")
    with open(jp, "w") as fh:
        json.dump(res.metrics_dict(), fh, indent=2, default=float)
    check("report/CSV/JSON written",
          all(os.path.getsize(p) > 100 for p in (rp, cp, jp)),
          "{} / {} / {} bytes".format(os.path.getsize(rp), os.path.getsize(cp),
                                      os.path.getsize(jp)))

    print("\n[6] GUI tabs (Tk + Matplotlib)")
    if not os.environ.get("DISPLAY"):
        print("  [SKIP] no DISPLAY - GUI smoke test not executed "
              "(run under Xvfb)")
    else:
        app = ParserApp(initial_file=None)
        app.update()
        r = app.selftest_load(iq)
        app.var_wav.set(wav_ok)
        app.var_zoom_start.set("0")
        app.var_zoom_span.set("10")
        app.render_all()
        app.update()
        pngs = app.save_all_figures(outdir, prefix="selftest_tab")
        widths = []
        for p in pngs:
            widths.append("{}:{}B".format(os.path.basename(p),
                                          os.path.getsize(p)))
        check("GUI built the full window", len(app.nb.tabs()) == len(app.TABS),
              "{} tabs".format(len(app.nb.tabs())))
        app.geometry("1500x950")
        for _ in range(8):
            app.update()
            time.sleep(0.02)
        app._claim_sidebar_width()
        app.update_idletasks()
        app.update()
        side_w = app.side_holder.winfo_width()
        nb_w = app.nb.winfo_width()
        check("sidebar stays fixed width, notebook keeps the rest",
              300 <= side_w <= 460 and nb_w > 500,
              "sidebar {}px, notebook {}px".format(side_w, nb_w))
        check("sidebar content packs into rows (sane requested width)",
              app.side.winfo_reqwidth() <= 460,
              "requires {}px".format(app.side.winfo_reqwidth()))
        stray = [w.winfo_class() for w in app.side.pack_slaves()
                 if w.winfo_class() in ("TEntry", "TCombobox")]
        check("input widgets live inside their label rows", not stray,
              "stray at sidebar root: {}".format(stray))
        check("GUI analysis matches headless",
              r is not None and r.pulses.count == res.pulses.count,
              "{} pulses".format(None if r is None else r.pulses.count))
        check("all figure tabs rendered to PNG",
              all(os.path.getsize(p) > 2000 for p in pngs), " ".join(widths))
        app.logtxt.delete("1.0", tk.END)
        app.logtxt.insert("1.0", logdemo)
        app.parse_log()
        app.update()
        check("in-GUI log parse appended output",
              "Parsed OpenV2K event log" in app.logtxt.get("1.0", tk.END), "")
        app.destroy()

    npass = sum(1 for _, ok, _ in checks if ok)
    print("\n==== SELFTEST {}/{} PASSED ====".format(npass, len(checks)))
    for name, ok, detail in checks:
        if not ok:
            print("  FAILED: {} ({})".format(name, detail))
    print("\nfixtures + outputs in: {}".format(outdir))
    return 0 if npass == len(checks) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="openv2k_parser",
        description="OpenV2K IQ Parser {} - GUI/CLI parser and validator for "
                    "OpenV2K complex64 IQ recordings.".format(VERSION),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Analysis defaults match OpenV2K: fft 64, hop 32 (16 us/frame), "
               "threshold 5%% of peak, +/-1 ms padding, 3 s active cap.")
    ap.add_argument("file", nargs="?", help="IQ recording to open in the GUI")
    ap.add_argument("--fmt", default="complex64", choices=FORMAT_ORDER,
                    help="on-disk sample format (default: complex64)")
    ap.add_argument("--rate", type=float, default=OPENV2K_RATE,
                    help="sample rate in Sps (default: 2000000)")
    ap.add_argument("--wav", help="source WAV for the duration checksum")
    ap.add_argument("--src-ms", type=float,
                    help="manual source active duration in ms")
    ap.add_argument("--fft", type=int, default=DEFAULT_FFT)
    ap.add_argument("--hop", type=int, default=DEFAULT_HOP)
    ap.add_argument("--thresh", type=float, default=DEFAULT_THRESH_FACTOR,
                    help="active threshold factor of peak (default 0.05)")
    ap.add_argument("--min-width-us", type=float, default=1.0)
    ap.add_argument("--merge-gap-us", type=float, default=5.0)
    ap.add_argument("--max-active-s", type=float, default=DEFAULT_MAX_ACTIVE_S)
    ap.add_argument("--report", help="write a TXT report and exit (no GUI)")
    ap.add_argument("--json", help="write JSON metrics and exit (no GUI)")
    ap.add_argument("--pulses", help="write the pulse table CSV and exit "
                                     "(no GUI)")
    ap.add_argument("--generate-sample", help="write a synthetic OpenV2K-style "
                                              "IQ fixture to this path and exit")
    ap.add_argument("--generate-wav", help="with --generate-sample, also write a "
                                           "source WAV (2 ms longer than active)")
    ap.add_argument("--selftest", metavar="OUTDIR",
                    help="run the built-in verification suite into OUTDIR")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--version", action="version",
                    version="OpenV2K IQ Parser {}".format(VERSION))
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest_main(a.selftest, quiet=a.quiet)

    if a.generate_sample:
        meta = generate_sample_iq(a.generate_sample, rate=a.rate)
        print(json.dumps(meta, indent=2))
        if a.generate_wav:
            print(json.dumps(generate_sample_wav(
                a.generate_wav, meta["active_s"] * 1000.0 + 2.0), indent=2))
        return 0

    if a.file and (a.report or a.json or a.pulses):
        try:
            res = analyze_iq(
                a.file, fmt_key=a.fmt, rate=a.rate, fft_size=a.fft, hop=a.hop,
                thresh_factor=a.thresh, max_active_s=a.max_active_s,
                min_width_us=a.min_width_us, merge_gap_us=a.merge_gap_us,
                src_wav=a.wav, src_ms_manual=a.src_ms,
                progress=None if a.quiet else (lambda p, m: None))
        except (ValueError, OSError) as exc:
            sys.stderr.write("error: {}: {}\n".format(a.file, exc))
            return 2
        if a.report:
            with open(a.report, "w") as fh:
                fh.write(build_report(res))
            print("report  -> {}".format(a.report))
        if a.json:
            with open(a.json, "w") as fh:
                json.dump(res.metrics_dict(), fh, indent=2, default=float)
            print("metrics -> {}".format(a.json))
        if a.pulses:
            with open(a.pulses, "w") as fh:
                fh.write(pulses_csv(res))
            print("pulses  -> {}".format(a.pulses))
        if not (a.report or a.json or a.pulses):
            print(build_report(res))
        return 0

    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        sys.stderr.write(
            "WARNING: no DISPLAY - the Tk window cannot be shown. Use "
            "--report/--json/--pulses for headless use, or run under Xvfb.\n")
    app = ParserApp(initial_file=a.file, rate=a.rate, fmt_key=a.fmt)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
