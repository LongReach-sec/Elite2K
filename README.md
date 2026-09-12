[README.md](https://github.com/user-attachments/files/32137477/README.md)
# Elite2K

turn an
audio waveform into a train of short RF pulses whose density carries the
original information — and rebuilds it as a modern, modular instrument: a
pure-NumPy streaming DSP core, a 17-block reorderable chain, six output
back-ends, text-to-speech and microphone sources, portable JSON profiles, and a
dark-themed Tk console with a live spectrum, waterfall, scope and a
per-block parameter editor.

The entire project is **three dependencies deep — CPython, Tk and NumPy.**
Nothing else is required to run the DSP core, the head-less tests or the GUI.
SDR hardware support is *opportunistic*: if `hackrf_transfer`, GNU Radio or
SoapySDR exist, Elite2K uses them; if not, the simulated and IQ-file back-ends
keep every code path exercised.

```
+-----------+   +------------------+   +---------------------+   +-----------+
|  SOURCE   |-->|    DSP  CHAIN     |-->|   RESAMPLER  48k -> |-->|  SINK     |
| tone/mic/ |   | 17 blocks, any   |   |   IQ rate (linear,  |   | sim/iqfile|
| tts/noise |   | order, per-block |   |   phase-continuous) |   | hackrf/   |
| /silence  |   | params, bypass   |   |                     |   | gnuradio/ |
+-----------+   +------------------+   +---------------------+   | soapy/    |
                                                                 | audio     |
                                                                 +-----------+
```

---

## 1. Why it is different from OpenV2K160.py

| | OpenV2K160.py | Elite2K |
|---|---|---|
| Form | single monolithic script | package: `config` / `dsp` / `backend` / `gui` |
| Interface | console/CLI only | full Tk console *and* a scriptable CLI |
| Chain | fixed pipeline | 17 registered blocks, user-orderable, individually bypassable |
| Modulation | zero-crossing pulse | ZCP, PDM (sigma-delta), PWM, AM/DSB, FM, ASK, direct |
| Filtering | none | FIR design (HP/LP/BP), mains notch with harmonics, pre/de-emphasis |
| Dynamics | none | AGC, noise gate, envelope follower, spectral subtraction, decimator |
| Sources | audio in | tone, white noise, microphone, TTS (espeak-ng or built-in synth), silence, buffer |
| Sinks | soundcard | simulated, IQ file (+JSON sidecar), HackRF, GNU Radio/OsmoSDR, SoapySDR, audio |
| Configuration | hard-coded / prompted | 4 presets + portable JSON profiles with schema tag |
| Telemetry | text prints | 18 live metrics, log pane, spectrum/waterfall/scope |
| Testing | manual | 44 DSP self-tests, 105 GUI checks, extended DSP validator |

Everything is implemented from scratch in `elite2k/`; no code is copied from
upstream. `OpenV2K160.py` is only referenced as the conceptual ancestor and is
credited under *Acknowledgements*.

---

## 2. Requirements and installation

* Python **3.8+** (verified on CPython 3.14.6)
* NumPy (verified on 2.5.2)
* Tk 8.6 — part of the standard library on Windows/macOS; on Debian/Ubuntu
  install `python3-tk`

```bash
git clone <this repo> Elite2K      # or just unpack the tree
cd Elite2K
python3 -c "import numpy, tkinter; print('ok')"
python3 Elite2K.py --selftest      # 44 checks, no hardware needed
python3 Elite2K.py                 # the console
```

There is **no** `pip install` step. There is no build step. There are no
generated artefacts checked in.

Head-less hosts (CI, containers, SSH) need a virtual X server for the GUI:

```bash
sudo apt-get install -y xvfb
xvfb-run -a python3 Elite2K.py --guitest
```

---

## 3. Quick start

### Graphical console

```bash
python3 Elite2K.py                       # open the console
python3 Elite2K.py --simulate            # open and start transmitting at once
python3 Elite2K.py --preset "PDM experiment"
python3 Elite2K.py --profile myprofile.json
python3 Elite2K.py --freq 433920000
```

The window has three panes:

* **left** — RF configuration (frequency with band presets, IQ rate, TX
  amplitude, gains, ppm correction, antenna), source selection, TTS panel,
  output back-end and profile load/save;
* **centre** — live spectrum (1024-point Hann-windowed, FFT-shifted, dB),
  rolling waterfall (240 rows × 256 columns rendered straight into a
  `Tk.PhotoImage`), time-domain scope of the baseband drive (256 samples), and
  the metrics dashboard;
* **right** — the chain editor: one card per block with an enable checkbox,
  ▲/▼ reorder buttons and a slider plus entry box for every parameter.

The toolbar carries the preset drop-down, ▶ START, ■ STOP, ⏸ PAUSE and the
profile buttons. Everything runs on the Tk thread; the DSP engine runs on its
own worker thread and communicates through an immutable snapshot plus a
thread-safe log queue — the UI never touches engine internals.

### Command line

| Command | Purpose |
|---|---|
| `python3 Elite2K.py` | launch the graphical console |
| `python3 Elite2K.py --selftest` | head-less DSP self-test (44 checks), exit code = failures |
| `python3 Elite2K.py --guitest` | head-less GUI regression test (105 checks), needs a display |
| `python3 Elite2K.py --render 5 out.iq` | render 5 seconds of IQ offline and exit |
| `python3 Elite2K.py --list-blocks` | print the block library with parameters |
| `python3 Elite2K.py --list-presets` | print the built-in presets |
| `python3 Elite2K.py --dump-profile` | print the selected profile as JSON |
| `python3 Elite2K.py --preset NAME` | select a built-in preset |
| `python3 Elite2K.py --profile FILE` | load a saved profile |
| `python3 Elite2K.py --freq HZ` | override the centre frequency |
| `python3 Elite2K.py --simulate` | start the engine immediately on launch |
| `python3 Elite2K.py --guitest --keep` | run the GUI test and leave the window open |
| `python3 Elite2K.py --version` | print the version |

Head-less DSP use as a library:

```python
import elite2k

prof = elite2k.apply_preset("Maximum penetration")
prof.rf.frequency_hz = 433_920_000
prof.backend, prof.backend_target = "iqfile", "capture.cfile"

eng = elite2k.Engine(prof, on_log=print)
eng.start()
...                                    # let it run
eng.stop()
m = eng.snapshot()["metrics"]
print(m.iq_samples, m.pulses_session, m.throughput_ksps, m.error)
```

`import elite2k` pulls in **no** Tk, so batch conversions and embedded use work
on a bare interpreter.

---

## 4. The DSP chain

Blocks are registered in `elite2k/dsp/blocks.py` (`BLOCK_REGISTRY` /
`BLOCK_CLASSES`). Each one declares its label, category, `order_hint`, and a
`param_specs()` list that the GUI turns into controls automatically — adding a
block to the library makes it appear in the editor and in `--list-blocks` with
no GUI changes.

| Key | Block | Category | Parameters |
|---|---|---|---|
| `dcblock` | DC Blocker | filtering | `r` |
| `highpass` | High-Pass | filtering | `cutoff` |
| `lowpass` | Low-Pass | filtering | `cutoff` |
| `bandpass` | Band-Pass (F1) | filtering | `center`, `width` |
| `notch` | Mains Notch | filtering | `freq`, `q`, `harmonics` |
| `preemph` | Pre-emphasis | filtering | `a` |
| `deemph` | De-emphasis | filtering | `a` |
| `agc` | AGC | dynamics | `target`, `tc_ms`, `max_gain` |
| `noisegate` | Noise Gate | dynamics | `threshold_db`, `attack_ms`, `release_ms` |
| `envfollow` | Envelope Follower | dynamics | `tc_ms` |
| `spectralsub` | Spectral Subtract | dynamics | `reduction_db`, `alpha` |
| `decimator` | Decimator | dynamics | `cutoff`, `factor`, `hard` |
| `hwrect` | Half-Wave Rectifier | nonlinear | `floor` |
| `schmitt` | Schmitt Trigger | nonlinear | `hi`, `lo`, `level` |
| `hilbert` | Hilbert Envelope | nonlinear | `tc_env_ms`, `tc_mean_ms` |
| `zcp` | Zero-Cross Pulse | modulation | `pulse_width_us`, `threshold` |
| `modulator` | Modulator | modulation | `mode`, `carrier_hz`, `fm_dev_hz`, `pulse_width_us`, `depth` |

The classic OpenV2K-style signal path is
`hp → agc → noisegate → hwrect → schmitt → zcp → modulator`, i.e. a filtered
audio drive, an asymmetric half-wave rectification onto the *negative* rail, a
hysteretic Schmitt trigger that gives the pulse train its hysteresis margins,
zero-crossing pulse generation, then final modulation.

**Every block is streamable.** Processing a buffer in one call and in arbitrary
chunks must produce identical samples — the extended validator enforces that
for all 17 blocks at 2400-sample chunks, and the DSP self-test additionally
checks the chunk sizes 24000 / 2400 / 1200 / 997 / 256 / 100. This is the
property that makes real-time operation and offline rendering agree.

**`spectralsub` latency.** The overlap-add spectral subtractor is the only
block with intentional latency: a fixed 512-sample delay (10.7 ms at 48 kHz)
with `FRAME − HOP = 256` samples held back between calls. It uses a monotonic
accumulator with global frame indexing and an exact Hann² envelope
(`w²[:HOP] + w²[HOP:FRAME]`, period 256, minimum 0.5, never zero), so the tail
is always emitted — never truncated and zero-padded, which was the historical
defect that shifted the stream by up to one frame per call.

### Modulation modes

`zcp` (classic zero-crossing pulse), `pdm` (sigma-delta pulse density), `pwm`,
`am` (DSB), `fm`, `ask` (on/off keying) and `direct` (pass the upstream chain
straight to the resampler, used by the *Diagnostic* preset).

---

## 5. Sources

| Kind | Description |
|---|---|
| `tone` | sine generator, 20 Hz – 20 kHz, amplitude from the TX setting |
| `noise` | white noise, useful for measuring pulse density limits |
| `mic` | microphone capture at 48 kHz, falling back to silence if no device |
| `tts` | speech: `espeak-ng`/`espeak` when installed, otherwise the built-in formant synthesiser (19 voice codes) |
| `silence` | zero drive — calibrates the modulator with no input |
| buffer | any `float32` array handed to `Engine.set_tts_buffer()`, looped or one-shot |

`elite2k/backend/tts.py` exposes `synthesize(text, voice, speed, pitch) ->
(samples_48k_float32, engine_label)` plus `active_region_ms()` and
`backend_report()`, so the TTS subsystem is usable from scripts and testable
without a sound card.

---

## 6. Output back-ends

| Key | Label in the GUI | Requirements | Notes |
|---|---|---|---|
| `sim` | Simulated (headless, no RF) | — | computes and measures everything, transmits nothing |
| `iqfile` | IQ Capture to file | — | interleaved `int8` I/Q plus a JSON sidecar of the run parameters |
| `hackrf` | HackRF via `hackrf_transfer` | HackRF tools in `PATH` | pipes the stream to the CLI tool, no GNU Radio needed |
| `gnuradio` | GNU Radio / gr-osmosdr | GNU Radio + OsmoSDR Python bindings | real SDR transmit |
| `soapy` | SoapySDR | SoapySDR Python bindings | real SDR transmit |
| `audio` | Audio device | `sounddevice` / `soundcard` / `aplay` | lets you *hear* the drive waveform |

Availability is probed at start-up (`backend_availability()`), results are
logged in the console, and selecting an unavailable back-end degrades to a
simulated one instead of crashing. The self-test asserts this fallback
behaviour, so the GUI always starts even on a machine with no radio at all.

Rate conversion from the 48 kHz audio domain to the IQ rate (100 kS/s –
4 MS/s) is a phase-continuous fractional resampler: the step is derived
directly as `rate_in / rate_out`, so the whole range is reachable and the
default 2 MS/s resolves to the exact rational ratio 125/3. When down-sampling
it first applies a boxcar anti-alias average. The fractional sample position
and the previous sample are carried across chunks, so no click or drift
appears at chunk boundaries.

### Transmit legality

You are responsible for the frequency, power and antenna you select. The band
presets include ISM and amateur frequencies; transmitting on them is licensed
activity in most jurisdictions. Use the `sim` and `iqfile` back-ends — which
are the defaults in every shipped preset — for bench work and testing.

---

## 7. Presets and profiles

`python3 Elite2K.py --list-presets`:

| Preset | Enabled blocks | Intent |
|---|---|---|
| Balanced (recommended) | 10 | the everyday chain, defaults to `sim` |
| Maximum penetration | 15 | every useful gain and shaping stage engaged for weak links |
| Diagnostic (no modulation) | 2 | raw filtered drive, no pulse shaping — measure the front end |
| PDM experiment | 5 | sigma-delta pulse-density modulation instead of ZCP |

A profile is a plain JSON document (`elite2k.profile/1`) holding RF settings,
source settings, the chain `order`/`enabled`/`params`, and the back-end:

```json
{
  "schema": "elite2k.profile/1",
  "name": "capture-433",
  "rf": {"frequency_hz": 433920000.0, "iq_rate": 2000000, "tx_amplitude": 0.5},
  "source": {"kind": "tts", "tts_text": "elite two kay", "tts_voice": "en"},
  "chain": {"order": ["dcblock", "highpass", "agc"], "enabled": {"agc": true},
            "params": {"highpass": {"cutoff": 100.0}}},
  "backend": "iqfile",
  "backend_target": "capture.cfile",
  "realtime": false
}
```

Any block key missing from `order` is appended with library defaults when the
profile is loaded, so profiles written by older builds keep working. Writes are
atomic (`path.tmp` + `os.replace`).

---

## 8. Metrics

`Engine.snapshot()` returns the spectrum, scope, waterfall and an immutable
`Metrics` record; the dashboard renders 18 of these live:

`running`, `chunks`, `audio_rms`, `audio_peak`, `drive_rms`, `drive_duty`,
`zcr_hz`, `pulses_session`, `iq_rms`, `iq_peak`, `crest_db`,
`throughput_ksps`, `working_rate`, `iq_samples`, `elapsed_s`,
`energy_per_pulse_mj`, `peak_power_w`, `error`.

`zcr_hz` is the measured zero-crossing rate of the **source audio** — 880/s
for a 440 Hz tone, i.e. two sign changes per cycle — while `drive_rms` and
`drive_duty` describe the post-chain waveform (duty being the fraction of
samples above 0.5, which is what the Schmitt trigger keys on).
`energy_per_pulse_mj` is a *stipulated* per-pulse reference energy (16 mJ,
`config.PULSE_ENERGY_REFERENCE_MJ`) and `peak_power_w` is that energy divided
by the active pulse width (`zcp` or `modulator`, whichever is enabled). They
are chain-to-chain comparison figures, not a calibrated radiometric
measurement of your transmitter.

---

## 9. Testing

Three layers, none of which need SDR hardware:

```bash
python3 Elite2K.py --selftest            # 44 checks, ~10 s, exit code = failures
xvfb-run -a python3 Elite2K.py --guitest # 105 checks, ~10 s
python3 tools/validate_filters.py        # extended DSP validation, prints numbers
```

* **`--selftest`** covers the block library (streaming continuity for all 17
  blocks, DC-blocker exactness, FIR rounding), FIR frequency response
  (high-pass flat at 440 Hz and −48 dB at 50 Hz; low-pass −6 dB at the 2300 Hz
  cutoff and −120 dB at 4 kHz), chunk-size independence across six chunk sizes
  (max deviation 3.0e-07), the pulse chain (Schmitt levels ±0.5 with 44 edges,
  880 pulses/s for 440 Hz), resampler ratios, every back-end including IQ-file
  capture and fallback, and all four presets.
* **`--guitest`** builds the real window under a virtual display and drives it:
  widget tree (434 widgets), transport state machine, engine warm-up with a
  bounded wait, spectrum/scope/waterfall/dashboard feed, all four presets
  loaded through the toolbar code path with the chain verified against each
  preset, chain enable/reorder/parameter edits propagated to the engine,
  every source kind and back-end key applied, JSON profile save/load
  round-trip, and a TTS synthesis feeding the engine. It exits non-zero on any
  failure, so it drops straight into CI.
* **`tools/validate_filters.py`** is the slow companion that prints actual
  numbers rather than pass/fail: FIR magnitude responses in dB, per-block
  streaming error magnitudes, the pulse chain rails, and the SpectralSubtractor
  whole-vs-chunked alignment grid showing a shift-0 difference of exactly 0.0
  against 6e-01…9.6e-01 at every other shift.

Recorded evidence snapshots from the current build live in `docs/`
(`selftest_report.txt`, `guitest_report.txt`, `validate_report.txt`).

---

## 10. Project layout

```
Elite2K.py                    launcher / CLI (all sub-commands)
README.md                     this file
LICENSE                       Unlicense (public domain)
elite2k/
  __init__.py                 public surface; imports no Tk
  config.py                   themes, presets, band plan, backend/mode tables, profile dataclasses
  selftest.py                 head-less DSP self-test (44 checks)
  guitest.py                  head-less Tk regression harness (105 checks)
  dsp/
    blocks.py                 block base class, 17 blocks, FIR designers, registry
    engine.py                 worker thread, chain execution, resampling, metrics, visuals
  backend/
    sources.py                tone / noise / microphone / silence / buffer sources
    sinks.py                  simulated, IQ file, HackRF, GNU Radio, SoapySDR, audio sinks
    tts.py                    espeak-ng wrapper + built-in formant synthesiser
  gui/
    app.py                    the Tk console (spectrum, waterfall, scope, chain editor)
tools/
  validate_filters.py         extended DSP validation with numeric output
docs/                         recorded test evidence
```

Data flow: `Tk thread` → profile dataclasses → `Engine` worker → source
→ chain → resampler → sink → `snapshot()` → Tk thread repaints. No shared
mutable state crosses the thread boundary.

### Adding a block

```python
from elite2k.dsp.blocks import Block, P

class MyBlock(Block):
    key, label, category, order_hint = "myblock", "My Block", "dynamics", 45

    @classmethod
    def param_specs(cls):
        # P(name, label, lo, hi, default, step, fmt) -> dict for the GUI
        return [P("gain", "Gain", 0.0, 4.0, 1.0, 0.01, "{:.2f}")]

    def on_param(self, name, value):     # recompute coefficients here
        ...

    def reset(self):                     # clear streaming state on (re)start
        ...

    def process(self, x):                # must be chunk-size independent
        ...
```

Register it in `BLOCK_REGISTRY` / `BLOCK_CLASSES`. It immediately appears in
the chain editor, in `--list-blocks`, in profiles, and is picked up by the
continuity test in `tools/validate_filters.py` — which is exactly how the 17
shipped blocks are verified. Rate-changing blocks additionally expose a
`rate_factor()` so the engine can track the working rate through the chain.

---

## 11. Version

`Elite2K 2.0.0` (build stamp `2026/09/12`), profile schema `elite2k.profile/1`.

## Acknowledgements

The pulse-transmitter concept, the zero-crossing pulse idea and the name
lineage come from the OpenV2K project: <https://github.com/OpenV2K>. Elite2K is
an independent reimplementation; upstream code is not reused.

## License

Released into the public domain under the [Unlicense](LICENSE). No warranty —
RF transmission is regulated, and you are responsible for complying with the
rules that apply where you operate.
