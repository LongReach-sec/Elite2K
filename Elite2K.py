#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Unlicense
"""
Elite2K -- launcher.

A from-scratch, dependency-light successor to the OpenV2K zero-crossing
pulse RF transmitter family (https://github.com/OpenV2K).

Usage
-----
    python3 Elite2K.py                    # graphical console (default)
    python3 Elite2K.py --selftest         # head-less DSP self-test
    python3 Elite2K.py --guitest          # head-less GUI regression test
    python3 Elite2K.py --render 5 out.iq  # offline IQ render, 5 seconds
    python3 Elite2K.py --preset "PDM experiment"
    python3 Elite2K.py --profile my.json --simulate
    python3 Elite2K.py --list-presets
    python3 Elite2K.py --list-blocks
    python3 Elite2K.py --dump-profile       # print the selected profile as JSON

The GUI needs only the Python standard library (Tk) and NumPy.  SDR
hardware back-ends are used opportunistically when their tools/SDKs exist.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import elite2k                                   # noqa: E402
from elite2k import config                       # noqa: E402


# ---------------------------------------------------------------------------
#  Sub-commands
# ---------------------------------------------------------------------------

def _cmd_selftest(_args) -> int:
    from elite2k.selftest import run
    return run()


def _cmd_list_presets(_args) -> int:
    print("Elite2K presets:")
    for name in config.PRESETS:
        data = config.PRESETS[name]
        ch = data.get("chain") or {}
        enabled = sum(1 for v in (ch.get("enabled") or {}).values() if v)
        print("  %-28s backend=%-9s blocks=%d" % (name, data.get("backend", "sim"),
                                                   enabled))
    return 0


def _cmd_list_blocks(_args) -> int:
    from elite2k.dsp import blocks as block_mod
    print("Elite2K DSP block library (%d blocks):" % len(block_mod.BLOCK_CLASSES))
    for cls in sorted(block_mod.BLOCK_CLASSES, key=lambda c: c.order_hint):
        params = ", ".join(p["name"] for p in cls.param_specs()) or "-"
        print("  %-16s %-22s %-12s params: %s"
              % (cls.key, cls.label, cls.category, params))
    return 0


def _cmd_dump_profile(args) -> int:
    prof = config.apply_preset(args.preset or "Balanced (recommended)")
    prof.name = args.name or "default"
    text = prof.to_dict()
    import json
    print(json.dumps(text, indent=2, sort_keys=True))
    return 0


def _cmd_render(args) -> int:
    """Render IQ offline with the simulated pipeline, writing to a file."""
    import time
    prof = config.apply_preset(args.preset or "Balanced (recommended)")
    prof.backend = "iqfile"
    prof.backend_target = args.output or "elite2k_render.iq"
    if args.freq:
        prof.rf.frequency_hz = args.freq
    eng = elite2k.Engine(prof, on_log=lambda m: print("  " + m))
    print("rendering %.1f s of IQ -> %s" % (args.duration, prof.backend_target))
    eng.start()
    t0 = time.time()
    try:
        while time.time() - t0 < args.duration:
            time.sleep(0.1)
            if not eng.running:
                break
    finally:
        eng.stop()
    snap = eng.snapshot()
    m = snap["metrics"]
    print("done: %d IQ samples, %d pulses, %.1f ksps, error=%s"
          % (m.iq_samples, m.pulses_session, m.throughput_ksps, m.error or "none"))
    return 0 if not m.error else 1


def _cmd_guitest(args) -> int:
    """Drive the Tk console without a human (run under xvfb-run if head-less)."""
    from elite2k.guitest import run as guitest_run
    argv = ["--timeout", str(args.gui_timeout)]
    if args.quiet:
        argv.append("--quiet")
    if args.keep:
        argv.append("--keep")
    return guitest_run(argv)


def _cmd_gui(args) -> int:
    for stream, name in ((sys.stdout, "stdout"), (sys.stderr, "stderr")):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    from elite2k.gui.app import main as gui_main
    argv = []
    if args.profile:
        argv += ["--profile", args.profile]
    if args.preset:
        argv += ["--preset", args.preset]
    if args.simulate:
        argv += ["--simulate"]
    return gui_main(argv)


# ---------------------------------------------------------------------------
#  Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="Elite2K",
        description="Elite2K -- modular zero-crossing pulse RF transmitter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Successor to OpenV2K (https://github.com/OpenV2K).")
    p.add_argument("--version", action="version",
                   version="Elite2K %s" % config.VERSION)
    p.add_argument("--selftest", action="store_true",
                   help="run the head-less DSP self-test and exit")
    p.add_argument("--render", nargs="?", const=5.0, type=float, default=None,
                   metavar="SECONDS",
                   help="offline IQ render for SECONDS (default 5) and exit")
    p.add_argument("output", nargs="?", default=None,
                   help="output IQ path for --render")
    p.add_argument("--preset", default=None, help="load a built-in preset by name")
    p.add_argument("--profile", default=None, help="load a profile JSON file")
    p.add_argument("--freq", type=float, default=None, help="centre frequency (Hz)")
    p.add_argument("--simulate", action="store_true",
                   help="start the engine immediately on launch")
    p.add_argument("--guitest", action="store_true",
                   help="run the head-less Tk GUI regression test and exit "
                        "(requires a display; use xvfb-run when head-less)")
    p.add_argument("--gui-timeout", type=float, default=20.0,
                   metavar="SECONDS",
                   help="engine warm-up budget for --guitest (default 20)")
    p.add_argument("--quiet", action="store_true",
                   help="with --guitest, print only the summary")
    p.add_argument("--keep", action="store_true",
                   help="with --guitest, leave the window open for inspection")
    p.add_argument("--list-presets", action="store_true")
    p.add_argument("--list-blocks", action="store_true")
    p.add_argument("--dump-profile", action="store_true",
                   help="print the selected profile as JSON and exit")
    p.add_argument("--name", default=None, help="profile name for --dump-profile")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.selftest:
        return _cmd_selftest(args)
    if args.guitest:
        return _cmd_guitest(args)
    if args.list_presets:
        return _cmd_list_presets(args)
    if args.list_blocks:
        return _cmd_list_blocks(args)
    if args.dump_profile:
        return _cmd_dump_profile(args)
    if args.render is not None:
        args.duration = args.render
        return _cmd_render(args)
    return _cmd_gui(args)


if __name__ == "__main__":        # pragma: no cover
    raise SystemExit(main())