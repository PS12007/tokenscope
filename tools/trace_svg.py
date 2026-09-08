#!/usr/bin/env python3
"""Render one decode token from a level-3 trace as a standalone SVG.

The README has wanted a picture since session 1 and the plan was always a
Perfetto screenshot, which needs a browser, a person and a PNG nobody can
regenerate. This produces the same picture from the committed trace, in a file
that is text, diffable, theme-aware and reproducible by anyone who clones the
repo:

    python tools/trace_svg.py examples/qwen3-8b-named-attnout.trace.json \\
        -o docs/token.svg

What it draws: one row per worker thread, time left to right, every node scope
as a filled span and every barrier wait as a recessive one. The shape of a
token -- ffn dominating, attention next, and the ragged grey edge where threads
wait for each other -- is the whole argument of FINDINGS F6 and F9 in one
image.

Colour follows the data-viz rules rather than taste. Four fills, not nine:
three categorical hues (validated all-pairs for colour-vision deficiency in
both light and dark) plus one neutral for barrier wait, because barrier wait is
the absence of work and should not read as a fourth kind of work. Phases beyond
the three fold into "other", which is what the palette rules say to do rather
than inventing a fourth hue.

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from trace_analyze import Trace, HOST_CATS, _node_base  # noqa: E402

# --- palette -----------------------------------------------------------------
# Categorical slots 1-3 of the reference palette, in fixed order. Validated
# with scripts/validate_palette.js under --pairs all in both modes: worst CVD
# dE 9.2 light / 9.4 dark, worst normal-vision dE 24.0 light / 20.9 dark.
# Aqua is below 3:1 on the light surface, so the legend carries text labels --
# identity is never colour alone here.
LIGHT = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
    "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
    "ffn": "#2a78d6", "attn": "#eb6834", "other": "#1baf7a",
    "barrier": "#e1e0d9", "idle": "#f0efec",
}
DARK = {
    "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
    "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
    "ffn": "#3987e5", "attn": "#d95926", "other": "#199e70",
    "barrier": "#2c2c2a", "idle": "#232322",
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def bucket(cat: str) -> str:
    """Nine trace categories into three fills plus barrier."""
    if cat == "barrier":
        return "barrier"
    if cat.startswith("ffn"):
        return "ffn"
    if cat.startswith("attn") or cat.startswith("~attn"):
        return "attn"
    return "other"


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def pick_token(tr: Trace, want: int | None):
    """The token to draw: the requested one, else the first captured decode
    token that actually has node scopes on more than one thread."""
    cands = tr.decode or tr.prefill
    if not cands:
        sys.exit("no token slices in this trace")
    for t in cands:
        tok = t["args"]["tok"]
        if want is not None and tok != want:
            continue
        evs = [e for e in tr.by_token.get(tok, [])
               if e.get("cat") not in HOST_CATS and e.get("tid", -1) >= 0]
        if len({e["tid"] for e in evs}) > 1:
            return t, evs
    if want is not None:
        sys.exit("token %d has no multi-thread node scopes in this trace" % want)
    sys.exit("no token in this trace has node scopes on more than one thread\n"
             "(needs TOKENSCOPE_LEVEL=3)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace")
    ap.add_argument("-o", "--out", default=None,
                    help="output .svg (default: alongside the trace)")
    ap.add_argument("--token", type=int, default=None,
                    help="token ordinal to draw (default: first usable)")
    ap.add_argument("--width", type=int, default=1180)
    ap.add_argument("--row", type=int, default=22, help="row height in px")
    ap.add_argument("--labels", type=int, default=6,
                    help="direct-label the N widest spans (0 to disable)")
    args = ap.parse_args()

    tr = Trace(args.trace)
    slice_, evs = pick_token(tr, args.token)
    tok = slice_["args"]["tok"]

    tids = sorted({e["tid"] for e in evs})
    t0 = min(e["ts"] for e in evs)
    t1 = max(e["ts"] + e.get("dur", 0.0) for e in evs)
    span_us = max(t1 - t0, 1e-6)

    # --- geometry ------------------------------------------------------------
    W = args.width
    gutter, right = 74, 14
    plot_w = W - gutter - right
    head_h, legend_h, axis_h, pad = 46, 26, 30, 10
    row_h, row_gap = args.row, 3
    body_h = len(tids) * (row_h + row_gap) - row_gap
    H = head_h + legend_h + body_h + axis_h + pad

    def x(us: float) -> float:
        return gutter + (us - t0) / span_us * plot_w

    o = []
    a = o.append
    a('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" '
      'width="%d" height="%d" font-family=\'%s\' role="img" '
      'aria-label="Per-thread node timeline for one decode token">'
      % (W, H, W, H, FONT))

    # Theme: light values on :root, dark redefined under prefers-color-scheme.
    # An SVG loaded through <img> gets its own rendering context and still sees
    # the viewer's colour scheme, so one file serves both.
    def vars_block(d):
        return "".join("--%s:%s;" % (k, v) for k, v in d.items())
    a('<style>\n'
      ':root{%s}\n'
      '@media (prefers-color-scheme: dark){:root{%s}}\n'
      '.s{fill:var(--surface)}.ink{fill:var(--ink)}.ink2{fill:var(--ink2)}\n'
      '.mut{fill:var(--muted)}.grid{stroke:var(--grid)}.ax{stroke:var(--axis)}\n'
      '.ffn{fill:var(--ffn)}.attn{fill:var(--attn)}.other{fill:var(--other)}\n'
      '.barrier{fill:var(--barrier)}.idle{fill:var(--idle)}\n'
      '</style>' % (vars_block(LIGHT), vars_block(DARK)))

    a('<rect width="%d" height="%d" class="s"/>' % (W, H))

    # --- header --------------------------------------------------------------
    thr = tr.meta.get("threads", len(tids))
    a('<text x="%d" y="20" class="ink" font-size="14" font-weight="600">'
      'One decode token, %d worker threads</text>' % (gutter, len(tids)))
    sub = ("%s  ·  token %d  ·  %.2f ms  ·  %d node and barrier scopes  ·  "
           "threading=%s" % (esc(os.path.basename(tr.path)), tok,
                             span_us / 1000.0, len(evs),
                             esc(str(tr.meta.get("threading", "unrecorded")))))
    a('<text x="%d" y="36" class="mut" font-size="10.5">%s</text>' % (gutter, sub))

    # --- legend --------------------------------------------------------------
    ly = head_h + 12
    lx = gutter
    for cls, label in (("ffn", "feed-forward"), ("attn", "attention"),
                       ("other", "norm / residual / output"),
                       ("barrier", "barrier wait")):
        a('<rect x="%.1f" y="%.1f" width="10" height="10" rx="2" class="%s"/>'
          % (lx, ly - 8, cls))
        a('<text x="%.1f" y="%.1f" class="ink2" font-size="10.5">%s</text>'
          % (lx + 14, ly + 1, label))
        lx += 14 + 7.0 * len(label) + 22

    # --- rows ----------------------------------------------------------------
    top = head_h + legend_h
    by_tid = {}
    for e in evs:
        by_tid.setdefault(e["tid"], []).append(e)

    labelled = []
    for i, tid in enumerate(tids):
        y = top + i * (row_h + row_gap)
        a('<rect x="%d" y="%.1f" width="%.1f" height="%d" rx="2" class="idle"/>'
          % (gutter, y, plot_w, row_h))
        a('<text x="%d" y="%.1f" class="mut" font-size="10" text-anchor="end">'
          'worker %d</text>' % (gutter - 8, y + row_h / 2 + 3.5, tid))

        for e in sorted(by_tid.get(tid, []), key=lambda e: e["ts"]):
            dur = e.get("dur", 0.0)
            x0, x1 = x(e["ts"]), x(e["ts"] + dur)
            w = x1 - x0
            cls = bucket(e.get("cat", "other"))
            # A 2px surface gap between adjacent fills is the mark spec, but at
            # this density most spans are under 2px wide. Inset only where
            # there is room, so separation appears exactly where it can be
            # seen and thin spans stay visible instead of disappearing.
            if w > 2.5:
                x0 += 0.5
                w -= 1.0
            w = max(w, 0.4)
            a('<rect x="%.2f" y="%.1f" width="%.2f" height="%d" rx="%s" '
              'class="%s"/>' % (x0, y, w, row_h, "1.5" if w > 4 else "0", cls))
            if cls != "barrier":
                labelled.append((w, x0, y, _node_base(e.get("name", "")), dur))

    # --- direct labels on the widest spans ----------------------------------
    # The light-mode aqua sits below 3:1 on the surface, so the relief rule
    # applies: the legend is text, and the biggest spans say what they are.
    if args.labels:
        seen = set()
        n = 0
        for w, x0, y, name, dur in sorted(labelled, reverse=True):
            if n >= args.labels or w < 34:
                break
            if name in seen:
                continue
            seen.add(name)
            n += 1
            a('<text x="%.1f" y="%.1f" class="ink" font-size="9.5" '
              'font-weight="600">%s</text>'
              % (x0 + 4, y + row_h / 2 + 3.5, esc(name)))

    # --- axis ----------------------------------------------------------------
    ay = top + body_h + 16
    a('<line x1="%d" y1="%.1f" x2="%.1f" y2="%.1f" class="ax" '
      'stroke-width="1"/>' % (gutter, ay - 8, gutter + plot_w, ay - 8))
    ms = span_us / 1000.0
    step = 1.0
    while ms / step > 10:
        step *= 2 if str(step)[0] == "1" else 2.5
    while ms / step < 4:
        step /= 2
    t = 0.0
    while t <= ms + 1e-9:
        px = gutter + (t / ms) * plot_w
        a('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="ax" '
          'stroke-width="1"/>' % (px, ay - 8, px, ay - 4))
        a('<text x="%.1f" y="%.1f" class="mut" font-size="9.5" '
          'text-anchor="middle">%g ms</text>' % (px, ay + 6, round(t, 4)))
        t += step

    a('</svg>')

    out = args.out or os.path.splitext(args.trace)[0] + ".svg"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(o) + "\n")
    print("wrote %s (%d bytes, %d threads, %d spans, token %d, %.2f ms)"
          % (out, os.path.getsize(out), len(tids), len(evs), tok,
             span_us / 1000.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
