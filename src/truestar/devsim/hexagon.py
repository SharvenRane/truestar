"""Qualcomm Hexagon NPU (HTP) GELU, fitted to real Snapdragon outputs.

Measured with the Qualcomm AI Hub battery (docs/QAIHUB.md, battery v1, seed 20261124) on a Galaxy
S26 (Hexagon v81) through TFLite and QNN DLC, and on a Snapdragon X Elite (v73) through QNN DLC and
ONNX Runtime QNN; the four runs agree on 99.98% of the GELU outputs. Fit and held out score:
``scripts/hexagon_sim_fit.py fit``; numbers in docs/DEVICES.md section 7.

What the data says (not a vendor description; Qualcomm does not document the kernel):

* GELU is fused into the linear layer before it: its input is the unrounded accumulator, not the
  float16 output of the linear layer. Fed the float16 rounded input instead, the same curve matches
  72.5% of the held out outputs bit for bit; fed the accumulator, 89.4%.
* the output is ``x * phi(x)`` rounded once to float16, where ``phi`` is a piecewise approximation of
  the normal CDF: a line for |x| < 0.6 (slope 0.387 where the true slope is 0.399), a smooth curve
  above, then clamped: every input at or below -3.03 gives 0 and every input at or above 3.04
  gives x. It is not the tanh or sigmoid form of GELU, and it is not symmetric
  (``phi(-x) != 1 - phi(x)``), so each sign has its own fit.
* the error against exact GELU peaks at about 4e-3 near |x| = 0.6, the break of the approximation.

The polynomials below are the fit (numpy ``polyval`` order, highest power first, variable |x|).
"""

from __future__ import annotations

import numpy as np
import torch

BREAK = 0.6
# measured range of the fit; outside it phi is the measured clamp (0 below, 1 above)
FIT_LIMIT = 3.0

NEG_LINE = (-0.3871219739045472, 0.5002750483116811)
NEG_CURVE = (4.9911250206989861e-06, -3.9719506171150554e-03, 8.1007077230326793e-03,
             1.0668622003656532e-01, -4.5269988368256070e-01, 5.0020638487324875e-01)
POS_LINE = (0.3870892145940244, 0.5002768892590844)
POS_CURVE = (1.1067478859003914e-04, 2.9576803050543905e-03, -4.8354888044605466e-03,
             -1.1145337223745046e-01, 4.5582540660908133e-01, 4.9959783506771277e-01)


def _root(curve, target: float) -> float:
    """First |x| at or above FIT_LIMIT where the curve reaches ``target`` (the clamp point)."""
    c = np.array(curve, dtype=np.float64)
    c[-1] -= target
    real = [r.real for r in np.roots(c) if abs(r.imag) < 1e-9 and FIT_LIMIT - 0.5 <= r.real <= 4.0]
    return min(real) if real else FIT_LIMIT


NEG_ZERO = _root(NEG_CURVE, 0.0)
POS_ONE = _root(POS_CURVE, 1.0)


def _poly(c, a: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(a)
    for k in c:
        out = out * a + k
    return out


def phi(x: torch.Tensor) -> torch.Tensor:
    """The device's normal CDF approximation, float64 in and out."""
    a = x.abs()
    neg = torch.where(a < BREAK, _poly(NEG_LINE, a), _poly(NEG_CURVE, a))
    neg = torch.where(a >= NEG_ZERO, torch.zeros_like(a), neg)
    pos = torch.where(a < BREAK, _poly(POS_LINE, a), _poly(POS_CURVE, a))
    pos = torch.where(a >= POS_ONE, torch.ones_like(a), pos)
    return torch.where(x < 0, neg, pos).clamp(0.0, 1.0)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Hexagon GELU of float64 ``x`` (the caller rounds the result to float16). NaN stays NaN; minus
    infinity gives 0, plus infinity gives infinity."""
    p = phi(torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4))
    out = torch.where(p == 0, torch.zeros_like(x), x * p)
    out = torch.where(torch.isposinf(x), x, out)
    return torch.where(torch.isnan(x), x, out)
