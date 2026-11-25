# Near tie diagnosis

Script: [examples/cookbook/near_tie.py](https://github.com/SharvenRane/truestar/blob/main/examples/cookbook/near_tie.py)

Any run in lower precision changes some predictions. Where a row's float64 top two logits are
closer than the rounding a correct run in that precision makes on that row, either answer is
correct and the flip is a near tie. A flip with a wider margin is real: something computed the
wrong thing. Truestar reports the two apart everywhere it counts decisions: `truestar check`
(TS013, "3/8 predictions changed (0 near ties, 3 real)"), `conform` (TS624), quantization (TS462)
and the gradient check (TS188).

The script does it by hand on a native classifier, with the `decisions` helper from
`truestar.quant.model`:

```python
from truestar.devsim import simulate
from truestar.quant.model import decisions

with simulate("sim:ieee-bf16"):                     # a correct bfloat16 run
    bf16 = net(p32, x).double()
tie = 4 * (bf16 - ref).abs().amax(-1)               # rounding bound per row, times 4
with simulate("sim:neuron-bf16"):                   # the device under test: matmuls in bfloat16
    dev = net(p32, x).double()
d = decisions(ref, dev, tie)                        # d.flips, d.near_ties, d.real
```

Output:

```
sim:neuron-bf16: 9 of 4000 predictions changed (9 near ties, 0 real)
  row  247: float64 class 12 by 6.34e-04 over class 9; this run picked 9; rounding bound on the row 2.53e-03 -> near tie
  row  661: float64 class 13 by 4.09e-04 over class 9; this run picked 9; rounding bound on the row 2.17e-03 -> near tie
  row  795: float64 class 18 by 3.18e-04 over class 9; this run picked 9; rounding bound on the row 2.93e-03 -> near tie

bfloat16 with the planted bug: 29 of 4000 predictions changed (24 near ties, 5 real)
  row  107: float64 class 3 by 4.71e-03 over class 6; this run picked 6; rounding bound on the row 3.71e-03 -> REAL
  row  247: float64 class 12 by 6.34e-04 over class 9; this run picked 9; rounding bound on the row 2.53e-03 -> near tie
  row  372: float64 class 19 by 1.87e-04 over class 9; this run picked 9; rounding bound on the row 2.33e-03 -> near tie
```

The planted bug negates class 3's weights in the last layer. Its 24 near ties are the same rows
any bfloat16 run may flip; the 5 real flips are rows the bug moved by more than rounding can, and
row 107 is the place to start looking.

## Limits

* The bound is 4 times the error of one correct run on that row, the rule
  `truestar.quant.model.compare` uses; `truestar check` uses twice its module threshold times the
  output scale. A row whose margin sits right at the bound can go either way.
* Top 1 decisions only (classifier and language model heads).
