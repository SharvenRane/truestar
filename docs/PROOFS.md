# Machine checked proofs (Lean 4)

Truestar's exactness and determinism claims rest on a few small theorems. They are machine checked
in Lean 4 (no `sorry`; the axioms used are Lean's standard `propext`, `Classical.choice` and
`Quot.sound`, no `native_decide`).

* `proofs/lean/`: Lean core only (no mathlib). T1, T2, T3, T6, T8 (uniqueness half only), T9, the instantiation at the
  Python constants, and the cross check executable.
* `proofs/lean-analysis/`: needs real numbers, so it uses mathlib (pinned to commit
  `fc4f59e577feb1a4956671bc04ac18aaaf8cd842`, the `v4.36.0` tag). T4 and T7 (T7 in real arithmetic; its
  rounding bound is not done). T5 is not done.

Both pin `leanprover/lean4:v4.36.0`.

## Build

```
cd proofs/lean && lake build && lake exe tccheck fixtures/tc_probes.json
cd proofs/lean-analysis && lake exe cache get && lake build
```

`tests/test_lean_constants.py` checks the generated constants file on every run; with `lake` on
PATH it also builds, runs the cross check, and checks that weakened constants fail the build.

## The constants link: the build fails if a constant is weakened

`scripts/gen_lean_constants.py` reads the constants of `src/truestar/kernel/tcexact.py` (with
`ast`, no import) and writes `proofs/lean/TruestarProofs/Constants.lean`.
`TruestarProofs/Instances.lean` instantiates the theorems at those values and discharges every
side condition with `decide`, so `lake build` fails when:

| change in tcexact.py | why the build fails |
|---|---|
| `TC_FRAC_BITS_MIN` below 23 | T1 needs a grid no coarser than one integer at `E = 23` (`frac_bits_ok`) |
| `TC_FRAC_BITS_MIN` below 24 | guard on the measured value (`frac_bits_measured`); see the note below |
| `_RN` (the accumulator start C) with `C - 2,605,056 < 2^23` or `C + 2,605,056 >= 2^24`, or not `1.5 * 2^23` | `acc_window_ok`, `acc_is_rn_constant` |
| `MAX_KB` above 128, `PORTABLE_BK` above `MAX_KB` | `block_consts_ok` (T3's block sum bound is for 128 columns) |
| `TOP_MAX` not 127, `LOW_MAX` not 64, `DIGIT_BITS` not 7, `DIGITS` not 3 | `digit_consts_ok` |
| `SUM_BITS` such that `2,605,056 >= 2^SUM_BITS` | `sum_bits_ok` |

`tests/test_lean_constants.py` fails if the generated file is stale or differs from the imported
module's values.

Note on the window: the proof itself holds from 23 fraction bits (at `E = 23` the grid is
`2^0 = 1`, so integers lose nothing; `tcexact.py`'s docstring says the same for sm_70). 24 is the
narrowest window measured on the accepted generations (sm_80, sm_86, sm_89), and the separate
`frac_bits_measured` guard keeps the constant at that value. The cross check confirms the margin:
with every unit's window set to 23, the fp16 engine digit chains stay exact while the cross chip
probe vectors change.

## T1: exact tensor core window (`TCWindow.lean`, `Instances.lean`)

**Plain words.** The fp16 engine of portable tensor cores starts every float32 accumulator at
`C = 1.5 * 2^23` and adds integer digit products. Whatever the hardware does inside an MMA
instruction (how many products it fuses, in what order, whether `c` comes first or last), every
partial sum stays in `[2^23, 2^24)`, no bit is dropped, and the result is exactly `C + I_w`, the
exact integer sum.

**Hardware model (the assumptions).** One fused group takes the exact, unnormalized products and
the accumulator `c`, reads an exponent for each nonzero term, takes the largest (`E`, raised to an
exponent floor if the unit has one), cuts every term to the grid `2^(E - F)` toward zero or toward
minus infinity, adds the cut terms exactly, and rounds the sum once. Groups chain: the rounded
result is the next group's `c`. This is the model of `src/truestar/devsim/tensorcore.py` and
docs/TENSOR_CORE_MODELS.md (Khattak and Mikaitis), fitted bit for bit on sm_89 and sm_120. The
theorem assumes, of the hardware:

1. it follows this model with some group size, cut mode, exponent floor at most `2^23` and window
   `F >= 23` (`F` is 24 on sm_80/86/89 and 25 on sm_90/120, measured);
2. the exponent it reads for a nonzero term is never above the term's true exponent (true for
   normalized `c` and for products of normal float16 operands, which the integer digits are);
3. the final rounding returns every integer of magnitude below `2^24` unchanged (true of
   truncation and of round to nearest even into float32).

Group size, instruction shape, order of products and grouping are not assumed: `chain_exact_perm`
holds for every split of every permutation of the products into groups.

**Statements.** `group_exact` (one group is exact when all terms and the sum are integers below
`2^24`); `chain_exact` (the chain from an integer `a` with `2^23 <= a - R` and `a + R < 2^24`,
`R` the absolute sum of the products, returns `a + sum`); `chain_exact_perm` (any grouping and
order); `chain_window` (every intermediate accumulator is `C` plus the partial sum, inside
`[2^23, 2^24)`); `Inst.fp16_engine_exact` (all of it at the constants of tcexact.py, for every
digit weight `w` and every K block of at most `MAX_KB` columns whose digits satisfy T3).

## T2: determinism under a fixed reduction order (`Determinism.lean`)

**Plain words.** Fix a reduction tree. Any schedule that evaluates each node after its two
children (any warp count, pipeline depth, SM count, split K placement) returns the tree's value:
the result depends on the inputs only.

**Model.** The operation is arbitrary (no associativity or commutativity assumed; float32
addition with any rounding qualifies). A trace is a list of node positions; each step reads the
children's stored values and stores the operation's result; steps whose children are missing do
nothing and repeated steps are allowed. **Statements.** `run_sound` (any stored value is the
subtree's value, for any trace), `run_complete` (a valid trace containing the root stores it),
`determinism` (every valid trace returns `eval T x`), `schedule_independent`.

**Assumption.** Each step is atomic and reads values that were stored (no torn reads, no data
races): the kernel's synchronization provides that, it is not modelled.

## T3: the digit split (`Digits.lean`)

**Plain words.** Cutting `t = a * 2^(7 - ea)` into three digits `d0 + d1 2^-7 + d2 2^-14` with
round to nearest even loses at most `2^-15` (in units of `t`), nothing when `t` is a multiple of
`2^-14`, and every digit fits int8 (`|d0| <= 127`, `|d1|, |d2| <= 64`); the integer sums of a K
block of up to 128 columns stay at most 2,605,056 `< 2^22`.

**Model.** Every finite float32 is a multiple of `2^-149`, so `t = T / 2^149` with `T` an integer
and every step is integer arithmetic. **Statements.** `digits_reconstruct` (the identity with the
remainder bound), `digits_exact_of_fits`, `digits_bounds` and `digits_int8` (for `|t| < 127.5`,
which the block scale guarantees: `|t| < 127.01`), `rn_add_C` (the magic constant: adding
`1.5 * 2^23` commutes with `rn`, and in `[2^23, 2^24)` float32 round to nearest even is `rn`),
`magic_decode` (the bits `0x4B400000 + i` decode to `C + i` for `|i| < 2^22`),
`block_sum_bound`.

**Assumed, not proved.** That the float32 operations of the kernel (`t - d0`, the multiplication
by 128, the block scaling by a power of two inside the normal range) are exact; they are
(Sterbenz style subtraction, power of two scaling) and `tests/test_tcexact.py` checks the float32
code against the integer results.

## T4: tree error bound (`lean-analysis/TreeBound.lean`)

**Plain words.** A sum over any reduction tree, every addition rounded with relative error at
most `u`, is within `((1 + u)^depth - 1) * sum |x|` of the exact sum; that factor is at most
`gamma_depth = depth u / (1 - depth u)`.

**Model.** The standard model `fl(a + b) = (a + b)(1 + delta)`, `|delta| <= u`, with `delta`
chosen arbitrarily per node (any rounding mode with that bound; float32 nearest has
`u = 2^-24`). **Assumed:** no underflow or overflow (the standard model's range). Statements:
`tree_bound`, `tree_bound_root`, `pow_sub_one_le_gamma`.

## T5: running error bound soundness: not done

The rule (ranked item 1 of LUMEN_MATH_FOUNDATIONS.md) is not implemented in the code yet, so
there is no exact statement to check. It is the next theorem to add once `tolerance.py` has the
rule.

## T6: order free (binned) sums (`Binned.lean`)

**Plain words.** `ordersum.py` turns every term into an integer on one grid and adds integers in
int64. Any tree of those additions, over the terms in any order, gives the same int64 total, and
with at most `2^24 - 1` terms below `2^39` that total is the exact integer sum (no overflow), so the
single final rounding sees the same number whatever the order.

**Statements.** `tree_perm` (for any associative, commutative operation with identity, trees over
permuted leaves agree), `osum_tree`, `osum_order_free`, `osum_exact`. **Assumed:** the per term
grid rounding and the final float32 rounding are functions of their input (they are elementwise).

## T7: online softmax rescaling (`lean-analysis/OnlineSoftmax.lean`)

**Plain words.** The running state of flash style attention (reference `m`, denominator `s`,
weighted sum `o`) rescaled by `exp(m - m')` and merged across key blocks always represents the
keys seen; at the end `o / s` equals the softmax weighted sum of the values exactly, for any
sequence of steps and merges and any choice of the references (the running maximum only avoids
overflow). In real arithmetic. **Not done:** the rounding bound in terms of the key block width.

## T8: CRT reconstruction (`Crt.lean`)

**Plain words.** For Ozaki scheme II: two integers with the same residues modulo pairwise coprime
moduli, both in the symmetric range `2|x| < M = prod m_i`, are equal, so any reconstruction that
returns a value in that range with the right residues returns the exact integer product.
**Statements.** `crt_dvd`, `crt_unique`, `crt_unique_symm`. **Not done:** the existence half (that
a given reconstruction formula produces those residues).

## T9: fp8 decoders (`Fp8.lean`)

**Plain words.** All 256 codes of e4m3fn, e4m3fnuz and e5m2, checked by `decide`: NaN codes
(e4m3fn exactly 0x7F and 0xFF, e4m3fnuz exactly 0x80), largest finite values (448, 240, 57344),
e4m3fnuz is half of e4m3fn on every code except 0x7F, 0xFF, 0x80, and positive e4m3fn codes are
strictly increasing. The decoders are compared with torch's float8 dtypes on all 256 codes by the
cross check.

## Cross check against the Python simulator

`scripts/gen_lean_tc_fixture.py` exports probe vectors from `src/truestar/devsim/tensorcore.py`
to `proofs/lean/fixtures/tc_probes.json` (exact integers in units of `2^-160`): the cross chip
probe (1.0 plus 15 products of `2^-k`, k from 18 to 33, large product first or last) on every
generation's fp16, bf16 and tf32 unit, random vectors with wide exponent spreads, the fp16 engine
digit chains from C (worst cases included) on every generation, and variants with a 10 or 12 bit
window, `rd` cut and `rne` final rounding. `lake exe tccheck` evaluates each probe with the same
Lean functions T1 is proved about and compares bit for bit, checks that every digit chain is the
exact integer sum, and compares the fp8 decoders with torch. Result on 2026-11-25: 800 probes,
0 mismatches, 0 inexact digit chains, 256 fp8 codes, 0 mismatches. Changing one expected value
in the fixture gives 1 mismatch; setting every window to 23 bits gives 64 (cross chip probes)
while the digit chains stay exact.
