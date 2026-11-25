# Catch a runtime that lies

Script: [examples/cookbook/catch_lying_runtime.py](https://github.com/SharvenRane/truestar/blob/main/examples/cookbook/catch_lying_runtime.py)

The file that ships is not the model that was tested: a converter rewrites ops, and the runtime
picks its own kernels and precision. `conform` runs the shipped file in the runtime that executes
it and judges every result against the model in float64, with the conversion (the file's
meaning) and the runtime (what it computes) judged separately ([CONFORM.md](../CONFORM.md)).

```python
from truestar.conform import conform
from truestar.export import export

export(model, x, "classifier.tsx")      # refused unless the file matches the model
rep = conform(model, x, "classifier.tsx", runtimes=["tsx:reference", "tsx:cpu"])
print(rep.to_text())
```

To show a lie, the script runs the same compiled runtime with every result rounded through float16
before it is returned, as a runtime that quietly switches to half precision would (the
`fault_outputs` hook of `conform`'s `options`, used by its tests). Output:

```
truestar conform: .../classifier.tsx (tsx, 6 nodes, sha256 306c53796e85), 8 samples, 0 modules named in the file
  tsx:reference   OK         0/512 flipped  max rel 2.6e-08
      residency: all 6 nodes on cpu (single executor, no partitioning)
  tsx:cpu         OK         0/512 flipped  max rel 7.3e-08
      residency: all 6 nodes on cpu (single executor, no partitioning)
status: ok

the same runtime, returning float16 rounded results:
truestar conform: .../classifier.tsx (tsx, 6 nodes, sha256 306c53796e85), 8 samples, 0 modules named in the file
  tsx:cpu         SUSPECT    0/512 flipped  max rel 0.00022
      precision: declared float32, observed float16
      file in float64 vs model: rel 4.1e-16; runtime vs file: rel 0.00022
      [TS627] the error (0.00022) is what a correct float16 implementation shows (0.00058): tsx:cpu computes this float32 file in float16
status: suspect
```

No prediction changed, so a check on the labels alone would pass; conform names the precision
instead. The file itself is exact (4.1e-16 against the model), so the fault is the runtime's.

From the command line, with a model file that defines `get_model()` and `get_inputs()`:

```
truestar conform model.py model.tsx
truestar conform model.py model.tflite --runtimes tflite:interp,tflite:gpu
truestar conform replay model.tsx.tsgold model.tsx         # after a runtime, SDK or driver upgrade
```

The same check covers ONNX Runtime (`ort:cpu`, `ort:cuda`), LiteRT (`tflite:*`) and ExecuTorch
(`pte:cpu`) files when those runtimes are installed; `truestar conform runtimes` lists what can run
here and why the rest cannot.

## Limits

* The reference is a PyTorch module (what `truestar export` takes). A native function saved with
  `truestar.ir.serialize.save` has no model to judge it against yet.
* A `.tsx` file names no modules, so blame stops at the whole file; ONNX and LiteRT files are
  localized to the module ([CONFORM.md](../CONFORM.md)).
