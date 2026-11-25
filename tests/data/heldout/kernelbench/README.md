# KernelBench held out data (third party)

This folder holds third party data used only as a held out test set by
`scripts/bench_heldout.py` (protocol in `docs/KERNEL_CHECK.md`). It is not part of the
`truestar` package and is not shipped in the wheel or sdist.

* `problems/level1/`, `problems/level2/`: reference problems from KernelBench,
  <https://github.com/ScalingIntelligence/KernelBench>, commit `b866d6ab57`.
  Copyright (c) 2023 Anne Ouyang, Simon Guo, Azalia Mirhoseini, Scaling Intelligence Lab,
  Stanford University. MIT License, full text in [LICENSE](LICENSE). The files are unchanged
  except for their names: upstream names longer than 48 characters are shortened so deep
  Windows clones stay under the path limit. `problems/NAMES.json` maps each upstream name
  (`level<N>/<upstream name>`) to the stored name; `scripts/bench_heldout.py` reads it.
* `samples/<level>/<model>/problem_<id>.json`: generated kernels with their published eval
  results from the `kernelbench-samples` dataset,
  <https://huggingface.co/datasets/ScalingIntelligence/kernelbench-samples> (baseline_eval),
  Apache License 2.0 (the same text as this repository's root `LICENSE`). Unchanged.

`scripts/fetch_heldout.py` downloads the same files again. METR's KernelBenchFiltered tasks
are not included (that repository has no license); the fetch script puts them in
`results/heldout_metr/`, which is never committed. Truestar's own measured results on that set
(labels and ratios only, no task source) are in `results/heldout_runs/`.
