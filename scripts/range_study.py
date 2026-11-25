"""Measurements for docs/RANGE.md: `truestar range` against PyTorch's own low precision execution.

For every model and target: overflow sites found versus the ground truth (truestar.rangecheck
.groundtruth: PyTorch CPU kernels in float16, every op fed correct inputs), false positives, the
analysis time, and the baseline people use today (run the model in float16 and look at the output
cosine). Run from the repo root:

    PYTHONPATH=src python scripts/range_study.py
"""

import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "examples"))

import range_models as rm  # noqa: E402

from truestar.rangecheck import range_check  # noqa: E402
from truestar.rangecheck.groundtruth import chained_run, ground_truth, match  # noqa: E402


def cases():
    import range_model

    bge, bge_ids = rm.bge_small_like(), rm.bge_inputs()
    gem, gem_ids = rm.gemma3_like(), rm.gemma_inputs()
    yield "bge small (outliers)", bge, bge_ids, "webgpu-f16"
    yield "bge small (outliers)", bge, bge_ids, "fp16"
    yield "Gemma 3 style", gem, gem_ids, "fp16"
    yield "Gemma 3 style", gem, gem_ids, "ane"
    yield "Gemma 3 style", gem, gem_ids, "bf16"
    yield "Gemma 3 style", gem, gem_ids, "fp8-e5m2"
    bert = rm.bert_base()
    yield "BERT base (clean)", bert, bge_ids, "fp16"
    yield "BERT base (clean)", bert, bge_ids, "webgpu-f16"
    yield "ResNet50 (clean)", rm.resnet_calibrated("resnet50"), torch.randn(4, 3, 96, 96,
                                                                          generator=torch.Generator().manual_seed(3)), "fp16"
    yield "examples/range_model.py", range_model.get_model(), range_model.get_inputs()[0], "fp16"


def main():
    torch.set_num_threads(2)
    rows = []
    for name, model, x, target in cases():
        rep = range_check(model, (x,), target=target)
        pred = [s for s in rep.sites + rep.info if s.kind in ("overflow", "hidden_overflow") and s.origin]
        gt_dtype = {"fp16": torch.float16, "ane": torch.float16, "webgpu-f16": torch.float16,
                    "bf16": torch.bfloat16}.get(target)
        row = {"model": name, "target": target, "status": rep.status, "sites": len(rep.sites),
               "origins": len(pred), "analysis_s": rep.seconds["total"],
               "target_cos": rep.target_run.cosine_mean,
               "remedy": rep.chosen.remedy if rep.chosen else "-",
               "remedy_cos_min": rep.chosen.result.cosine_min if rep.chosen else None}
        if gt_dtype is not None:
            t = time.perf_counter()
            gt = ground_truth(model, (x,), gt_dtype, decompose_norms=target == "webgpu-f16")
            row["gt_s"] = time.perf_counter() - t
            m = match(pred, [g for g in gt.sites if g.inputs_in_range])
            row.update(true=m["true_sites"], found=m["found"], missed=m["missed"], fp=m["false_positives"],
                       misaligned=gt.misaligned)
            base = chained_run(model, (x,), gt_dtype, decompose_norms=target == "webgpu-f16")
            row["baseline_cos"] = base["cosine_mean"]
        rows.append(row)
        print(row, flush=True)
    print()
    print("| model | target | status | true sites | found | missed | false pos. | analysis s | baseline cosine "
          "(PyTorch chained) | emulated cosine | remedy | remedied cosine min |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        f = lambda k, fmt="{}": fmt.format(r[k]) if r.get(k) is not None else "-"  # noqa: E731
        print(f"| {r['model']} | {r['target']} | {r['status']} | {f('true')} | {f('found')} | {f('missed')} | "
              f"{f('fp')} | {r['analysis_s']:.1f} | {f('baseline_cos', '{:.4f}')} | {r['target_cos']:.4f} | "
              f"{r['remedy']} | {f('remedy_cos_min', '{:.6f}')} |")


if __name__ == "__main__":
    main()
