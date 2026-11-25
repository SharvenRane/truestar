"""Cross chip fingerprint: bit level hashes of compiled and eager results on this GPU.

    truestar fingerprint --json fp.json            run every case here (docs/CROSS_CHIP.md)
    truestar fingerprint compare a.json b.json     which settings differ, which rows are bitwise equal

Python:

    from truestar.fingerprint import run, compare_files
    doc = run(cases=["mlp"], dtypes=["float32"], batches=[1, 7])
    cmp = compare_files("a.json", "b.json"); cmp.claim_ok

Only the ``truestar_det`` arm (compiled with ``deterministic=True``) carries a claim; eager
PyTorch, torch.compile and the compiled executor's default mode are recorded for comparison.
"""

from .compare import Comparison, FingerprintFileError, compare, compare_files, load, match_table
from .data import DataMismatch, tensor_sha
from .models import BATCHES, CASES, DTYPES
from .run import ARMS, SCHEMA, environment, run, to_text

__all__ = ["run", "to_text", "environment", "compare", "compare_files", "load", "match_table", "Comparison",
           "FingerprintFileError", "DataMismatch", "tensor_sha", "CASES", "DTYPES", "BATCHES", "ARMS", "SCHEMA"]
