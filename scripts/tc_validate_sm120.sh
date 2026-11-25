#!/bin/sh
# The full sm_120 measurement behind docs/TENSOR_CORE_MODELS.md: features, fits, then about 10^6
# random dot products per row. Each step is a short GPU run (seconds to a minute).
# Usage: sh scripts/tc_validate_sm120.sh /path/to/python
PY=${1:-python}
S="$(dirname "$0")/tc_probe_sm120.py"
$PY "$S" features
for d in fp16 bf16 tf32 e4m3 e5m2; do $PY "$S" fit --dtype $d; done
$PY "$S" fit --dtype fp16 --acc fp16
for d in fp16 bf16 tf32; do for k in 16 64 256; do $PY "$S" validate --dtype $d --k $k --path triton; done; done
for d in e4m3 e5m2; do for k in 32 64 256; do $PY "$S" validate --dtype $d --k $k --path triton; done; done
for k in 32 256; do $PY "$S" validate --dtype fp16 --acc fp16 --k $k --path triton; done
for d in fp16 bf16; do for k in 16 256 1024; do $PY "$S" validate --dtype $d --k $k --path cublas; done; done
$PY "$S" validate --dtype tf32 --k 256 --path cublas
$PY "$S" validate --dtype tf32 --k 16 --path cublas --split 2
$PY "$S" validate --dtype bf16 --k 256 --path cublas --side 447 --group 8
$PY "$S" validate --dtype fp16 --k 256 --path cublas --side 447
for k in 32 256 1024; do $PY "$S" validate --dtype e4m3 --k $k --path scaled_mm; $PY "$S" validate --dtype e4m3 --k $k --path scaled_mm --fast; done
