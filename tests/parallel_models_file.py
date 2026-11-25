"""Model file for the `truestar parallel` command line and gloo tests: the tiny transformer of
tests/parallel_models.py with its plan (sequence parallel marks when the layout asks for sp)."""

import parallel_models as M


def get_model():
    return M.tiny()


def get_inputs():
    return M.tiny_inputs()


def get_parallel_plan(layout=None):
    return M.tiny_plan(sp=True if layout is None else layout.sp)
