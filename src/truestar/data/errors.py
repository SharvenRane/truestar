"""Error codes of the data loader and the checkpoint format (TS1380 to TS1389).

The codes live in ``truestar.codes.CODES`` with their rows in docs/AGENT_INTERFACE.md;
``DATA_CODES`` is the view of that range. Every error raised by ``truestar.data`` and by checkpoint
loading carries one of these codes in ``.code`` and starts its message with it.
"""

from __future__ import annotations

from typing import Dict, Tuple

from ..codes import CODES
from ..ir.core import IRError

# code: (severity, slug, one line meaning), the TS1380 to TS1389 range of codes.CODES
DATA_CODES: Dict[str, Tuple[str, str, str]] = {c: CODES[c] for c in (f"TS{i}" for i in range(1380, 1390))}


class DataError(IRError):
    """An error of the data loader or the checkpoint format, with its code in ``.code``."""

    def __init__(self, code: str, message: str):
        if code not in DATA_CODES:
            raise ValueError(f"unknown data code {code}")
        self.code = code
        self.slug = DATA_CODES[code][1]
        self.detail = message
        super().__init__(f"{code} {self.slug}: {message}")

    def __reduce__(self):  # picklable across worker processes
        return (DataError, (self.code, self.detail))


def raise_(code: str, message: str):
    raise DataError(code, message)
