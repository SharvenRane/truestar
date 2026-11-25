"""`truestar serve`: a small LLM server whose answers do not change with batching, prefix caching or
chunked prefill (docs/SERVE.md).

* serve/model.py: the model (a Llama style decoder from a config in code) and its one step program,
  every token a row of a paged KV cache step.
* serve/engine.py: continuous batching, KV blocks with a prefix cache, chunked prefill, keyed sampling.
* serve/check.py: the guarantee, checked bit for bit, and the float64 check.
* serve/http.py: the OpenAI compatible HTTP subset (/v1/completions, /v1/chat/completions, /v1/models).
"""

from .engine import Engine, Request, ServeError, StepRunner, encode, sample, text_of  # noqa: F401
from .model import PRESETS, ServeConfig, TinyLM, config  # noqa: F401
