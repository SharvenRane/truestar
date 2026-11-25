"""Run a GPU serving engine on the built in decoder and save what it computed (docs/CONFORM_SERVING.md).

This side runs where the engine and a CUDA GPU are. It writes a results directory that
`truestar conform vllm DIR` (or `sglang`) judges anywhere, with no GPU:

    python -m truestar.conform.serving_battery run --engine vllm --out DIR [--dtype bfloat16]

Every engine setting runs in its own process (an engine holds GPU memory until its process ends):

    base             prefix caching off, chunked prefill off (one step per prompt)
    prefix_cache     prefix caching on: the prompts share a prefix and are sent one after another
    chunked_prefill  chunked prefill on, `chunk` tokens per step

Passes inside a setting (each prompt's greedy generation with top logprobs at every position):

    batch1           each prompt alone, prompt logprobs and generation
    batch1_repeat    the same requests again in the same engine
    batch8           all prompts in one call
    gen_only         each prompt alone without prompt logprobs (the request a cache can serve)

Nothing is judged here; the engine's numbers are saved as it returned them. A setting the engine
refuses is saved with its error, never replaced by another setting.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

from .serving_model import ServeSpec, prompts, write_model_dir

FORMAT = "truestar-serving-results"
FORMAT_VERSION = 1
ENGINES = ("vllm", "sglang")
# engines named on purpose and refused (docs/CONFORM_SERVING.md "After launch")
REFUSED = {"tensorrt-llm": "TensorRT-LLM", "trtllm": "TensorRT-LLM", "tensorrt_llm": "TensorRT-LLM"}

SETTINGS: Dict[str, Dict[str, Any]] = {
    "base": {"prefix_cache": False, "chunked": False,
             "passes": ["batch1", "batch1_repeat", "batch8", "gen_only"]},
    "prefix_cache": {"prefix_cache": True, "chunked": False, "passes": ["batch1", "gen_only"]},
    "chunked_prefill": {"prefix_cache": False, "chunked": True, "passes": ["batch1"]},
}
MAX_LEN = 512
MAX_SEQS = 16


def _pos(lp: Optional[float], tok: int, top: List[List[float]]) -> Dict[str, Any]:
    return {"token": int(tok), "logprob": None if lp is None else float(lp),
            "top": [[int(t), float(v)] for t, v in top]}


def _entry(tok: int, d: Any) -> Dict[str, Any]:
    """A vLLM position: mapping token id -> Logprob(logprob, rank, ...)."""
    items = list(d.items())
    lp = None
    tops = []
    for t, v in items:
        val = float(getattr(v, "logprob", v))
        if int(t) == int(tok):
            lp = val
        rank = getattr(v, "rank", None)
        tops.append((rank if rank is not None else 1 << 30, int(t), val))
    tops.sort()
    return _pos(lp, tok, [[t, v] for _, t, v in tops])


class VllmDriver:
    name = "vllm"

    def __init__(self, model_dir: str, spec: ServeSpec, setting: Dict[str, Any]):
        import vllm
        from vllm import LLM

        self.version = vllm.__version__
        kw = dict(model=model_dir, tokenizer=model_dir, dtype=spec.dtype, skip_tokenizer_init=True, seed=0,
                  max_model_len=MAX_LEN, gpu_memory_utilization=0.5, max_num_seqs=MAX_SEQS,
                  enable_prefix_caching=setting["prefix_cache"], enable_chunked_prefill=setting["chunked"],
                  max_num_batched_tokens=spec.chunk if setting["chunked"] else 4 * MAX_LEN)
        self.flags = {k: v for k, v in kw.items() if k not in ("model", "tokenizer")}
        self.llm = LLM(**kw)
        self.spec = spec

    def run(self, batch: List[List[int]], prompt_lp: bool) -> List[Dict[str, Any]]:
        from vllm import SamplingParams

        sp = SamplingParams(temperature=0.0, max_tokens=self.spec.max_tokens, logprobs=self.spec.top,
                            prompt_logprobs=self.spec.top if prompt_lp else None, ignore_eos=True,
                            detokenize=False)
        outs = self.llm.generate([{"prompt_token_ids": p} for p in batch], sp, use_tqdm=False)
        recs = []
        for p, o in zip(batch, outs):
            c = o.outputs[0]
            toks = [int(t) for t in c.token_ids]
            prompt = None
            if prompt_lp:
                prompt = [None] + [_entry(p[j], o.prompt_logprobs[j]) for j in range(1, len(p))]
            recs.append({"prompt": prompt, "tokens": toks,
                         "gen": [_entry(t, d) for t, d in zip(toks, c.logprobs or [])],
                         "cached_tokens": getattr(o, "num_cached_tokens", None)})
        return recs

    def close(self) -> None:
        pass


def _sg_triple(x) -> tuple:
    """SGLang gives (logprob, token_id, text) per position."""
    return float(x[0]) if x[0] is not None else None, int(x[1])


class SglangDriver:
    name = "sglang"

    def __init__(self, model_dir: str, spec: ServeSpec, setting: Dict[str, Any]):
        import sglang

        self.version = sglang.__version__
        kw = dict(model_path=model_dir, skip_tokenizer_init=True, dtype=spec.dtype, random_seed=0,
                  mem_fraction_static=0.5, context_length=MAX_LEN, max_running_requests=MAX_SEQS,
                  disable_radix_cache=not setting["prefix_cache"],
                  chunked_prefill_size=spec.chunk if setting["chunked"] else -1, log_level="error")
        self.flags = {k: v for k, v in kw.items() if k != "model_path"}
        self.engine = sglang.Engine(**kw)
        self.spec = spec

    def run(self, batch: List[List[int]], prompt_lp: bool) -> List[Dict[str, Any]]:
        outs = self.engine.generate(input_ids=[list(p) for p in batch],
                                    sampling_params={"temperature": 0.0, "max_new_tokens": self.spec.max_tokens,
                                                     "ignore_eos": True},
                                    return_logprob=True, logprob_start_len=0 if prompt_lp else -1,
                                    top_logprobs_num=self.spec.top)
        if isinstance(outs, dict):
            outs = [outs]
        recs = []
        for p, o in zip(batch, outs):
            m = o["meta_info"]
            toks = [int(t) for t in o["output_ids"]]
            prompt = None
            if prompt_lp:
                lps = m["input_token_logprobs"]
                tops = m.get("input_top_logprobs") or [None] * len(lps)
                if len(lps) != len(p):
                    raise RuntimeError(f"sglang returned {len(lps)} prompt logprobs for {len(p)} tokens")
                prompt = [None]
                for j in range(1, len(p)):
                    lp, t = _sg_triple(lps[j])
                    top = [list(_sg_triple(x))[::-1] for x in (tops[j] or [])]
                    prompt.append(_pos(lp, t, top))
            gen = []
            otops = m.get("output_top_logprobs") or [None] * len(toks)
            for k, x in enumerate(m["output_token_logprobs"]):
                lp, t = _sg_triple(x)
                gen.append(_pos(lp, t, [list(_sg_triple(y))[::-1] for y in (otops[k] or [])]))
            recs.append({"prompt": prompt, "tokens": toks, "gen": gen,
                         "cached_tokens": m.get("cached_tokens")})
        return recs

    def close(self) -> None:
        self.engine.shutdown()


class CpuEmulatedDriver:
    """No engine: Truestar's own CPU run in the serving dtype. Batch and cache independent by
    construction; the tests use it (and subclass it to plant faults)."""
    name = "cpu-emulated"
    version = "0"

    def __init__(self, model_dir: str, spec: ServeSpec, setting: Dict[str, Any]):
        from .serving_model import DTYPES, weights

        self.spec, self.setting, self.flags = spec, setting, {}
        self.W = weights(spec)
        self.dt = DTYPES[spec.dtype]

    def logprobs(self, ids: List[int], batch_size: int):
        from .serving_model import logprobs

        return logprobs(self.spec, self.W, ids, self.dt)

    def run(self, batch: List[List[int]], prompt_lp: bool) -> List[Dict[str, Any]]:
        recs = []
        top = self.spec.top
        for p in batch:
            toks: List[int] = []
            gen = []
            for _ in range(self.spec.max_tokens):  # greedy, recomputing the whole sequence each step
                lp = self.logprobs(p + toks, len(batch))
                row = lp[-1]
                t = int(row.argmax())
                v, i = row.topk(top)
                gen.append(_pos(float(row[t]), t, [[int(a), float(b)] for a, b in zip(i, v)]))
                toks.append(t)
            prompt = None
            if prompt_lp:
                lp = self.logprobs(p, len(batch))
                prompt = [None]
                for j in range(1, len(p)):
                    v, i = lp[j - 1].topk(top)
                    prompt.append(_pos(float(lp[j - 1, p[j]]), p[j], [[int(a), float(b)] for a, b in zip(i, v)]))
            recs.append({"prompt": prompt, "tokens": toks, "gen": gen, "cached_tokens": None})
        return recs

    def close(self) -> None:
        pass


DRIVERS = {"vllm": VllmDriver, "sglang": SglangDriver, "cpu-emulated": CpuEmulatedDriver}


def run_setting(engine: str, setting_name: str, model_dir: str, spec: ServeSpec, driver_cls=None) -> Dict[str, Any]:
    """One engine instance, every pass of one setting. Errors are saved, not raised."""
    setting = SETTINGS[setting_name]
    P = prompts(spec)
    out: Dict[str, Any] = {"setting": setting_name, "passes": {}, "error": None, "flags": None, "version": None}
    t0 = time.time()
    drv = None
    try:
        drv = (driver_cls or DRIVERS[engine])(model_dir, spec, setting)
        out["flags"], out["version"] = drv.flags, drv.version
        for name in setting["passes"]:
            if name == "batch8":
                recs = drv.run(P, True)
            else:
                recs = []
                for p in P:
                    recs += drv.run([p], name != "gen_only")
            out["passes"][name] = recs
    except Exception as exc:  # saved with the setting; the judge reports it (TS1322)
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()[-4000:]
    finally:
        if drv is not None:
            try:
                drv.close()
            except Exception:
                pass
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _gpu() -> Optional[str]:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


def run_all(engine: str, out_dir: str, spec: ServeSpec, settings: Optional[List[str]] = None,
            in_process: bool = False, driver_cls=None, plant: Optional[str] = None) -> str:
    """Write out_dir/meta.json and out_dir/runs.json. Returns out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    model_dir = os.path.join(out_dir, "model")
    sha = write_model_dir(spec, model_dir)
    if plant:  # a deliberate conversion bug, to show the judge catches one in a real engine
        key, _, val = plant.partition("=")
        cp = os.path.join(model_dir, "config.json")
        with open(cp, encoding="utf-8") as f:
            conf = json.load(f)
        if key not in conf:
            raise ValueError(f"--plant {plant}: config.json has no key {key!r}")
        conf[key] = json.loads(val)
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(conf, f, indent=1)
    with open(os.path.join(out_dir, "spec.json"), "w", encoding="utf-8") as f:
        json.dump(spec.as_dict(), f)
    runs: Dict[str, Any] = {}
    for s in settings or list(SETTINGS):
        if in_process:
            runs[s] = run_setting(engine, s, model_dir, spec, driver_cls)
            continue
        res = os.path.join(out_dir, f"setting_{s}.json")
        cmd = [sys.executable, "-m", "truestar.conform.serving_battery", "setting", "--engine", engine,
               "--setting", s, "--model-dir", model_dir, "--spec", os.path.join(out_dir, "spec.json"), "--out", res]
        p = subprocess.run(cmd, capture_output=True, text=True)
        with open(os.path.join(out_dir, f"log_{s}.txt"), "w", encoding="utf-8") as f:
            f.write(p.stdout[-20000:] + "\n--- stderr ---\n" + p.stderr[-20000:])
        if os.path.isfile(res):
            with open(res, encoding="utf-8") as f:
                runs[s] = json.load(f)
            os.remove(res)  # merged into runs.json
        else:
            runs[s] = {"setting": s, "passes": {}, "flags": None, "version": None,
                       "error": f"the setting's process exited {p.returncode} without results; "
                                f"stderr tail: {p.stderr[-600:]}"}
    try:
        import torch

        tv = torch.__version__
    except Exception:
        tv = None
    meta = {"format": FORMAT, "format_version": FORMAT_VERSION, "engine": engine,
            "engine_version": next((r.get("version") for r in runs.values() if r.get("version")), None),
            "torch": tv, "gpu": _gpu(), "platform": platform.platform(), "weights_sha256": sha,
            "spec": spec.as_dict(), "prompts": prompts(spec), "settings": list(runs), "planted": plant}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    with open(os.path.join(out_dir, "runs.json"), "w", encoding="utf-8") as f:
        json.dump(runs, f)
    return out_dir


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m truestar.conform.serving_battery")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--engine", required=True, choices=sorted(DRIVERS))
    r.add_argument("--out", required=True)
    r.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    r.add_argument("--settings", default=None, help="comma list (default: all)")
    r.add_argument("--plant", default=None, help="KEY=JSON: serve a config.json with this value changed (a planted "
                   "conversion bug the judge must catch), e.g. rope_theta=1000000.0")
    s = sub.add_parser("setting")
    s.add_argument("--engine", required=True, choices=sorted(DRIVERS))
    s.add_argument("--setting", required=True, choices=sorted(SETTINGS))
    s.add_argument("--model-dir", required=True)
    s.add_argument("--spec", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        spec = ServeSpec(dtype=a.dtype)
        run_all(a.engine, a.out, spec, a.settings.split(",") if a.settings else None, plant=a.plant)
        print(a.out)
        return 0
    with open(a.spec, encoding="utf-8") as f:
        spec = ServeSpec.from_dict(json.load(f))
    res = run_setting(a.engine, a.setting, a.model_dir, spec)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f)
    print(json.dumps({"setting": a.setting, "error": res["error"], "seconds": res["seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
