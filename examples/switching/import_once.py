"""Bring a PyTorch or Hugging Face model in once, then keep working in Truestar (docs/SWITCHING.md, section 4).

    python examples/switching/import_once.py        # CPU, about 20 s

Step 1 needs the PyTorch class: import the model into a Truestar program (shapes typed, batch
symbolic), check it against the module in float64, save it as a `.tsx` file.
Step 2 does not: load the file, fine tune it with Truestar's trainer (one program per step), write
the trained program back to a file and run it compiled. If `transformers` is installed, a small
random weight Llama goes through step 1 as well.
"""

import dataclasses
import os
import sys
import tempfile

import torch
from torch.export import Dim

from truestar import export, ir
from truestar import train as T
from truestar.ir import serialize


class Classifier(torch.nn.Module):            # an existing PyTorch model
    def __init__(self):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(20, 64), torch.nn.GELU(), torch.nn.LayerNorm(64),
                                        torch.nn.Linear(64, 5))

    def forward(self, x):
        return self.body(x)


def step1_import(path):
    torch.manual_seed(0)
    model, x = Classifier().double().eval(), torch.randn(8, 20, dtype=torch.float64)
    prog = ir.import_module(model, (x,), dynamic_shapes=({0: Dim("B", min=1, max=4096)},))
    with torch.no_grad():
        got, want = ir.run(prog, x)[0], model(x)
    print(f"imported {prog.name}: {len(prog.nodes)} IR nodes, {len(prog.params)} params; "
          f"max difference to the module in float64 {float((got - want).abs().max()):.1e}")
    serialize.save(prog, path)
    return float((got - want).abs().max())


def step2_keep_working(path, out_path):
    prog = serialize.load(path).program                      # no PyTorch class from here on
    step = T.build_train_step(prog, T.CrossEntropy(), T.AdamW(3e-3, weight_decay=0.01),
                              params=[v.name for v in prog.params], clip_grad_norm=1.0)
    trainer = T.Trainer(step)
    g = torch.Generator().manual_seed(1)
    losses = []
    for _ in range(30):
        x = torch.randn(64, 20, generator=g, dtype=torch.float64)
        y = (x[:, :5] - x[:, 5:10]).argmax(-1)
        losses.append(trainer.train_step(x, y).loss)
    trained = dataclasses.replace(prog, param_data={**prog.param_data, **trainer.params()})
    serialize.save(trained, out_path)
    run = export.load(out_path, executor="cpu")              # compiled for the CPU, checked on first call
    xs = torch.randn(3, 20, dtype=torch.float64)
    print(f"fine tuned 30 steps: loss {losses[0]:.3f} -> {losses[-1]:.3f}; compiled run of the trained "
          f"file on 3 rows: {tuple(run(xs)[0].shape)}")
    return losses[-1] < losses[0]


def step1_hugging_face():
    try:
        import transformers
    except ImportError:
        print("transformers not installed: Hugging Face step skipped")
        return True
    cfg = transformers.LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                   attn_implementation="sdpa")
    hf = transformers.LlamaForCausalLM(cfg).double().eval()

    class Logits(torch.nn.Module):            # the call Truestar should own: ids -> logits
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, ids):
            return self.m(input_ids=ids, use_cache=False).logits

    ids = torch.randint(0, 128, (2, 16), generator=torch.Generator().manual_seed(0))
    prog = ir.import_module(Logits(hf), (ids,), dynamic_shapes=({0: Dim("B", min=1, max=64)},))
    got = ir.run(prog, ids)[0]
    with torch.no_grad():
        plain = hf(input_ids=ids).logits
        # Llama's rotary embedding casts to float32 inside the float64 model. Truestar's reference
        # computes those casts exactly; this mode does the same on the PyTorch side for the comparison.
        with ir.reference.exact_float32_casts():
            exact = hf(input_ids=ids).logits
    d_plain, d_exact = float((got - plain).abs().max()), float((got - exact).abs().max())
    print(f"Hugging Face Llama ({transformers.__version__}): {len(prog.nodes)} IR nodes; max difference to "
          f"transformers in float64 {d_exact:.1e} (with its float32 rotary rounding left in: {d_plain:.1e})")
    return d_exact < 1e-10


def main():
    with tempfile.TemporaryDirectory() as tmp:
        diff = step1_import(os.path.join(tmp, "classifier.tsx"))
        trained = step2_keep_working(os.path.join(tmp, "classifier.tsx"), os.path.join(tmp, "classifier_tuned.tsx"))
    hf_ok = step1_hugging_face()
    ok = diff < 1e-12 and trained and hf_ok
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
