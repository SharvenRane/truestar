"""Where a float16 Llama decode program loses accuracy, before and after mixed precision islands.

    PYTHONPATH=src python scripts/probe_fp16_rope.py

Compiles examples/llama_paged_decode.py's dense step (the scripts/bench_decode.py model) in float16,
once with rotary angles in the model's own dtype and once with ``rope_f32`` (an explicit ``.float()``,
as Hugging Face computes them), checks off, and compares one step at positions 10 to 4000 with the
float64 model (logits and the new K); then runs each with Truestar's own checks on. Found 2026-11-24:
without islands the new K's error follows the float16 angle's (0.32 at position 2047, 0.41 at 4000)
and the checked program is refused; with them it stays below 1e-3 and the checked program passes.
"""
import sys, os, torch
sys.path.insert(0, "examples")
import llama_paged_decode as P
from truestar.compiler import compile_program
torch.manual_seed(0)
for rope_f32 in (False, True):
    c = P.Cfg(vocab=8192, dim=512, heads=8, kv_heads=2, head_dim=64, layers=4, ffn=1408, block=16, rope_f32=rope_f32)
    torch.manual_seed(0)
    model = P.Llama(c).eval()
    dense, _ = P.import_steps(model, capacity=64, max_capacity=1 << 15)
    c16 = compile_program(dense, "cuda", "float16", verify="off")
    print("rope_f32", rope_f32, "islands", len(c16.report.islands), [i[0] for i in c16.report.islands][:6])
    B, N = 1, 4096
    for p in (10, 300, 2047, 4000):
        kc = torch.zeros(c.layers, B, c.kv_heads, N, c.head_dim, device="cuda"); vc = kc.clone()
        ids = torch.randint(0, c.vocab, (B, 1), device="cuda", generator=torch.Generator(device="cuda").manual_seed(p))
        lg, k16, v16 = c16([ids, torch.tensor([p], device="cuda"), kc.half(), vc.half()])
        m64 = P.DenseStep(model).double().cuda()
        with torch.no_grad():
            ref, k64, v64 = m64(ids, torch.tensor([p], device="cuda"), kc.double(), vc.double())
        model.float().cpu()
        rel = lambda a, b: ((a.double() - b).abs().max() / b.abs().max()).item()
        print(f"  pos {p}: logits rel {rel(lg, ref):.3e}, new K rel {rel(k16[:, :, :, p], k64[:, :, :, p]):.3e}")
    v = compile_program(dense, "cuda", "float16")      # Truestar's own first call checks
    kc = torch.zeros(c.layers, B, c.kv_heads, N, c.head_dim, device="cuda").half()
    try:
        v([ids, torch.tensor([2047], device="cuda"), kc, kc.clone()]); print("  checked float16 program: passes")
    except Exception as e:
        print("  checked float16 program:", type(e).__name__, str(e).splitlines()[0][:120])
