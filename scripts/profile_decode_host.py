"""Host time of one compiled decode step (cProfile over 20 steps of the dense Llama step).

    PYTHONPATH=src python scripts/profile_decode_host.py
"""
import sys, os, cProfile, pstats, io, torch
sys.path.insert(0, "examples")
os.environ["TRUESTAR_VERIFY_DEVICE"] = "cpu"
import llama_paged_decode as P
from truestar.compiler import compile_program
torch.manual_seed(0)
c = P.Cfg(vocab=8192, dim=512, heads=8, kv_heads=2, head_dim=64, layers=4, ffn=1408, block=16)
model = P.Llama(c).eval()
cd = compile_program(P.import_steps(model, capacity=64, max_capacity=1 << 15)[0], "cuda", "float32")
N = 2048
kc = torch.zeros(c.layers, 1, c.kv_heads, N + 64, c.head_dim, device="cuda"); vc = kc.clone()
ids = torch.zeros(1, 1, dtype=torch.long, device="cuda")
for s in range(3):
    lg, kc, vc = cd([ids, torch.tensor([N + s], device="cuda"), kc, vc])
torch.cuda.synchronize()
pr = cProfile.Profile(); pr.enable()
for s in range(20):
    lg, kc, vc = cd([ids, torch.tensor([N + 3 + s], device="cuda"), kc, vc])
torch.cuda.synchronize()
pr.disable()
st = io.StringIO(); pstats.Stats(pr, stream=st).sort_stats("cumulative").print_stats(18); print(st.getvalue()[:4500])
print("steps in plan:", len(cd.plan.steps) if hasattr(cd, "plan") else "?")
