"""nn.summary and nn.count: parameter counts equal torch.nn's and the drawn trees'."""

import torch

from truestar import nn


def test_counts_match_torch_and_the_drawn_tree():
    net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm(8), nn.ReLU(), nn.GlobalAvgPool(),
                        nn.Linear(8, 10))
    tw = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.BatchNorm2d(8), torch.nn.ReLU(),
                             torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(8, 10))
    s = nn.summary(net)
    assert s.params == sum(p.numel() for p in tw.parameters()) == net.num_params()
    assert s.state == 16                     # running mean and variance (torch also counts num_batches_tracked)
    n, b = nn.count(net.init(seed=0))
    assert n == s.params and b == 4 * s.params
    assert nn.count(net.init_state())[0] == s.state
    assert [c for c, _ in s.by_child()] == ["0", "1", "4"]
    text = str(s)
    assert "0.w" in text and "(state)" in text and f"{s.params:,}" in text


def test_transformer_block_and_dtype_bytes():
    blk = nn.TransformerBlock(64, 4, 256, dtype="bf16")
    s = nn.summary(blk, "block")
    tw = torch.nn.TransformerEncoderLayer(64, 4, 256, norm_first=True, batch_first=True)
    assert s.params == sum(p.numel() for p in tw.parameters())
    assert s.bytes == 2 * s.params and all(r.dtype == "bf16" for r in s.rows)
    assert str(s).startswith("block")
