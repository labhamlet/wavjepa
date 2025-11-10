import time
from contextlib import nullcontext
from itertools import product
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

device = "cuda:0"

# device = 'cpu'

nrep = 10

args = [
    "evaluate",
    "d_model",
    "dim_feedforward",
    "num_encoder_layers",
    "num_decoder_layers",
    "length_in",
    "length_tgt",
    "batch_size",
]


def benchmark(
    length_in=10,
    length_tgt=10,
    d_model=128,
    batch_size=10,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=128,
    evaluate=False,
):
    out = {k: v for k, v in locals().items() if k in args}
    loss_fn = nn.MSELoss()
    transformer = nn.Transformer(
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        batch_first=True,
        device=device,
    )
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)
    x = torch.randn(batch_size, length_in, d_model, device=device)
    tgt = torch.randn(batch_size, length_tgt, d_model, device=device)
    targets = torch.randn(batch_size, length_tgt, d_model, device=device)
    mem = None
    y = None
    if evaluate:
        transformer.eval()
    with torch.inference_mode() if evaluate else nullcontext():
        try:
            t0 = time.time()
            mem = transformer.encoder(x)
            out["time_encoder"] = time.time() - t0
            t1 = time.time()
            y = transformer.decoder(tgt=tgt, memory=mem)
            out["time_decoder"] = time.time() - t1
            t2 = time.time()
            loss = loss_fn(y, targets)
            out["time_loss"] = time.time() - t2
            t3 = time.time()
            loss.backward()
            out["time_backward"] = time.time() - t3
            t4 = time.time()
            optimizer.step()
            out["time_step"] = time.time() - t4
            out["time_total"] = time.time() - t0
        finally:
            del mem, y, transformer, x, tgt
            return out


grid_args = dict(
    evaluate=[True, False],
    # length_in=[10, 8 * 10, 8 * 20, 64 * 100],
    length_in=[8 * 40, 8 * 60],
    length_tgt=[10, 8 * 10, 8 * 20, 64 * 100],
    batch_size=[10, 30, 50, 100],
)
kw_list = [
    dict(length_in=2, length_tgt=2, evaluate=True),
]
kw_list += [
    dict(zip(grid_args.keys(), values)) for values in product(*grid_args.values())
]

outs = []
for kw in tqdm(kw_list):
    for _ in tqdm(range(nrep)):
        outs.append(benchmark(**kw))

df = pd.DataFrame(outs)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)
print(df.groupby(args).mean())
df.to_csv(Path("export") / "benchmark_transformer.csv")
