import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.channels.layout import _cart_to_sph, _pol_to_cart
from mne.viz.evoked import _rgb
from umap import UMAP

from sjepa.functions import pos_encode_continuous

n_dim = 16
montage = mne.channels.make_standard_montage("standard_1020")
ch_names, locs = zip(*montage.get_positions()["ch_pos"].items())
pos = np.array(locs)
pos2d = _pol_to_cart(_cart_to_sph(pos))

xmin = 0
xmax = 10 * max(abs(pos.max()), abs(pos.min()))
emb = np.stack(
    [
        np.concatenate(
            [pos_encode_continuous(p, xmin, xmax, n_dim) for p in row], axis=0
        )
        for row in pos
    ],
    axis=0,
)
metric = "manhattan"
metric = "cosine"

emb2d = UMAP(n_components=2, metric=metric).fit_transform(emb)

colours = _rgb(pos[:, 0], pos[:, 1], pos[:, 2])

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].scatter(pos2d[:, 0], pos2d[:, 1], c=colours)
axes[1].scatter(emb2d[:, 0], emb2d[:, 1], c=colours)
plt.show()
