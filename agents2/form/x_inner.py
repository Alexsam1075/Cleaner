"""Inspect inner-CV tables per fold: how do scores vary with lam / spec; robustness of the selection."""
import numpy as np
from bank import gt_bank
from form_model2 import FormModel2, GRID
from common import load_split

bank, y, oid, src, meta = gt_bank()
grid = {"spec": ["clipL_c14+zs", "clipL_c14+zs+depth", "clipL_c14+dino+zs"], "sw": [0.5], "wd": [1e-2], "lam": [0.0, 1.0, 2.0, 4.0, 8.0]}
for fold in (1, 2, 3):
    sp = load_split(fold)
    for seed in (0, 10):
        m = FormModel2(bank, y, oid, src, grid=grid, seed=seed, repeats=3).fit(sp["train_object_ids"])
        print(f"fold {fold} seed {seed}")
        for sc, cfg in m.inner_table:
            print(f"   {cfg[0]:22s} wd={cfg[2]} lam={cfg[3]:<4} bal {sc[0]:.3f} obj {sc[1]:.3f} ll {sc[2]:.3f}")
