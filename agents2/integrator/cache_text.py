"""Одноразово: текстовые эмбеддинги описаний материалов (fp.Encoders().text) -> text_mat.npz,
чтобы оценка на фолдах не грузила CLIP/SigLIP на GPU."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/cleaner/Cleaner/audit")
import full_pipeline as fp  # noqa: E402

OUT = Path(__file__).resolve().parent / "text_mat.npz"
enc = fp.Encoders()
np.savez(OUT, **{f"{tag}|{sname}": v for (tag, sname), v in enc.text.items()})
print("saved", OUT, len(enc.text))
