"""Text embeddings of all form prompt sets for CLIP-L and SigLIP -> W/text_cache.npz"""
import numpy as np
from common import W
from encoders import Encoders
enc = Encoders()
np.savez_compressed(W / "text_cache.npz", **{f"{t}|{s}": T for (t, s), T in enc.text.items()})
print(sorted(enc.text))
