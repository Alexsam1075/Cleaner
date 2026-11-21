"""Object volume from RGB-D, given the object's form and material.

    from volume import VolumeEstimator
    est = VolumeEstimator.load("checkpoints/volume_parts/calibrator.json",
                               "datasets/raw/densities.csv", camera)
    r = est.estimate(depth_mm, mask, floor_plane, form="solid", material="glass")
    r.volume_ml, r.mass_g
"""
from .estimator import VolumeEstimator, VolumeResult

__all__ = ["VolumeEstimator", "VolumeResult"]
