#!/usr/bin/env bash
cd /home/cleaner/Cleaner/audit; export ASTRAMIS_DATA=/home/cleaner/Cleaner/datasets/raw
for t in 0.80 0.85 0.90; do ~/venv/bin/python -W ignore full_pipeline.py --agg $t > fp_agg_$t.log 2>&1; done
