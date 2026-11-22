#!/usr/bin/env bash
cd /home/cleaner/Cleaner/audit
export ASTRAMIS_DATA=/home/cleaner/Cleaner/datasets/raw
~/venv/bin/python -W ignore full_pipeline.py --folds 3 --detector final/big_f3/weights/last.pt --tag _big > fp_big_f3.log 2>&1
~/venv/bin/python -W ignore full_pipeline.py --folds 1 2 3 > fp_base.log 2>&1
