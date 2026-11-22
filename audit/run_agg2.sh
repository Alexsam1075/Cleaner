#!/usr/bin/env bash
cd /home/cleaner/Cleaner/audit; export ASTRAMIS_DATA=/home/cleaner/Cleaner/datasets/raw
~/venv/bin/python -W ignore full_pipeline.py --agg 0.90 --agg-mat frame > fp_agg_frame.log 2>&1
~/venv/bin/python -W ignore full_pipeline.py --agg 0.90 --agg-mat conf > fp_agg_conf.log 2>&1
~/venv/bin/python -W ignore full_pipeline.py --folds 2 --detector final/big_f2/weights/last.pt --tag _big > fp_big_f2.log 2>&1
