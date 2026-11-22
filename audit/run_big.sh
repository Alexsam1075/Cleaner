#!/usr/bin/env bash
cd /home/cleaner/Cleaner/audit
export ASTRAMIS_DATA=/home/cleaner/Cleaner/datasets/raw
for f in 3 2 0; do ~/venv/bin/python -W ignore train_det_big.py --fold $f > big_f$f.log 2>&1; done
echo ALLDONE > big.done
