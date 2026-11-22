#!/usr/bin/env bash
cd /home/cleaner/Cleaner/audit; export ASTRAMIS_DATA=/home/cleaner/Cleaner/datasets/raw
P=/home/cleaner/Cleaner/agents2/integrator
for g in 10 40 1000; do for s in 0.90 0.85 0.93; do
~/venv/bin/python $P/pipeline_v2.py --folds 1 2 3 4 --variants b --external material_v2=/home/cleaner/Cleaner/agents2/material/predict_material.py:comp_material form_v2=$P/ext_form.py:comp_form series_pool=$P/ext_form.py:series_pool --combo "fvp=form_v2+series_pool+material_v2+b" --only fvp --two-box 0.0001 --agg $s --agg-gap $g --tag gap${g}_s${s} 2>&1 | grep -v -i warn | grep "fvp\|== fold"
echo "=== gap $g sim $s done"
done; done
