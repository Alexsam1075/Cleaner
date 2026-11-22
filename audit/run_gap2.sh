#!/usr/bin/env bash
cd /home/cleaner/Cleaner/audit; export ASTRAMIS_DATA=/home/cleaner/Cleaner/datasets/raw
P=/home/cleaner/Cleaner/agents2/integrator
for s in 0.85 0.80 0.88; do
~/venv/bin/python $P/pipeline_v2.py --folds 1 2 3 4 --variants b --external material_v2=/home/cleaner/Cleaner/agents2/material/predict_material.py:comp_material form_v2=$P/ext_form.py:comp_form series_pool=$P/ext_form.py:series_pool take_t1=$P/ext_take.py:take_t1 --combo "fvp=form_v2+series_pool+material_v2+b" "fvp_t1=form_v2+series_pool+material_v2+b+take_t1" --only fvp fvp_t1 --two-box 0.0001 --agg $s --agg-gap 1000 --tag g1k_s${s} 2>&1 | grep -v -i warn | grep "fvp\|== fold" | cut -c1-100
echo "=== gap 1000 sim $s done"
done
