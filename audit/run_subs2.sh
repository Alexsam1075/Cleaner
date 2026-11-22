#!/usr/bin/env bash
cd /home/cleaner/Cleaner/audit; export ASTRAMIS_DATA=/home/cleaner/Cleaner/datasets/raw
P=/home/cleaner/Cleaner/agents2/integrator
EXT="material_v2=/home/cleaner/Cleaner/agents2/material/predict_material.py:comp_material form_v2=$P/ext_form.py:comp_form series_pool=$P/ext_form.py:series_pool take_t7=$P/ext_take.py:take_t7"
run() { name=$1; shift; ~/venv/bin/python $P/make_submission.py --external $EXT --combo "t7=form_v2+series_pool+material_v2+b+take_t7" --variant t7 --out /home/cleaner/Cleaner/publish/cand_$name.csv "$@" 2>&1 | grep -v -i warn | tail -1 | cut -c1-300; }
run tb_t7_gap1k_s90_fb --two-box 0.0001 --agg 0.90 --agg-gap 1000 --fallback-conf 0.04
run tb_t7_gap3_fb --two-box 0.0001 --agg 0.90 --agg-gap 3 --fallback-conf 0.04
run ob_t7_gap1k_s90_fb --agg 0.90 --agg-gap 1000 --fallback-conf 0.04
