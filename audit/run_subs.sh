#!/usr/bin/env bash
cd /home/cleaner/Cleaner/audit; export ASTRAMIS_DATA=/home/cleaner/Cleaner/datasets/raw
P=/home/cleaner/Cleaner/agents2/integrator
EXT="material_v2=/home/cleaner/Cleaner/agents2/material/predict_material.py:comp_material form_v2=$P/ext_form.py:comp_form series_pool=$P/ext_form.py:series_pool take_t1=$P/ext_take.py:take_t1"
run() { name=$1; shift; ~/venv/bin/python $P/make_submission.py --external $EXT --combo "fvp=form_v2+series_pool+material_v2+b" "fvp_t1=form_v2+series_pool+material_v2+b+take_t1" --out /home/cleaner/Cleaner/publish/cand_$name.csv "$@" 2>&1 | grep -v -i warn | tail -1; }
run tb_gap3 --variant fvp --two-box 0.0001
run tb_t1_gap3 --variant fvp_t1 --two-box 0.0001
run tb_t1_gap40_s90 --variant fvp_t1 --two-box 0.0001 --agg 0.90 --agg-gap 40
run tb_t1_gap1k_s90 --variant fvp_t1 --two-box 0.0001 --agg 0.90 --agg-gap 1000
run tb_t1_gap10_s90 --variant fvp_t1 --two-box 0.0001 --agg 0.90 --agg-gap 10
