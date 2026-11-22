#!/usr/bin/env bash
cd /home/cleaner/Cleaner/audit; export ASTRAMIS_DATA=/home/cleaner/Cleaner/datasets/raw
~/venv/bin/python -W ignore predict_test.py --out /home/cleaner/Cleaner/audit/submission.csv > predict_final.log 2>&1
~/venv/bin/python -W ignore predict_test.py --two-class --out /home/cleaner/Cleaner/audit/submission_twoform.csv > predict_final2.log 2>&1
echo FINALDONE >> predict_final2.log
