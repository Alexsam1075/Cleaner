#!/usr/bin/env bash
# Список новых predict_*.py / REPORT.md у агентов (новее отметки .last_seen)
cd /home/cleaner/Cleaner/agents2
for d in material volume form; do
  find $d -maxdepth 1 \( -name "predict_*.py" -o -name "REPORT.md" \) -newer integrator/.last_seen 2>/dev/null
done
