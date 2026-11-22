#!/usr/bin/env bash
cd /home/cleaner/Cleaner/sim && . ./env.sh
export BLENDER_THREADS=6 OMP_NUM_THREADS=4
nice -n 10 $PY scenes.py --split train --start 0   --n 100 --poses 8 > logs/render_a.log 2>&1 &
nice -n 10 $PY scenes.py --split train --start 100 --n 100 --poses 8 > logs/render_b.log 2>&1 &
nice -n 10 $PY scenes.py --split val   --start 0   --n 30  --poses 8 > logs/render_v.log 2>&1 &
wait
echo DONE > logs/render.done
