#!/usr/bin/env bash
name=$1; shift
cd /home/cleaner/Cleaner/agents2/volume; export ASTRAMIS_DATA=/home/cleaner/Cleaner/datasets/raw
rm -f "$name.done"
nohup bash -c "\"\$@\" > $name.log 2>&1; echo \"exit=\$? \$(date +%H:%M:%S)\" > $name.done" _ "$@" >/dev/null 2>&1 < /dev/null &
disown
echo "started $name pid $!"
