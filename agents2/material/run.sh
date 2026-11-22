#!/usr/bin/env bash
# run.sh <name> <cmd...> : фон, лог <name>.log, маркер <name>.done (в ~/Cleaner/agents2/material)
name=$1; shift
cd /home/cleaner/Cleaner/agents2/material; export ASTRAMIS_DATA=/home/cleaner/Cleaner/datasets/raw
rm -f "$name.done"
nohup bash -c "$(printf '%q ' "$@") > $name.log 2>&1; echo \"exit=\$? \$(date +%H:%M:%S)\" > $name.done" > /dev/null 2>&1 < /dev/null &
disown
echo "started $name pid $!"
