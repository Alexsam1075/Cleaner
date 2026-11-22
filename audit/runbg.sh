#!/usr/bin/env bash
# runbg.sh <name> <cmd...> : запуск в фоне, лог <name>.log, по завершении маркер <name>.done (код выхода внутри)
name=$1; shift
cd /home/cleaner/Cleaner/audit; export ASTRAMIS_DATA=/home/cleaner/Cleaner/datasets/raw
rm -f "$name.done"
( "$@" > "$name.log" 2>&1; echo "exit=$? $(date +%H:%M:%S)" > "$name.done" ) < /dev/null &
disown
echo "started $name pid $!"
