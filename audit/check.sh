#!/usr/bin/env bash
cd /home/cleaner/Cleaner/audit
for f in *.done; do [ -f "$f" ] && echo "ГОТОВО  ${f%.done}: $(cat $f)"; done
ps -u cleaner -o args | grep "venv/bin/python" | grep -v grep | sed -E "s/.*python( -W ignore)? //" | awk "{print \"ИДЁТ    \" \$1}"
