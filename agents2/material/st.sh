#!/usr/bin/env bash
cd /home/cleaner/Cleaner/agents2/material
for f in *.done; do [ -f "$f" ] && echo "ГОТОВО  ${f%.done}: $(cat $f)"; done
ps -u cleaner -o pid,etime,args | grep "agents2/material" | grep -v grep | grep python
