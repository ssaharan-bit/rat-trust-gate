#!/bin/bash
PY=${PYTHON:-python}
MAXP=${MAXP:-16}; MINFREE=${MINFREE:-5000}
launch() {
  prog=$1; ds=$2; s=$3; tag=$(echo $ds | tr ':' '_')
  if [ $prog = plug ]; then o=res/plugconf/pc_${tag}_${s}.json; l=res/plugconf/log_${tag}_${s}.txt; script=p1_plugin_conf.py
  else o=res/betacap/bc_${tag}_${s}.json; l=res/betacap/log_${tag}_${s}.txt; script=beta_cap_dev.py; fi
  [ -s $o ] && return 0
  pgrep -f -- "--out $o" > /dev/null && return 0
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    n=$(pgrep -fc "(p1_plugin_conf|beta_cap_dev).py --ds")
    if [ "$free" -gt "$MINFREE" ] && [ "$n" -lt "$MAXP" ]; then break; fi
    sleep 20
  done
  echo "$(date +%T) start $prog $ds $s" >> res/sched_plug.log
  OMP_NUM_THREADS=2 $PY -u $script --ds $ds --seed $s --out $o > $l 2>&1 &
  sleep 20
}
mkdir -p res/plugconf res/betacap
for pass in 1 2; do
  while read prog ds s; do launch $prog $ds $s; done < $1
  wait
done
echo "$(date +%T) done" >> res/sched_plug.log
