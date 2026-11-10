#!/bin/bash
# Re-evaluation of the confirmatory study with calibration metrics. Run from studies/.
PY=${PYTHON:-python}; MAXP=${MAXP:-14}; MINFREE=${MINFREE:-5000}
mkdir -p res/ploteval
while read ds s; do
  tag=$(echo $ds | tr ':' '_'); o=res/ploteval/pe_${tag}_${s}.json; l=res/ploteval/log_${tag}_${s}.txt
  [ -s $o ] && continue
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); n=$(pgrep -fc "p1_plugin_eval.py --ds")
    if [ "$free" -gt "$MINFREE" ] && [ "$n" -lt "$MAXP" ]; then break; fi; sleep 15
  done
  OMP_NUM_THREADS=2 $PY -u p1_plugin_eval.py --ds $ds --seed $s --out $o > $l 2>&1 &
  sleep 10
done < $1
wait; echo done > res/ploteval/DONE
