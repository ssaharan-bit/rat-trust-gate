#!/bin/bash
PY=${PYTHON:-python}
MAXP=${MAXP:-16}; MINFREE=${MINFREE:-5000}
launch() {
  prog=$1; ds=$2; s=$3; tag=$(echo $ds | tr ':' '_')
  case $prog in
    rv2) o=res/rev2/r2_${tag}_${s}.json; cmd="p1_rev2.py --ds $ds --seed $s --out $o";;
    cal) r=$4; o=res/rev2/cs_${tag}_${s}_${r}.json; cmd="p1_calsize.py --ds $ds --seed $s --rcal $r --out $o";;
  esac
  [ -s $o ] && return 0
  pgrep -f -- "$o" > /dev/null && return 0
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    n=$(pgrep -fc "(p1_rev2|p1_calsize).py")
    if [ "$free" -gt "$MINFREE" ] && [ "$n" -lt "$MAXP" ]; then break; fi
    sleep 20
  done
  echo "$(date +%T) start $prog $ds $s $4" >> res/sched_rev2.log
  OMP_NUM_THREADS=2 $PY -u $cmd > ${o%.json}.log 2>&1 &
  sleep 10
}
for pass in 1 2; do
  while read prog ds s r; do launch $prog $ds $s $r; done < $1
  wait
done
echo "$(date +%T) all tasks done; latency (isolated)" >> res/sched_rev2.log
OMP_NUM_THREADS=8 $PY -u p1_latency.py res/rev2/latency.json > res/rev2/latency.log 2>&1
echo "$(date +%T) done" >> res/sched_rev2.log
