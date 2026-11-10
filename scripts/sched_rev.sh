#!/bin/bash
PY=${PYTHON:-python}
MAXP=${MAXP:-16}; MINFREE=${MINFREE:-5000}
launch() {
  prog=$1; ds=$2; s=$3; tag=$(echo $ds | tr ':' '_')
  case $prog in
    rev) o=res/revision/rv_${tag}_${s}.json; cmd="p1_revision.py --ds $ds --seed $s --out $o";;
    sen) o=res/revision/sn_${ds}_${s}.json; cmd="p1_sensor.py --seed $s --test $ds --out $o";;
    sev) o=res/revision/severity.json; cmd="p1_severity.py $o";;
    ovh) o=res/revision/overhead.json; cmd="p1_overhead.py $o";;
  esac
  [ -s $o ] && return 0
  pgrep -f -- "$o" > /dev/null && return 0
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    n=$(pgrep -fc "(p1_revision|p1_sensor|p1_severity|p1_overhead).py")
    if [ "$free" -gt "$MINFREE" ] && [ "$n" -lt "$MAXP" ]; then break; fi
    sleep 20
  done
  echo "$(date +%T) start $prog $ds $s" >> res/sched_rev.log
  OMP_NUM_THREADS=2 $PY -u $cmd > ${o%.json}.log 2>&1 &
  sleep 15
}
for pass in 1 2; do
  while read prog ds s; do launch $prog $ds $s; done < $1
  wait
done
echo "$(date +%T) done" >> res/sched_rev.log
