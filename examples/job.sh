#!/bin/bash

#PBS -q short
#PBS -m be
#PBS -l nodes=1:ppn=8
#PBS -l walltime=00:25:00

cd $PBS_O_WORKDIR
module load cluster
module load Python/2.6.4-ictce-3.2.1.015.u4

export PYTHONPATH=$PWD:$PYTHONPATH

nrkb=1
#nrkb="1/256"
nrkb="1/4"

nn=4000
ii=100

nn=20
ii=50


nb=$((1024*$nrkb))

n="1n_${nb}_${nn}_${ii}_test"

./mympirun.py python pingpong.py -n $nn -i $ii -m $nb -f ${n}
#./mympirun.py python pingpong.py -n $nn -i $ii -g groupexcl -m $nb -f ${n}_groupexcl
#./mympirun.py python pingpong.py -n $nn -i $ii -g incl -m $nb -f ${n}_incl
#./mympirun.py python pingpong.py -n $nn -i $ii -m $nb -f ${n}_hwloc -g hwloc 
#./mympirun.py python pingpong.py -n $nn -i $ii -m $nb -f ${n}_hwloc -d -g hwloc

