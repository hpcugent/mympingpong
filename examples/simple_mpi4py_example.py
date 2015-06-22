#!/usr/bin/env python

from mpi4py import MPI
import numpy as n


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()


send=n.ones(size,int)*rank

recv=comm.alltoall(send)

import time

time.sleep(rank*3)

print rank,send,recv

