mympingpong
===========

A mpi4py based random pair pingpong network stress test

TODO
====

 - make code work again
   - test on laptop
   - fix output format, only keep subset of data (what analysis uses to plot)

 - add docstrings, general code cleanup
   - refactor in class per module?
   - unittests
   - use vsc.utils.generaloption and vsc.utils.fancylogger

 - switch to pandas
 - introduce data format for full details, make it optional

Results
=======

The end result of a `mympingponganalysis` is a visual representation of the
pingpong RTT of all the pairs. This can give insight in the architecture and/or topology
(or any issues with it).

There are 3 areas in each image:
 - largest plot: each datapoint is the average pingpong RTT from MPI rank from X-axis initiating pingpong to rank on Y-axis.
 - histogram of all pingpong RTT
 - heatmap of number of pingpongs initiated from MPI rank from X-axis to rank on Y-axis 

## Intra node NUMA
![1 dual socket quad core harpertown (L5420)](/result_images/1node_1024byte_gengar.png)

2 cores share L2 cache; 4 cores per socket

![1 dual socket quad core nehalem (L5520)](/result_images/1node_1024byte_gastly.png)

Much improved cache architecture with nehalem

## Inter+intra nodes: IB vs GigE
![2 dual scoket quad core harpertown (L5420) with DDR IB](/result_images/2nodes_1024byte_gengar.png)

Inter node communication is clearly slower then intra node, but only a factor of 3-5.
The histogram for intranode communication shows the 3 regions:shared L2 cache, on die and inter-die.

![2 dual scoket quad core nehalem (L5520) with GigE](/result_images/2nodes_1024byte_gastly.png)

GigE cleary has issues with latency wrt IB.

## 2 nodes with issue
![2 out of 32 harpertown (L5420) with DDR IB nodes with IB firmware issue](/result_images/32node_firmwareissue_node114_115.png)

32 nodes with 8 mpi tasks per node. 2 nodes (16 tasks in total) are slightly slower.
It's not much, but the colours really make it stand out.
