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

## Intra node NUMA
![1 dual socket quad core harpertown (L5420)](/result_images/1node_1024byte_gengar.png)
![1 dual socket quad core nehalem (L5520)](/result_images/1node_1024byte_gastly.png)

## Inter+intra nodes: IB vs GigE
![2 dual scoket quad core harpertown (L5420) with DDR IB](/result_images/2nodes_1024byte_gengar.png)
![2 dual scoket quad core nehalem (L5520) with GigE](/result_images/2nodes_1024byte_gastly.png)

## 2 nodes with issue
![2 out of 32 harpertown (L5420) with DDR IB nodes with IB firmware issue](/result_images/32node_firmwareissue_node114_115.png)

