Description
===========

mympingpong is a mpi4py based random pair pingpong network stress test.

Installation
============

We recommend using [EasyBuild][eb_url] to automatically install `mympingpong`
and it's dependencies.

The required steps involve building a patched mpi4py
and a parallel enabled h5py.
Instructions on manual installing these can be found in
manual/install_insructions.


Usage
========

After successfull installation, simply submit a job that runs
```
mympirun mympingpong -f output_dir -i nr_iterations -n nr_tests_per_rank
```

Dependencies
============

(including but not limited to)

 - numpy >= 1.8.2
 - vsc-base >= 1.8.6
 - matplotlib >= 1.3.1
 - h5py >= 2.5.0

Examples
=======

The end result of a `mympingponganalysis` is a visual representation of the
pingpong Round Trip Time (RTT) of all the pairs. This can give insight in the architecture and/or topology
(or any issues with it).

## Example output
![](/result_images/example.png)

each plot graph up to 5 graphs
 - largest plot: each datapoint is the average pingpong RTT between pairs (x,y), where x and y are the MPI ranks.
 - histogram of all pingpong RTT
 - heatmap of number of pingpongs ran on the (x,y) pair
 - heatmap of the standard deviation in the data from running tests on the (x,y) pair
 - if a mask as been used, a histogram of all pingpong RTT in the mask interval

### Every MPI rank on a unique node
![](/result_images/stdev.png)

The result of running pingpong on 128 nodes, with each rank pinned to core 0.
The latency graph clearly shows which ranks are located on the same switch.
The standard deviation graph show that something fishy is going on with ranks 32-48.

### Every MPI rank on a unique core
![](/result_images/cores.png)

The result of running pingpong on 4 nodes with 16 cores per node.
In this example the NUMA nodes are visible. Inter node communication is clearly slower then intra node, but only by a factor of 3-5.
The histogram shows 3 regions:shared L2 cache, on die and inter-die.
On the latency graph the switch is also visible as a greenish shade.

### Oversubscribing
![](/result_images/oversubscribe.png)

The result of running 32 ranks per node on 4 nodes with 16 cores per node

Using PingPong to its fullest potential
======================

You should always take care to have enough samples per pair. In other words, 
the -n parameter should be high enough to ensure every rank has a consistent result. 
A quick way to see if consistent results are achieved, is when the pair samples graph has a deep red color

Knowing that there is a problem might be useful, but you're more than like also going to want to know where the problem is located. 
Information on what rank is pinned to what core on which node is present in the outputfile, but this data is not plotted with mympingponganalysis. Open it with h5dump or any other HDF5 file reader to get access to this data.
