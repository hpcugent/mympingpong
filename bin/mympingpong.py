#!/usr/bin/env python
#
# Copyright 2010-2016 Ghent University
#
# This file is part of mympingpong,
# originally created by the HPC team of Ghent University (http://ugent.be/hpc/en),
# with support of Ghent University (http://ugent.be/hpc),
# the Flemish Supercomputer Centre (VSC) (https://www.vscentrum.be),
# the Flemish Research Foundation (FWO) (http://www.fwo.be/en)
# and the Department of Economy, Science and Innovation (EWI) (http://www.ewi-vlaanderen.be/en).
#
# https://github.com/hpcugent/mympingpong
#
# mympingpong is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation v2.
#
# mympingpong is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with mympingpong.  If not, see <http://www.gnu.org/licenses/>.
#
"""
@author: Stijn De Weirdt (Ghent University)
@author: Jeroen De Clerck (Ghent University)

Pingpong related classes and tests, based on mympi
"""
# this needs to be imported before other loggers or fancylogger won't work
from vsc.utils.generaloption import simple_option

import array
import time
import datetime
import os
import signal
import sys
from itertools import permutations

import h5py
import numpy as n
from mpi4py import MPI

from vsc.mympingpong.pingpongers import PingPongSR
from vsc.mympingpong.pairs import Pair
from vsc.mympingpong.tools import hwlocmap
from vsc.utils.affinity import sched_getaffinity, sched_setaffinity


class MyPingPong(object):

    def __init__(self, logger, it, num):
        self.log = logger

        self.rngfilter = None
        self.mapfilter = None
        self.pairmode = None

        self.fn = None

        self.comm = MPI.COMM_WORLD
        self.name = MPI.Get_processor_name()
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.core = self.setrankaffinity()

        self.it = it
        self.nr = num

        self.outputfile = None

        self.abortsignal = False

        signal.signal(signal.SIGUSR1, self.abort)

    def setfilename(self, directory, msg):
        """generate a filename for the outputfile"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') if self.rank == 0 else None
        timestamp = self.comm.bcast(timestamp, root=0)

        name = self.name if self.rank == 0 else None
        name = self.comm.bcast(name, root=0)

        args = (directory, name, self.size, msg, self.nr, self.it, timestamp)
        self.fn = '%s/PP%s-%03i-msg%07iB-nr%05i-it%05i-%s.h5' % args

    def setpairmode(self, pairmode='shuffle', rngfilter=None, mapfilter=None):
        """set the pairmode, rngfilter and mapfilter for the pairgenerator """
        self.pairmode = pairmode
        self.rngfilter = rngfilter
        self.mapfilter = mapfilter
        self.log.debug("pairmode: pairmode %s rngfilter %s mapfilter %s", pairmode, rngfilter, mapfilter)

    def setrankaffinity(self):
        """pins the rank to an available core on its node"""
        ranknodes = self.comm.alltoall([self.name] * self.size)
        ranksonnode = [i for i, j in enumerate(ranknodes) if j == self.name]

        rankaffinity = sched_getaffinity()
        self.log.debug("affinity pre-set: %s", rankaffinity)

        cores = [i for i, j in enumerate(rankaffinity.cpus) if j == 1L]

        topin = None
        for index, iterrank in enumerate(ranksonnode):
            if iterrank == self.rank:
                topin = cores[index % len(cores)]
                self.log.debug("setting affinity to core: %s", topin)

        if topin is None:
            topin = cores[0]
            self.log.warning("could not determine core to pin the rank to. automatically set it to %s", topin)

        rankaffinity.convert_hr_bits(str(topin))
        rankaffinity.set_bits()
        sched_setaffinity(rankaffinity)

        rankaffinity = sched_getaffinity()
        self.log.debug("affinity post-set: %s", rankaffinity)
        return str(rankaffinity)

    def abort(self, sig, frame):  # pylint: disable-msg=W0613
        """intercepts a SIGUSR1 signal."""
        self.log.warning("received abortsignal on rank %s", self.rank)
        self.abortsignal = True

    def alltoallabort(self, maxruntime, start):
        """
        communicates between all ranks and determines if they can continue or should abort.
        returns True when any rank in the world has received the signal to abort.
        """
        abort = self.abortsignal

        if (maxruntime and (time.time() - start) > maxruntime):
            self.log.warning("maximum runtime was reached on rank %s", self.rank)
            abort = True

        abortlist = [abort] * self.size
        alltoall = self.comm.alltoall(abortlist)
        return any(alltoall)

    def makedata(self, l=1024):
        """create data with size l (in Bytes)"""
        return array.array('c', '\0'*l)

    def makecpumap(self):
        """
        returns information on the processor that is being used for the task

        Returns:
        a list with information of all the ranks in the world, in this format
        MPI processor name, pinned core, [socket-id, core-id, absolute Processor Unit ID of core]
        """

        hwloc = hwlocmap()
        prop = None
        try:
            prop = hwloc[int(self.core)]
        except KeyError as err:
            # it's important to continue, due to alltoall
            # (if one rank has issues, comm should still complete)
            self.log.error("makecpumap: failed to get hwloc info: map %s, err %s", hwloc, err)

        pc = "core_%s" % self.core
        ph = "hwloc_%s" % prop
        self.log.debug("makecpumap: found property core %s hwloc %s", pc, ph)

        myinfo = [self.name, pc, ph]
        mymap = [myinfo] * self.size
        alltoall = self.comm.alltoall(mymap)
        self.log.debug("Received map %s", alltoall)

        return alltoall

    def setup(self, seed, cpumap):
        """
        Set up all variables necessary for running PingPong

        Returns a dictionary with global attributes, a list of pairs for pingponging with and
        a data array for storing outputdata
        """

        if self.nr is None:
            self.nr = int(self.size/2)+1

        if not self.pairmode:
            self.pairmode = 'shuffle'

        if type(seed) == int:
            self.seed = seed
        elif self.pairmode in ['shuffle']:
            self.log.error("Runpingpong in mode shuffle and no seeding: this will never work.")

        try:
            pair = Pair.pairfactory(pairmode=self.pairmode, seed=self.seed,
                                    rng=self.size, pairid=self.rank, logger=self.log)
        except KeyError as err:
            self.log.error("Failed to create pair instance %s: %s", self.pairmode, err)
        pair.setcpumap(cpumap, self.rngfilter, self.mapfilter)
        pair.setnr(self.nr)
        mypairs = pair.makepairs()

        attrs = {
            'pairmode': self.pairmode,
            'totalranks': self.size,
            'nr_tests': self.nr,
            'iterations': self.it,
            'aborted': False,
        }

        if self.nr > (2 * (self.size-1)):
            # the amount of pairs made is greater that the amount of possible combinations
            # therefore, create the keys beforehand to minimize hash collisions
            # possible combinations are the permutations of range(size) that contain rank
            keys = [tup for tup in permutations(range(self.size), 2) if self.rank in tup]
            data = dict.fromkeys(keys, (0, 0))
            self.log.debug("created a datadict from keys: %s", keys)
        else:
            data = dict()
            self.log.debug("created an empty datadict")

        return attrs, mypairs, data

    def run(self, abort_check=True, seed=1, msgsize=1024, maxruntime=0):
        """
        sets up and runs the main test loop

        Arguments:
        abort_check: if True, will check if the test should be aborted before every pingpong (includes maxruntime check)
        seed: a seed for the random number generator used in pairs.py, should be an int.
        msgsize: size of the data that will be sent between pairs
        maxruntime: the maximum amount of time that the test will run. Will abort the main loop if exceeded.
        barrier: if true, wait until every action in a set is finished before starting the next set

        Returns nothing but will pass the following to writehdf5
        attr: a dictionary containing metadata
        data: a dict that maps a pair to the amount of times it has been tested and the sum of its test timings
        fail: a 2D array that contains information on how many times a rank has failed a test
        """
        cpumap = self.makecpumap()
        attrs, mypairs, data = self.setup(seed, cpumap)
        fail = n.zeros((self.size, self.size), int)
        dattosend = self.makedata(l=msgsize)
        pmode = 'fast2'

        self.comm.barrier()
        self.log.debug("run: setup finished")
        start = time.time()

        for runid, pair in enumerate(mypairs):
            self.comm.barrier()
            if abort_check:
                if self.alltoallabort(maxruntime, start):
                    attrs.update({
                        'nr_tests': runid*self.size,
                        'aborted': True,
                    })
                    self.log.info("breaking pingpong loop at runid %s", runid)
                    break
                self.comm.barrier()

            timingdata, group = self.pingpong(pair[0], pair[1], pmode=pmode, dat=dattosend)

            # log progress
            #   log first 10 per iteration,
            #   next 10 per 10 (till 100)
            #   rest only update it when the percentage changes
            logok = False
            if ((runid < 10) or
                (runid < 100 and (runid % 10) == 0) or
                (runid >= 100 and (runid % (self.nr/100) == 0))):
                logok = True

            if self.rank == 0 and logok:
                progress = int(float(runid)*100/self.nr)
                self.log.debug("run %s/%s (%s%%)", runid*self.size, self.nr*self.size, progress)

            key = tuple(pair)
            try:
                count, old_timingdata = data.get(key, (0, 0))
                data[key] = (count + 1, n.append(old_timingdata, timingdata))
            except KeyError as _:
                self.log.error("pair %s is not in permutation", key)
            if (-1 in key) or (-2 in key):
                fail[self.rank][key[n.where(key > -1)[0][0]]] += 1

        for k, (count, timings) in data.items():
            data[k] = (count, n.sum(timings)/count, n.std(timings))
        self.log.debug("finished building data { (p1,p2) : (count, avg, stdev), }: %s", data)

        failed = n.count_nonzero(fail) > 0
        timing = int((time.time() - start))

        attrs.update({
            'msgsize': msgsize,
            'ppmode': pmode,
            'failed': failed,
            'timing': timing,
            'ppgroup': group,
        })

        self.writehdf5(data, attrs, failed, fail)

    def pingpong(self, p1, p2, pmode='fast2', dat=None, dummyfirst=False, test=False):
        """
        Pingpong between pairs

        Arguments:
        p1, p2: pair 1 & 2
        pmode: which pingpongmode is used (fast, fast2, U10) (default: fast2)
        dat: the data that is being sent
        dummyfirst: if true, do a dummyrun before pingponging $it times
        test: use pingpongtest()

        Returns:
        timing: the time that it took to pingpong between 1 and 2 $it times
        group: the group attribute of the pingponger
        """

        if not dat:
            dat = self.makedata()
        if p1 == p2:
            self.log.debug("pingpong: do nothing p1 == p2")
            return -1, {}

        if (p1 == -1) or (p2 == -1):
            self.log.debug("pingpong: do nothing: 0 results in pair (ps: %s p2 %s)", p1, p2)
            return -1, {}
        if (p1 == -2) or (p2 == -2):
            self.log.debug("pingpong: do nothing: result from odd number of elements (ps: %s p2 %s)", p1, p2)
            return -1, {}

        if test:
            pp = PingPongSR.pingpongfactory('test')
        elif self.rank == p1:
            pp = PingPongSR.pingpongfactory('SR' + pmode, self.comm, p2, self.log)
        elif self.rank == p2:
            pp = PingPongSR.pingpongfactory('RS' + pmode, self.comm, p1, self.log)
        else:
            self.log.debug("pingpong: do nothing myrank %s p1 %s p2 %s pmode %s", self.rank, p1, p2, pmode)
            return -1, {}

        pp.setdat(dat)

        if dummyfirst:
            self.log.debug("pingpong: dummy first")
            pp.dopingpong(1)

        timingdata = pp.dopingpong(self.it)
        return timingdata, pp.group

    def writehdf5(self, data, attributes, failed, fail, remove=True):
        """
        writes data to a .h5 defined by the -f parameter

        Arguments:
        data: a 3D matrix containing the data from running pingpong. data[p1][p2][information]
        attrs: a dict containing the attributes of the test
        failed: a boolean that is False if there were no fails during testing
        fail: a 2D array containing information on how many times a rank has failed a test

        will generate a hdf5 file containing all this data plus a dataset containing information on the rank
        """
        filename = self.fn

        if remove and os.path.exists(filename):
            try:
                MPI.File.Delete(filename)
            except Exception as err:
                self.log.error("Failed to delete file %s: %s" % (filename, err))
                filename = "_%s" % filename

        f = h5py.File(filename, 'w', driver='mpio', comm=self.comm)

        for k, v in attributes.items():
            f.attrs[k] = v
            if self.rank == 0:
                self.log.debug("added attribute %s: %s to data.attrs", k, v)

        dataset = f.create_dataset('data', (self.size, self.size, len(data.values()[0])), 'f')
        for ((sendrank, recvrank), val) in data.items():
            if sendrank != self.rank:
                # we only use the timingdata if the current rank is the sender
                continue
            dataset[sendrank, recvrank] = tuple(val)

        if failed:
            failset = f.create_dataset('fail', (self.size, self.size), dtype='i8')
            failset[self.rank] = fail[self.rank]

        STR_LEN = 64
        rankname = f.create_dataset('rankdata', (self.size, 2), dtype='S%s' % str(STR_LEN))
        rankname[self.rank] = (self.fitstr(self.name, STR_LEN), self.fitstr(self.core, STR_LEN))

        f.close()

    def fitstr(self, string, length):
        return '{message: <{fill}}'.format(message=string[0:length], fill=length)

if __name__ == '__main__':

    options = {
        'number': ('set the amount of samples that will be made', int, 'store', None, 'n'),
        'messagesize': ('set the message size in Bytes', int, 'store', 1024, 'm'),
        'iterations': ('set the number of iterations', int, 'store', 20, 'i'),
        'groupmode': ('set the groupmode', str, 'store', None, 'g'),
        'output': ('set the outputdirectory. a file will be written in format \
            PP<name>-<worldssize>-msg<msgsize>-nr<number>-it<iterations>-<ddmmyy-hhmm>.h5', str, 'store', 'test2', 'f'),
        'seed': ('set the seed', int, 'store', 2, 's'),
        'maxruntime': ('set the maximum runtime of pingpong in seconds \
                       (default will run infinitely)', int, 'store', 0, 't'),
        'abort_check': ('check for abort signals or maxruntime', '', 'store_true', True, 'a'),
    }

    go = simple_option(options)

    if not go.options.abort_check and go.options.maxruntime != 0:
        go.log.warning(
            "maxruntime has been set, but abort checks have been disabled, tests wont stop after exceeding maxruntime")

    mpp = MyPingPong(go.log, go.options.iterations, go.options.number)

    if not os.path.isdir(go.options.output):
        go.log.error("could not set outputfile: %s doesn't exist or isn't a path", go.options.output)
        sys.exit(3)
    mpp.setfilename(go.options.output, go.options.messagesize)

    if go.options.groupmode == 'incl':
        mpp.setpairmode(rngfilter=go.options.groupmode)
    elif go.options.groupmode == 'groupexcl':
        mpp.setpairmode(pairmode=go.options.groupmode, rngfilter=go.options.groupmode)
    elif go.options.groupmode == 'hwloc':
        # no rngfilter needed (hardcoded to incl)
        mpp.setpairmode(pairmode=go.options.groupmode)

    mpp.run(abort_check=go.options.abort_check, seed=go.options.seed,
            msgsize=go.options.messagesize, maxruntime=go.options.maxruntime)

    go.log.info("data written to %s", mpp.fn)
