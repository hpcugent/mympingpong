#!/usr/bin/env python
##
# Copyright 2010-2015 Ghent University
#
# This file is part of mympingpong,
# originally created by the HPC team of Ghent University (http://ugent.be/hpc/en),
# with support of Ghent University (http://ugent.be/hpc),
# the Flemish Supercomputer Centre (VSC) (https://vscentrum.be/nl/en),
# the Hercules foundation (http://www.herculesstichting.be/in_English)
# and the Department of Economy, Science and Innovation (EWI) (http://www.ewi-vlaanderen.be/en).
#
# http://github.ugent.be/hpcugent/mympingpong
#
# mympingpong is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation v2.
#
# mympingpong is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with mympingpong. If not, see <http://www.gnu.org/licenses/>.
##

"""
@author: Stijn De Weirdt (Ghent University)
@author: Jeroen De Clerck (Ghent University)

Pingpong related classes and tests, based on mympi

TODO: 
 - factor out the pingpong class in vsc.mympingpong.pingpong
 - refactor mypingpong class in regular main() function
"""
# this needs to be imported before other loggers or fancylogger won't work
from vsc.utils.generaloption import simple_option

import array
import time
import datetime
import os
import re
import sys
from itertools import permutations

import h5py
import numpy as n
from lxml import etree
from mpi4py import MPI

from vsc.mympingpong.pingpongers import PingPongSR
from vsc.mympingpong.pairs import Pair
from vsc.utils.run import run_simple


class MyPingPong(object):

    def __init__(self, logger):
        self.log = logger

        self.rngfilter = None
        self.mapfilter = None
        self.pairmode = None

        self.fn = None

        self.comm = MPI.COMM_WORLD
        self.name = MPI.Get_processor_name()
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

    def setfn(self, directory, it, nr, msg, remove=True):
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.fn = '%s/PP%s-%03i-msg%07iB-nr%05i-it%05i-%s.h5' % (directory, self.name, self.size, msg,nr, it, timestamp)
        if remove and os.path.exists(self.fn):
            try:
                MPI.File.Delete(self.fn)
            except Exception as err:
                self.log.error("Failed to delete file %s: %s" % (self.fn, err))

    def setpairmode(self, pairmode='shuffle', rngfilter=None, mapfilter=None):
        self.pairmode = pairmode
        self.rngfilter = rngfilter
        self.mapfilter = mapfilter
        self.log.debug("pairmode: pairmode %s rngfilter %s mapfilter %s", pairmode, rngfilter, mapfilter)

    def makedata(self, l=1024L):
        """create data with size l (in Bytes)"""
        return array.array('c', '\0'*l)

    def getprocinfo(self):
        """
        returns information on the processor that is being used for the task

        Arguments:
        None

        Returns:
        pc: the current Processor Unit
        ph: its socket-id and core-id (output from hwlocmap())
        """

        try:
            mypid = os.getpid()
        except OSError as err:
            self.log.error("Can't obtain current process id: %s", err)

        cmd = "taskset -c -p %s" % mypid
        ec, out = run_simple(cmd)
        regproc = re.compile(r"\s+(\d+)\s*$")
        r = regproc.search(out)
        if r:
            myproc = r.group(1)
            self.log.debug("getprocinfo: found proc %s taskset: %s", myproc, out)
        else:
            self.log.error("No single proc found. Was pinning enabled? (taskset: %s)", out)

        hwlocmap = self.hwlocmap()
        prop = None

        try:
            prop = hwlocmap[int(myproc)]
        except KeyError as err:
            self.log.error("getprocinfo: failed to get hwloc info: map %s, err %s", hwlocmap, err)

        pc = "core_%s" % myproc
        ph = "hwloc_%s" % prop
        self.log.debug("getprocinfo: found property core %s hwloc %s", pc, ph)

        return pc, ph

    def hwlocmap(self):
        """parse and return output from hwloc-ls
        Arguments:
        None
        Returns:
        A dict that maps the absolute Processor Unit ID to its socket-id and its core-id
        """

        res = {}
        xmlout = "/tmp/test.xml.%s" % os.getpid()
        exe = "/usr/bin/hwloc-ls"
        if not os.path.exists(exe):
            self.log.error("hwlocmap: Can't find exe %s", exe)

        cmd = "%s --output-format xml %s" % (exe, xmlout)
        ec, txt = run_simple(cmd)

        # parse xmloutput
        base = etree.parse(xmlout)

        sks_xpath = '//object[@type="Socket"]'
        # list of socket ids
        sks = map(int, base.xpath(sks_xpath + '/@os_index'))
        self.log.debug("sockets: %s", sks)

        aPU = 0

        for x, sk in enumerate(sks):
            cr_xpath = '%s[@os_index="%s"]//object[@type="Core"]' % (sks_xpath, x)
            # list of core ids in socket x
            crs = map(int, base.xpath('%s/@os_index' % cr_xpath))
            self.log.debug("cores: %s", crs)

            for y, cr in enumerate(crs):
                pu_xpath = '%s[@os_index="%s"]//object[@type="PU"]' % (cr_xpath, y)
                # list of PU ids in core y from socket x
                pus = map(int, base.xpath('%s/@os_index' % pu_xpath))
                self.log.debug("PU's: %s", pus)

                # absolute PU id = (socket id * cores per socket * PU's in core) + PU id
                # in case of errors, revert back to this
                # aPU = sks[x] * len(crs) * len(pus) + pus[z]
                for pu in pus:
                    t = "socket %s core %s abscore %s" % (sk, cr, aPU)
                    res[aPU] = t
                    aPU += 1

        self.log.debug("result map: %s", res)

        return res

    def makecpumap(self):
        """
        returns the internal structure of the machine

        Arguments:
        None

        Returns:
        a list with all the processor units on the Machine, in this format
        'hostname', 'Processor Unit name', [socket-id, core-id, absolute Processor Unit ID]
        """

        pc, ph = self.getprocinfo()
        myinfo = [self.name, pc, ph]
        mymap = [myinfo] * self.size
        alltoall = self.comm.alltoall(mymap)
        self.log.debug("Received map %s", alltoall)
        return alltoall

    def runpingpong(self, seed=1, msgsize=1024, it=20, nr=None, maxruntime=0, barrier=True, barrier2=False):
        """
        makes a list of pairs and calls pingpong on those

        Arguments:
        seed: a seed for the random number generator used in pairs.py, should be an int.
        msgsize: size of the data that will be sent between pairs
        it: amount of times a pair will send and receive from eachother
        nr: the number of pairs that will be made for each Processing Unit, in other words the sample size
        barrier: if true, wait until every action in a set is finished before starting the next set

        Returns:
        nothing, but will the following to writehdf5(): 

        myrank: MPI jobrank of the task
        nr_tests: number of pairs made, given by the -n argument
        totalranks: total amount of MPI jobs
        name: the MPI processor name
        msgsize: the size of a message that is being sent between pairs, given by the -m argument
        iter: the amount of iterations, given by the -i argument
        pairmode: the way that pairs are generated (randomly or 'smart'), 
                  partially given by the -g argument (defaulf shuffle)
        ppmode: which pingpongmode is being used
        ppgroup: pingpongs can be bundled in groups, this is the size of those groups
        ppiterations: duplicate of iter
        timing: the total runtime of pingpong (does not include setup)

        data: a dict that maps a pair to the amount of times it has been tested and the sum of its test timings
        
        fail: a 2D array that contains information on how many times a rank has failed a test
        """

        if not nr:
            nr = int(self.size/2)+1

        if not self.pairmode:
            self.pairmode = 'shuffle'
        if type(seed) == int:
            self.seed = seed
        elif self.pairmode in ['shuffle']:
            self.log.error("Runpingpong in mode shuffle and no seeding: this will never work.")
 
        cpumap = self.makecpumap()

        try:
            pair = Pair.pairfactory(pairmode=self.pairmode, seed=self.seed, 
                                    rng=self.size, pairid=self.rank, logger=self.log)
        except KeyError as err:
            self.log.error("Failed to create pair instance %s: %s", self.pairmode, err)
        pair.setcpumap(cpumap, self.rngfilter, self.mapfilter)
        pair.setnr(nr)
        mypairs = pair.makepairs()

        if nr > (2 * (self.size-1)):
            # the amount of pairs made is greater that the amount of possible combinations
            # therefore, create the keys beforehand to minimize hash collisions
            # possible combinations are the permutations of range(size) that contain rank
            keys = [tup for tup in permutations(range(self.size), 2) if self.rank in tup]
            data = dict.fromkeys(keys, (0,0))
            self.log.debug("created a datadict from keys: %s", keys)
        else:
            data = dict()
            self.log.debug("created an empty datadict")
        fail = n.zeros((self.size, self.size), int)

        # introduce barrier
        self.comm.barrier()
        self.log.debug("runpingpong: barrier before real start (map + pairs done)")
        start = time.time()

        pmode = 'fast2'
        dattosend = self.makedata(l=msgsize)
        for runid, pair in enumerate(mypairs):
            if maxruntime and (time.time() - start) > maxruntime:
                self.log.info("Maximum runtime %s reached", maxruntime)
                break

            if barrier:
                self.comm.barrier()

            timing, pmodedetails = self.pingpong(pair[0], pair[1], pmode=pmode, dat=dattosend, it=it)

            if barrier2:
                self.comm.barrier()

            key = tuple(pair)
            try:
                count, old_timing = data.get(key, (0, 0))
                data[key] = (count + 1, old_timing + timing)
            except KeyError as err:
                self.log.error("pair %s is not in permutation", key)
            if (-1 in key) or (-2 in key):
                fail[self.rank][key[n.where(key > -1)[0][0]]] += 1

        failed = n.count_nonzero(fail) > 0
        timing = round((time.time() - start))

        attrs = {
            'pairmode': self.pairmode,
            'totalranks': self.size,
            'name': self.name,
            'nr_tests': nr,
            'msgsize': msgsize,
            'iter': it,
            'pingpongmode' : pmode,
            'failed' : failed,
            'timing' : timing,
        }

        if not failed:
            attrs.update(pmodedetails)

        self.log.debug("bool pmodedetails: %s", bool(pmodedetails))
        self.writehdf5(data, attrs, failed, fail)  

    def pingpong(self, p1, p2, pmode='fast2', dat=None, it=20, barrier=True, dummyfirst=False, test=False):
        """
        Pingpong between pairs

        Arguments:
        p1: pair 1
        p2: pair 2
        pmode: which pingpongmode is used (fast, fast2, U10) (default: fast2)
        dat: the data that is being sent
        it: amount of pingpongs between p1 & p2 (default: 20)
        barrier: if true, wait until every action in a set is finished before starting the next set
        dummyfirst: if true, do a dummyrun before pingponging $it times
        test: use pingpongtest()

        Returns:
        timing: a list wich contains an average time, a starttime and an endtime
        details: a dictionary with the pp.group, pp.number and pp.builtindummyfirst
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

        timing = float(pp.dopingpong(it))
        self.log.debug("%s->%s: %s", p1, p2, timing)

        details = {
            'ppgroup': pp.group,
            'ppiterations': pp.it,
        }

        return timing, details

    def writehdf5(self, data, attributes, failed, fail):
        """
        writes data to a .h5 defined by the -f parameter

        Arguments:
        data: a 3D matrix containing the data from running pingpong. data[p1][p2][information]
        attrs: a dict containing the attributes of the test
        failed: a boolean that is False if there were no fails during testing
        fail: a 2D array containing information on how many times a rank has failed a test
        """
        try:
            f = h5py.File(self.fn, 'w', driver='mpio', comm=self.comm)
        except IOError as err:
            self.log.error("Something went wrong while opening the parallel outputfile! %s", err)
            sys.exit(4)

        for k,v in attributes.items():
            f.attrs[k] = v
            self.log.debug("added attribute %s: %s to data.attrs", k, v)

        dataset = f.create_dataset('data', (self.size,self.size,len(data.values()[0])), 'f')
        for ind, ((sendrank,recvrank),val) in enumerate(data.items()):
            if sendrank != self.rank:
                # we only use the timingdata if the current rank is the sender
                continue
            dataset[sendrank,recvrank] = tuple(val)

        if failed:
            failset = f.create_dataset('fail', (self.size,self.size), dtype='i8')
            failset[self.rank] = fail[self.rank]

        f.close()

if __name__ == '__main__':

    options = {
        'number': ('set the amount of samples that will be made', int, 'store', None, 'n'),
        'messagesize': ('set the message size in Bytes', int, 'store', 1024, 'm'),
        'iterations': ('set the number of iterations', int, 'store', 20, 'i'),
        'groupmode': ('set the groupmode', str, 'store', None, 'g'),
        'output': ('set the outputdirectory. a file will be written in format \
            PP<name>-<worldssize>-msg<msgsize>-nr<number>-it<iterations>-<ddmmyy-hhmm>.h5', str, 'store', 'test2', 'f'),
        'seed': ('set the seed', int, 'store', 2, 's'),
        'maxruntime': ('set the maximum runtime of pingpong in seconds', int, 'store', 0, 't'),
    }

    go = simple_option(options)

    m = MyPingPong(go.log)

    if not os.path.isdir(go.options.output):
        go.log.error("could not set outputfile: %s doesn't exist or isn't a path", go.options.output)
        sys.exit(3)
    m.setfn(go.options.output, go.options.iterations, go.options.number, go.options.messagesize)

    if go.options.groupmode == 'incl':
        m.setpairmode(rngfilter=go.options.groupmode)
    elif go.options.groupmode == 'groupexcl':
        m.setpairmode(pairmode=go.options.groupmode, rngfilter=go.options.groupmode)
    elif go.options.groupmode == 'hwloc':
        # no rngfilter needed (hardcoded to incl)
        m.setpairmode(pairmode=go.options.groupmode)

    m.runpingpong(seed=go.options.seed, msgsize=go.options.messagesize, 
                  it=go.options.iterations, nr=go.options.number, maxruntime=go.options.maxruntime)

    go.log.info("data written to %s", go.options.output)
