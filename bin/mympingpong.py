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

Pingpong related classes and tests, based on mympi

TODO: 
 - factor out the pingpong class in vsc.mympingpong.pingpong
 - refactor mypingpong class in regular main() function
"""
# this needs to be imported before other loggers or fancylogger won't work
from vsc.utils.generaloption import simple_option

import copy
import os
import re
import sys

import numpy as n
from lxml import etree
from mpi4py.MPI import Wtime as wtime

from vsc.mympingpong.mympi import mympi, getshared
import vsc.mympingpong.pairs as pairs
from vsc.mympingpong.pairs import Pair
from vsc.utils.run import run_simple
from vsc.utils.missing import get_subclasses

from logging import getLogger


class PingPongSR(object):

    """standard pingpong"""
    """
    Define real work
    - no status check
    - no receiving obj check 

    when using the high level recv/send, this slows things down 
    - objects need to be pickled, more data is send too

    """

    def __init__(self, comm, other, logger):

        self.log = logger

        self.comm = comm

        self.sndbuf = None
        self.rcvbuf = None

        self.other = other

        self.tag1 = 123
        self.tag2 = 234

        self.groupforce = None
        self.group = 0
        self.builtindummyfirst = False

        self.run1 = None
        self.run2 = None

        self.setsr()

        self.setcomm()

    @staticmethod
    def pingpongfactory(pptype, comm, p, log):
        """a factory for creating PingPong objects"""

        log.debug("in pingpongfactory with pptype: %s", pptype)
        for cls in get_subclasses(PingPongSR, include_base_class=True):
            if "PingPong%s" % pptype == cls.__name__:
                return cls(comm, p, log)
        raise KeyError

    def setsr(self):
        self.send = self.comm.Send
        self.recv = self.comm.Recv

    def setcomm(self):
        self.run1 = self.send
        self.run2 = self.recv

    def setdat(self, dat):
        self.sndbuf = dat
        self.rcvbuf = copy.deepcopy(self.sndbuf)

    def setit(self, it):
        self.it = it
        self.start = n.zeros(it, float)
        self.end = n.zeros(it, float)

    def dopingpong(self, it=None):
        if it:
            self.setit(it)

        for x in xrange(self.it):
            self.start[x] = wtime()
            self.run1(self.sndbuf, self.other, self.tag1)
            self.run2(self.rcvbuf, self.other, self.tag2)
            self.end[x] = wtime()

        avg = n.average((self.end-self.start)/(2.0*group))

        return avg


class PingPongRS(PingPongSR):
    """standard pingpong"""

    def setcomm(self):
        self.run1 = self.recv
        self.run2 = self.send


class PingPongSRfast(PingPongSR):

    def setsr(self):
        """set the send-recieve optimisation """
        self.send = self.comm.PingpongSR
        self.recv = self.comm.PingpongRS

    def setcomm(self):
        self.run1 = self.send

    def setit(self, itall, group):
        it = itall/group
        self.group = group
        self.it = itall
        self.start = n.zeros(it, float)
        self.end = n.zeros(it, float)

    def dopingpong(self, it=None, group=50):
        if self.groupforce:
            group = self.groupforce

        if it < group:
            group = it
            self.log.debug("dopingpong: number of iterations is set to group size")

        if it is not None:
            self.setit(it, group)
        else:
            self.log.debug("dopingpong: number of iterations is not set")

        for x in xrange(it/group):
            """
            Comm.PingpongSR(self, 
                            rbuf, sbuf, 
                            int rsource=0, int sdest=0,  
                            int rtag=0, int stag=0, 
                            int num=1, 
                            Status rstatus=None)
            """
            start, end = self.run1(self.rcvbuf, self.sndbuf,
                                   self.other, self.other,
                                   self.tag1, self.tag2,
                                   group)
            self.start[x] = start
            self.end[x] = end

        avg = n.average((self.end-self.start) / (2.0*group))

        return avg


class PingPongRSfast(PingPongSRfast):

    def setcomm(self):
        self.run1 = self.recv
        # flip tags
        a = self.tag2
        self.tag2 = self.tag1
        self.tag1 = a


class PingPongSRU10(PingPongSRfast):
    """send-receive optimized for pingponging 10 times"""

    def setsr(self):
        self.groupforce = 10
        self.builtindummyfirst = True
        self.send = self.comm.PingpongSRU10
        self.recv = self.comm.PingpongRSU10


class PingPongRSU10(PingPongRSfast):
    """receive-send optimized for pingponging 10 times"""

    def setsr(self):
        self.groupforce = 10
        self.builtindummyfirst = True
        self.send = self.comm.PingpongSRU10
        self.recv = self.comm.PingpongRSU10


class PingPongSRfast2(PingPongSRfast):
    """send-receive optimized for pingponging 25 times in a for loop"""

    def setsr(self):
        self.groupforce = 25
        self.builtindummyfirst = True
        self.send = self.comm.PingpongSR25
        self.recv = self.comm.PingpongRS25


class PingPongRSfast2(PingPongRSfast):
    """receive-send optimized for pingponging 25 times in a for loop"""

    def setsr(self):
        self.groupforce = 25
        self.builtindummyfirst = True
        self.send = self.comm.PingpongSR25
        self.recv = self.comm.PingpongRS25


class PingPongtest(PingPongSR):

    def dopingpong(it):
        for x in xrange(it):
            self.start[x] = wtime()
            self.end[x] = wtime()
        return self.start, self.end


class MyPingPong(mympi):

    def __init__(self, logger):
        mympi.__init__(self, nolog=False, serial=False)
        self.log = logger
        self.rngfilter = None
        self.mapfilter = None
        self.pairmode = None

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


    def makemap(self):
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

        res = {}
        for x in xrange(self.size):
            res[x] = alltoall[x]
        self.log.debug("makemap result: %s", res)
        return res

    def setpairmode(self, pairmode='shuffle', rngfilter=None, mapfilter=None):
        self.pairmode = pairmode
        self.rngfilter = rngfilter
        self.mapfilter = mapfilter
        self.log.debug("pairmode: pairmode %s rngfilter %s mapfilter %s", pairmode, rngfilter, mapfilter)

    def runpingpong(self,seed=None, msgsize=1024, it=20, nr=None, barrier=True):
        """
        makes a list of pairs and calls pingpong on those

        Arguments:
        seed: a seed for the random number generator used in pairs.py, should be an int.
        msgsize: size of the data that will be sent between pairs
        it: amount of times a pair will send and receive from eachother
        nr: the number of pairs that will be made for eqch Processing Unit, in other words the sample size
        barrier: if true, wait until every action in a set is finished before starting the next set

        Returns:
        nothing, but will write a dict to a file defined by the -f parameter.

        myrank: MPI jobrank of the task
        nr_tests: number of tests, given by the -n argument
        totalranks: total amount of MPI jobs
        name: the MPI processor name
        msgsize: the size of a message that is being sent between pairs, given by the -m argument
        iter: the amount of iterations, given by the -i argument
        pairmode: the way that pairs are generated (randomly or 'smart'), partially given by the -g argument (defaulf shuffle)
        #mapfilter: partially defines the way that pairs are generated
        #rngfilter: partially defines the way that pairs are generated
        #ppbarrier: wether or not a barrier is used during the run
        #mycore: the processor unit that is being used for the task
        #myhwloc: the socket id and core id of the aformentioned processor unit
        #pairs: a list of pairs that has been used in the test
        #data: a list of timing data for each pingpong between pairs
        #ppdummyfirst: wether or not a dummyrun is executed before the actual iterations
        ppmode: which pingpongmode is being used
        ppgroup: pingpongs can be bundled in groups, this is the size of those groups
        ppiterations: duplicate of iter
        #ppbuiltindummyfirst: True if the pingpong method uses a built in dummyrun
        """

        # highest precision mode till now. has 25 internal grouped tests
        pmode = 'fast2'
        barrier2 = False

        dattosend = self.makedata(l=msgsize)

        if not nr:
            nr = int(self.size/2)+1

        if not self.pairmode:
            self.pairmode = 'shuffle'
        if type(seed) == int:
            self.setseed(seed)
        elif self.pairmode in ['shuffle']:
            self.log.error("Runpingpong in mode shuffle and no seeding: this will never work.")
 
        cpumap = self.makemap()
        if self.master:
            self.log.info("runpingpong: making map finished")

        res = {
            'myrank': self.rank,
            'nr_tests': nr,
            'totalranks': self.size,
            'name': self.name,
            'msgsize': msgsize,
            'iter': it,
            'pairmode': self.pairmode,
        }

        data = n.zeros(nr, float)

        try:
            pair = Pair.pairfactory(pairmode=self.pairmode, seed=self.seed, rng=self.size, pairid=self.rank, logger=self.log)
        except KeyError as err:
            self.log.error("Failed to create pair instance %s: %s", self.pairmode, err)

        pair.setcpumap(cpumap, self.rngfilter, self.mapfilter)

        pair.setnr(nr)

        mypairs = pair.makepairs()
        if self.master:
            self.log.info("runpingpong: making pairs finished")

        # introduce barrier
        self.comm.barrier()
        self.log.debug("runpingpong: barrier before real start (map + pairs done)")

        for runid, pair in enumerate(mypairs):
            if barrier:
                self.log.debug("runpingpong barrier before pingpong")
                self.comm.barrier()

            timing, pmodedetails = self.pingpong(pair[0], pair[1], pmode, dattosend, it=it)

            if barrier2:
                self.log.debug("runpingpong barrier after pingpong")
                self.comm.barrier()
            data[runid] = timing

        res['pairs'] = mypairs
        res['data'] = data

        # add the details
        res.update(pmodedetails)

        self.write(res)

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

        details = {
            'ppmode': pmode,
            'ppgroup': None,
            'ppiterations': None,
        }

        if not dat:
            dat = self.makedata()
        if p1 == p2:
            self.log.debug("pingpong: do nothing p1 == p2")
            return -1, details

        if (p1 == -1) or (p2 == -1):
            self.log.debug("pingpong: do nothing: 0 results in pair (ps: %s p2 %s)", p1, p2)
            return -1, details
        if (p1 == -2) or (p2 == -2):
            self.log.debug("pingpong: do nothing: result from odd number of elements (ps: %s p2 %s)", p1, p2)
            return -1, details

        if test:
            pp = PingPongSR.pingpongfactory('test')
        elif self.rank == p1:
            pp = PingPongSR.pingpongfactory('SR' + pmode, self.comm, p2, self.log)
        elif self.rank == p2:
            pp = PingPongSR.pingpongfactory('RS' + pmode, self.comm, p1, self.log)
        else:
            self.log.debug("pingpong: do nothing myrank %s p1 %s p2 %s pmode %s", self.rank, p1, p2, pmode)
            return -1, details

        pp.setdat(dat)

        if dummyfirst:
            self.log.debug("pingpong: dummy first")
            pp.dopingpong(1)

        if self.master:
            self.log.info("runpingpong: starting dopingpong")
        timing = float(pp.dopingpong(it))
        if self.master:
            self.log.info("runpingpong: end dopingpong")

        self.log.debug("pingpong p1 %s p2 %s avg %s", p1, p2, timing)

        details.update({
            'ppgroup': pp.group,
            'ppiterations': pp.it,
        })

        return timing, details

if __name__ == '__main__':

    options = {
        'number': ('set the amount of samples that will be made', int, 'store', None, 'n'),
        'messagesize': ('set the message size in Bytes', int, 'store', 1024, 'm'),
        'iterations': ('set the number of iterations', int, 'store', 20, 'i'),
        'groupmode': ('set the groupmode', str, 'store', None, 'g'),
        'output': ('set the outputfile', str, 'store', 'test2', 'f'),
        'seed': ('set the seed', int, 'store', 2, 's'),    
    }

    go = simple_option(options)

    m = MyPingPong(go.log)

    try:
        fn = os.path.join(getshared(), go.options.output)
    except KeyError as err:
        go.log.error("%s is not set", err)
        sys.exit(3)
    m.setfn(fn)

    if go.options.groupmode == 'incl':
        m.setpairmode(rngfilter=go.options.groupmode)
    elif go.options.groupmode == 'groupexcl':
        m.setpairmode(pairmode=go.options.groupmode, rngfilter=go.options.groupmode)
    elif go.options.groupmode == 'hwloc':
        # no rngfilter needed (hardcoded to incl)
        m.setpairmode(pairmode=go.options.groupmode)

    m.runpingpong(seed=go.options.seed, msgsize=go.options.messagesize, it=go.options.iterations, nr=go.options.number)
