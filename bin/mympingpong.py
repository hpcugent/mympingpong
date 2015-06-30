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
 - factor out the pingpong class in vsc.mympinpong.pingpong
 - refactor mypingpong class in regular main() function
 - remove exec usage

"""

import sys
import os
import re
from lxml import etree

from vsc.utils.run import run_simple

try:
    from vsc.mympingpong.mympi import mympi, getshared
except Exception, err:
    print "Can't load mympi: %s" % err
    sys.exit(1)

import numpy as n

from vsc.mympingpong.log import initLog, setdebugloglevel


class pingpong_sr:

    """
    Define real work
    - no status check
    - no receiving obj check 

    when using the high level recv/send, this slows things down 
    - objects need to be pickled, more data is send too

    """

    def __init__(self, comm, other):

        self.comm = comm
        try:
            global wtime
            from mpi4py.MPI import Wtime as wtime
        except Exception, err:
            pass

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

    def setsr(self):
        self.send = self.comm.Send
        self.recv = self.comm.Recv

    def setcomm(self):
        self.run1 = self.send
        self.run2 = self.recv

    def setdat(self, dat):
        self.sndbuf = dat
        # make a copy of the receivebuffer
        import copy
        self.rcvbuf = copy.deepcopy(self.sndbuf)

    def setnumber(self, number):
        self.number = number
        self.start = n.zeros(number, float)
        self.end = n.zeros(number, float)

    def dopingpong(self, number=None):
        if number:
            self.setnumber(number)

        # python float are double
        for x in xrange(self.number):
            self.start[x] = wtime()
            self.run1(self.sndbuf, self.other, self.tag1)
            self.run2(self.rcvbuf, self.other, self.tag2)
            self.end[x] = wtime()

        avg = n.average((self.end-self.start)/(2.0*group))

        return avg, self.start, self.end


class pingpong_rs(pingpong_sr):

    def setcomm(self):
        self.run1 = self.recv
        self.run2 = self.send


class pingpong_sr_fast(pingpong_sr):

    def setsr(self):
        # use faster pingpong
        self.send = self.comm.PingpongSR
        self.recv = self.comm.PingpongRS

    def setcomm(self):
        self.run1 = self.send

    def setnumber(self, numberall, group):
        number = numberall/group
        self.group = group
        self.number = numberall
        self.start = n.zeros(number, float)
        self.end = n.zeros(number, float)

    def dopingpong(self, number=None, group=50):
        if self.groupforce:
            group = self.groupforce

        if number < group:
            group = number

        if number:
            self.setnumber(number, group)

        # python float are double
        for x in xrange(number/group):
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

        avg = n.average((self.end-self.start)/(2.0*group))

        return avg, self.start, self.end


class pingpong_rs_fast(pingpong_sr_fast):

    def setcomm(self):
        self.run1 = self.recv
        # flip tags
        a = self.tag2
        self.tag2 = self.tag1
        self.tag1 = a


class pingpong_sr_fastu10(pingpong_sr_fast):

    def setsr(self):
        # use faster pingpong
        self.groupforce = 10
        self.builtindummyfirst = True
        self.send = self.comm.PingpongSRU10
        self.recv = self.comm.PingpongRSU10


class pingpong_rs_fastu10(pingpong_rs_fast):

    def setsr(self):
        # use faster pingpong
        self.groupforce = 10
        self.builtindummyfirst = True
        self.send = self.comm.PingpongSRU10
        self.recv = self.comm.PingpongRSU10


class pingpong_sr_fast2(pingpong_sr_fast):

    def setsr(self):
        # use faster pingpong
        self.groupforce = 25
        self.builtindummyfirst = True
        self.send = self.comm.PingpongSR25
        self.recv = self.comm.PingpongRS25


class pingpong_rs_fast2(pingpong_rs_fast):

    def setsr(self):
        # use faster pingpong
        self.groupforce = 25
        self.builtindummyfirst = True
        self.send = self.comm.PingpongSR25
        self.recv = self.comm.PingpongRS25


class pingpongtest(pingpong_sr):

    def dopingpong(number):
        # python float are double
        for x in xrange(number):
            self.start[x] = wtime()
            self.end[x] = wtime()
        return self.start, self.end


class mypingpong(mympi):

    def __init__(self):
        mympi.__init__(self, nolog=False, serial=False)
        self.rngfilter = None
        self.mapfilter = None
        self.pairmode = None

    def getprocinfo(self):
        # which cpus am i on?
        try:
            mypid = os.getpid()
        except Exception, err:
            self.log.error("Can't obtain current process id: %s" % err)

        # get taskset
        cmd = "taskset -c -p %s" % mypid
        ec, out = self.runrun(cmd, True)
        regproc = re.compile(r"\s+(\d+)\s*$")
        r = regproc.search(out)
        if r:
            myproc = r.group(1)
            self.log.debug(
                "getprocinfo: found proc %s taskset: %s" % (myproc, out))
        else:
            self.log.error(
                "No single proc found. Was pinning enabled? (taskset: %s)" % out)

        hwlocmap = self.hwlocmap()

        try:
            prop = hwlocmap[int(myproc)]
        except Exception, err:
            self.log.error(
                "getprocinfo: failed to get hwloc info: map %s, err %s" % (hwlocmap, err))

        pc = "core_%s" % myproc
        ph = "hwloc_%s" % prop
        self.log.debug(
            "getprocinfo: found property core %s hwloc %s" % (pc, ph))

        return pc, ph

    def hwlocmap(self):
        res = {}
        xmlout = "/tmp/test.xml.%s" % os.getpid()
        exe = "/usr/bin/hwloc-ls"
        if not os.path.exists(exe):
            self.log.error("hwlocmap: Can't find exe %s" % exe)

        cmd = "%s --output-format xml %s" % (exe, xmlout)
        ec, txt = self.runrun(cmd, True)

        ## parse xmloutput
        base = etree.parse(xmlout)

        sks_xpath = '//object[@type="Socket"]'
        #list of socket ids
        sks = map(int, base.xpath(sks_xpath + '/@os_index'))
        self.log.debug("sockets: %s" %sks)

        aPU = 0

        for x in xrange(len(sks)):
            cr_xpath = sks_xpath + '[@os_index="' + str(x) + '"]' + '//object[@type="Core"]'
            #list of core ids in socket x
            crs = map(int, base.xpath(cr_xpath + '/@os_index'))
            self.log.debug("cores: %s" %crs)

            for y in xrange(len(crs)):
                pu_xpath = cr_xpath + '[@os_index="' + str(y) + '"]//object[@type="PU"]'
                #list of PU ids in core y from socket x
                pus = map(int, base.xpath(pu_xpath + '/@os_index'))
                self.log.debug("PU's: %s" %pus)

                # absolute PU id = (socket id * cores per socket * PU's in core) + PU id
                for z in xrange(len(pus)):
                    #in case of errors, revert back to this
                    #aPU = sks[x] * len(crs) * len(pus) + pus[z]
                    t = "socket %s core %s abscore %s" % (sks[x], crs[y], aPU)
                    res[aPU] = t
                    aPU += 1

        self.log.debug("result map: %s"%res)

        return res

    def makemap(self):
        pc, ph = self.getprocinfo()

        myinfo = [self.name, pc, ph]
        mymap = [myinfo for x in xrange(self.size)]
        map = self.comm.alltoall(mymap)
        self.log.debug("Received map %s" % map)

        res = {}
        for x in xrange(self.size):
            res[x] = map[x]
        return res

    def setpairmode(self, pairmode='shuffle', rngfilter=None, mapfilter=None):
        self.pairmode = pairmode
        self.rngfilter = rngfilter
        self.mapfilter = mapfilter
        self.log.debug("pairmode: pairmode %s rngfilter %s mapfilter %s" % (
            pairmode, rngfilter, mapfilter))

    def runpingpong(self, seed=None, msgsize=1024, iter=None, nr=None, barrier=True):
        """Run PingPong

        Arguments:
        seed: a seed for the random number generator, should be an int.
        msgsize: size of the data that will be sent between pairs
        iter: amount of times a pair will send and receive from eachother
        nr: 
        barrier: if true, wait until every action in a set is finished before starting the next set

        Returns:
        nothing, but will write a dict to a file defined by the -f parameter.

        myrank: MPI jobrank of the task
        nr_tests: number of tests, given by the -n argument
        totalranks: total amount of MPI jobs
        name: the MPI processor name
        msgsize: the size of a message that is being sent between pairs, given by the -m argument
        iter: the amount of iterations, given by the -i argument
        pairmode: the way that pairs are grouped together (randomly or 'smart'), given by the -g argument
        mapfilter: partially defines the way that pairs are grouped together
        rngfilter: partially defines the way that pairs are grouped together
        ppbarrier: wether or not a barrier is used during the run
        mycore: the processor unit that is being used for the task
        myhwloc: the socket id and core id of the aformentioned processor unit
        pairs: a list of pairs that has been used in the test
        data: a list of timing data for each pingpong between pairs
        ppdummyfirst: wether or not a dummyrun is executed before the actual iterations
        ppmode: which pingpongmode is being used
        ppgroup:
        ppnumber:
        ppbuiltindummyfirst:
        """

        # highest precision mode till now. has 25 internal grouped tests
        pmode = 'fast2'
        barrier2 = False

        dattosend = self.makedata(l=msgsize)
        if not nr:
            nr = int(self.size/2)+1
        if not iter:
            iter = 20

        if not self.pairmode:
            self.pairmode = 'shuffle'
        if type(seed) == int:
            self.setseed(seed)
        elif self.pairmode in ['shuffle']:
            self.log.error(
                "Runpingpong in mode shuffle and no seeding: this will never work.")

        map = self.makemap()
        self.log.debug( "runpingpong map: %s" %map )
        if self.master:
            self.log.info("runpingpong: making map finished")

        res = {'myrank': self.rank,
               'nr_tests': nr,
               'totalranks': self.size,
               'name': self.name,
               'msgsize': msgsize,
               'iter': iter,
               'pairmode': self.pairmode,
               'mapfilter': self.mapfilter,
               'rngfilter': self.rngfilter,
               'ppbarrier': barrier,
               'mycore': map[self.rank][1],
               'myhwloc': map[self.rank][2],
               }

        data = n.zeros((nr, 3), float)

        import vsc.mympingpong.pairs as pairs

        exe = "pair=pairs.%s(seed=self.seed,rng=self.size,id=self.rank)" % self.pairmode
        try:
            # TODO: discover this via getchildren approach
            exec(exe)
        except Exception, err:
            self.log.error(
                "Failed to create pair instance %s: %s" % (pairmode, err))

        # pair.addmap(map,'incl','^hwloc')
        # pair.addmap(map,'incl','^(a|b)')
        # pair.addmap(map,'excl','^(a|b)')
        # pair.addmap(map,'groupexcl','^(a|b)')
        pair.addmap(map, self.rngfilter, self.mapfilter)

        pair.addnr(nr)

        mypairs = pair.makepairs()
        if self.master:
            self.log.info("runpingpong: making pairs finished")

        # introduce barrier
        self.comm.barrier()
        self.log.debug(
            "runpingpong: barrier before real start (map + pairs done)")

        runid = 0
        for pair in mypairs:
            if barrier:
                self.log.debug("runpingpong barrier before pingpong")
                self.comm.barrier()

            timing, pmodedetails = self.pingpong(
                pair[0], pair[1], pmode, dattosend, iter=iter)

            if barrier2:
                self.log.debug("runpingpong barrier after pingpong")
                self.comm.barrier()
            data[runid] = timing
            runid += 1

        res['pairs'] = mypairs
        res['data'] = data

        # add the details
        res.update(pmodedetails)

        self.write(res)

    def pingpong(self, p1, p2, pmode='fast2', dat=None, iter=20, barrier=True, dummyfirst=False, test=False):
        """Pingpong between pairs

        Arguments:
        p1: pair 1
        p2: pair 2
        pmode: which pingpongmode is used eg. fastu10, fast2 (default: fast2)
        dat: the data that is being sent
        iter: amount of pingpongs between p1 & p2 (default: 20)
        barrier: if true, wait until every action in a set is finished before starting the next set
        dummyfirst: if true, do a dummyrun before pingponging $iter times
        test: use pingpongtest()

        Returns:
        timing: a list wich contains an average time, a starttime and an endtime
        details: a dictionary with the pp.group, pp.number and pp.builtindummyfirst
        """

        details = {'ppdummyfirst': dummyfirst,
                   'ppmode': pmode,
                   'ppgroup': None,
                   'ppnumber': None,
                   'ppbuiltindummyfirst': None
                   }

        if not dat:
            dat = self.makedata()
        if p1 == p2:
            self.log.debug("pingpong: do nothing p1 == p2")
            return -1, details

        if (p1 == -1) or (p2 == -1):
            self.log.debug(
                "pingpong: do nothing: 0 results in pair (ps: %s p2 %s)" % (p1, p2))
            return -1, details
        if (p1 == -2) or (p2 == -2):
            self.log.debug(
                "pingpong: do nothing: result from odd number of elements (ps: %s p2 %s)" % (p1, p2))
            return -1, details

        if test:
            exe = 'pp=pingpongtest()'
        elif self.rank == p1:
            exe = 'pp=pingpong_sr_%s(self.comm,p2)' % pmode
        elif self.rank == p2:
            exe = 'pp=pingpong_rs_%s(self.comm,p1)' % pmode
        else:
            self.log.debug("pingpong: do nothing myrank %s p1 %s p2 %s pmode %s" % (
                self.rank, p1, p2, pmode))
            return -1, details

        try:
            exec(exe)
        except Exception, err:
            self.log.error("Can't make instance of pingpong in mode %s (test: %s): %s : %s" % (
                pmode, test, exe, err))

        pp.setdat(dat)

        if dummyfirst:
            self.log.debug("pingpong: dummy first")
            pp.dopingpong(1)

        if self.master:
            self.log.info("runpingpong: starting dopingpong")
        avg, start, end = pp.dopingpong(iter)
        if self.master:
            self.log.info("runpingpong: end dopingpong")

        timing = [float(avg), float(start[0]), float(end[0])]

        self.log.debug("pingpong p1 %s p2 %s avg/start/end %s" %
                       (p1, p2, timing))

        details.update({'ppgroup': pp.group,
                        'ppnumber': pp.number,
                        'ppbuiltindummyfirst': pp.builtindummyfirst
                        })

        return timing, details

if __name__ == '__main__':
    import getopt
    try:
        opts, args = getopt.getopt(sys.argv[1:], "dm:i:n:g:f:")
    except getopt.GetoptError, err:
        print str(err)  # will print something like "option -a not recognized"
        sys.exit(2)

    msgs = 1024
    nr = None
    iter = None
    debug = False
    group = None
    fnb = 'test2'
    for o, a in opts:
        if o in ['-n']:
            nr = int(a)
        if o in ['-m']:
            msgs = int(a)
        if o in ['-i']:
            iter = int(a)
        if o in ['-g']:
            group = a
        if o in ['-f']:
            fnb = a
        if o in ['-d']:
            debug = True

    setdebugloglevel(debug)

    m = mypingpong()

    try:
        fn = os.path.join(getshared(), fnb)
    except KeyError as err:
        print >> sys.stderr, str(err) + 'is not set, this should point to a shared directory.'
        sys.exit(3)

    m.setfn(fn)
    if group == 'incl':
        m.setpairmode(rngfilter=group)
    elif group == 'groupexcl':
        m.setpairmode(pairmode=group, rngfilter=group)
    elif group == 'hwloc':
        # no rngfilter needed (hradcoded to incl)
        m.setpairmode(pairmode=group)

    seed = 2

    m.runpingpong(seed=seed, msgsize=msgs, iter=iter, nr=nr)
