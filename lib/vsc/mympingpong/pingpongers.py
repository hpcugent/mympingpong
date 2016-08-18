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

import copy

import numpy

from mpi4py import MPI
from vsc.utils.missing import get_subclasses


class PingPongSR(object):
    """standard pingpong"""

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

        self.it = None
        self.start = None
        self.end = None

        self.setsr()

        self.setcomm()

    @staticmethod
    def pingpongfactory(pptype, comm, p, log):
        """a factory for creating PingPong objects"""

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

    def setit(self, it, group=None):  # pylint: disable-msg=W0613
        self.it = it
        self.start = numpy.zeros(it, float)
        self.end = numpy.zeros(it, float)

    def dopingpong(self, it=None, group=None):  # pylint: disable-msg=W0613
        if it:
            self.setit(it)

        for x in xrange(self.it):
            self.start[x] = MPI.Wtime()
            self.run1(self.sndbuf, self.other, self.tag1)
            self.run2(self.rcvbuf, self.other, self.tag2)
            self.end[x] = MPI.Wtime()

        return numpy.average((self.end - self.start) / (2.0 * self.group))


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

    def setit(self, itall, group=None):
        it = itall / group
        self.group = group
        self.it = itall
        self.start = numpy.zeros(it, float)
        self.end = numpy.zeros(it, float)

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

        for x in xrange(it / group):
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
                                   self.group)
            self.start[x] = start
            self.end[x] = end

        return numpy.average((self.end - self.start) / (2.0 * self.group))


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

    def dopingpong(self, it=None, group=None):  # pylint: disable-msg=W0613
        for x in xrange(it):
            self.start[x] = MPI.Wtime()
            self.end[x] = MPI.Wtime()
        return self.start, self.end
