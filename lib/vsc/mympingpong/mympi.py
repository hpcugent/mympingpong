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

MPI framework based upon mpi4py and some things taken from FZJ linktest

SDW VSC-UGent
- v1 26/04/2010 Looking for a replacement of FZJ linktest

TODO:
 - improved pair selection
  - option to select pure inter/intra node tests
 - combine with hwloc for correct placement of sockets/cores
"""

import sys
import os
import re
import zlib
import cPickle
import array
import logging
import numpy as n


def getshared():
    """
    returns path to a shared directory

    Returns:
    will raise a KeyError if 'VSC_SCRATCH' is not set
    """
    return os.environ['VSC_SCRATCH']


class mympi:

    def __init__(self, nolog=True, serial=False):
        if not nolog:
            self.log = logging.getLogger()

        self.serial = False
        self.pickledelim = "\nPICKLEDELIMITER\n"
        self.fn = None

        if serial:
            self.serial = True
            self.rank = 0
            self.size = 1
        else:
            try:
                global MPI
                from mpi4py import MPI
            except Exception as err:
                self.log.error("Can't load module MPI: %s" % err)

            self.comm = MPI.COMM_WORLD
            self.size = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
            self.name = MPI.Get_processor_name()

        self.master = None
        self.masterrank = 0
        self.seed = None

        # a bit silly with masterraank set here
        self.masterorslave()

    def setfn(self, fn, remove=True):
        self.fn = fn
        if self.master:
            if remove and os.path.exists(self.fn):
                try:
                    MPI.File.Delete(fn)
                except Exception as err:
                    self.log.error("Failed to delete file %s: %s" % (fn, err))

    def setseed(self, seed=None):
        """
        Broadcast the seed
        """
        if not seed:
            seed = 1
        self.comm.bcast(seed, root=self.masterrank)
        self.log.debug("Seed is %s" % seed)
        n.random.seed(seed)
        self.seed = seed

    def helloworld(self):
        txt = "Hello, World! I am process %d of %d on %s.\n" % (
            self.rank, self.size, self.name)
        if self.log:
            self.log.info(txt)
        else:
            sys.stdout.write(txt)

    def masterorslave(self):
        if self.rank == self.masterrank:
            self.master = True
        else:
            self.master = False

    def makedata(self, l=1024L):
        """
        Make some data
        - when passed to python MPI, the pickle headers are added, making it a different size
        """
        """
        old style: send string object. better to use buffer obj like array (See write/read)
        
        dat=''
        for i in xrange(long(l)):
            dat+=chr(i%256)
        """

        dat = array.array('c', '\0'*l)

        return dat

    def timeerror(self):
        """
        Get an idea of timing precision
        """
        

class master(mympi):

    def __init__(self):
        mympi.__init__(self, nolog=False)

    def setfn(self, fn, remove=True):
        self.fn = fn


class slave(mympi):

    def __init__(self):
        mympi.__init__(self, nolog=False)
