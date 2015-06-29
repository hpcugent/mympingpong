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

from vsc.mympingpong.log import initLog, setdebugloglevel


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
            self.log = initLog(name=self.__class__.__name__)

        try:
            global n
            import numpy as n
        except Exception as err:
            self.log.error("Can't load module numpy: %s" % err)

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

    def zdumps(self, obj):
        """
        From http://stackoverflow.com/questions/695794/more-efficient-way-to-pickle-a-string
        """
        try:
            return zlib.compress(cPickle.dumps(obj, cPickle.HIGHEST_PROTOCOL), 9)
        except Exception as err:
            self.log.error("zdumps: failed dump: %s" % (err))

    def zloads(self, zstr):
        """
        From http://stackoverflow.com/questions/695794/more-efficient-way-to-pickle-a-string
        """
        try:
            return cPickle.loads(zlib.decompress(zstr))
        except Exception as err:
            self.log.error("zloads: failed load: %s" % (err))

    def calcmasteroffset(self, ranks=None):
        """
        guess the header size: max processes?
        >>> len(zdumps(n.random.randn(100)))
        943
        >>> len(zdumps(n.random.randn(1000)))
        7839
        >>> len(zdumps(n.random.randn(10000)))
        76985
        >>> len(zdumps(n.random.randn(100000)))
        768702
        >>> len(zdumps(n.random.randn(1000000)))
        7685367
        >>> len(zdumps(n.random.randn(10000000)))
        76854383

        size=total number of ranks, round up to nearest power of 10, then * 10
        """
        if not ranks:
            ranks = self.size
        ten = 10
        while ranks/ten:
            ten *= 10
        ten *= 10
        if ten < 10000:
            # forced, small amounts have unpredictable sizes (10kb is not much)
            ten = 10000
        self.log.debug("calculated offset for ranks %s: %s" % (ranks, ten))
        return ten

    def write(self, data, fn=None):
        if not fn:
            if not self.fn:
                self.log.error("No filename set.")
            else:
                fn = self.fn

        towrite = self.zdumps(data)
        size = len(towrite)

        send = n.ones(self.size, int)*size

        masteroffset = self.calcmasteroffset()

        # broadcast the size to all
        map = n.array(self.comm.alltoall(send))

        offset = masteroffset + (map[0:self.rank]).sum()

        self.log.debug("write size %s masteroffset %s offset %s map %s" % (
            size, masteroffset, offset, map))

        amode = MPI.MODE_RDWR | MPI.MODE_CREATE

        try:
            fh = MPI.File.Open(self.comm, fn, amode)
            # fh.Write_at(offset,stringtowrite)
            if self.master:
                fh.Seek(0)
                header = self.zdumps(map)
                if len(header) > masteroffset:
                    self.log.error("Masteroffset %s smaller then real header length %s (%s)" % (
                        masteroffset, len(header), header))
                fh.Write(header)

            fh.Seek(offset)
            fh.Write(towrite)

            fh.Close()
        except Exception as err:
            self.log.error("Failed to open file %s: %s" % (fn, err))

        self.log.debug("write data: %s" % (towrite))

    def readbasic(self, fn, offset, bytestoread):
        if self.serial:
            try:
                fh = open(fn, 'r')
                fh.seek(offset)
                txt = fh.read(bytestoread)
                fh.close()
            except Exception as err:
                self.log.error(
                    "Read serial: failed to open file %s: %s" % (fn, err))
        else:
            amode = MPI.MODE_RDONLY
            try:
                fh = MPI.File.Open(self.comm, fn, amode)
                buf = array.array('c', '\0'*bytestoread)
                fh.Seek(offset)
                res = fh.Read_all(buf)
                fh.Close()
                txt = buf.tostring()
            except Exception as err:
                self.log.error(
                    "Read parallel: failed to open file %s: %s" % (fn, err))

        return txt

    def read(self, fn=None):
        if not fn:
            if not self.fn:
                self.log.error("No filename set.")
            else:
                fn = self.fn

        """
        type with buffer interface (eg strings or numpy array)
        - use string array (builtin, not numpy)
        """
        # guess the rank, based on file size?
        rank = 100
        masteroffset = self.calcmasteroffset(rank)

        # read map part + convert
        map = self.zloads(self.readbasic(fn, 0, masteroffset))

        alldata = []
        prev = masteroffset
        if self.serial:
            for l in map:
                # read one by one in array
                alldata.append(self.zloads(self.readbasic(fn, prev, l)))
                prev += l
        else:
            prev += sum(map[:self.rank])
            alldata.append(self.zloads(self.readbasic(fn, prev, l)))

        self.log.debug("Alldata %s" % (alldata))

        return alldata

    # TODO replace with from vsc.utils.run import simple_run
    def runrun(self, cmd, returnout=False):
        import time
        if sys.version_info[1] < 6:
            import popen2
        else:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                import popen2

        try:
            p = popen2.Popen4(cmd)
            p.tochild.close()
            ec = p.poll()
            out = ''
            while ec < 0:
                ec = p.poll()
                # need to read from time to time. otherwise the stdout/stderr
                # buffer gets filled and it all stops working
                out += p.fromchild.read()
                time.sleep(1)

            ec = os.WEXITSTATUS(ec)
            out += p.fromchild.read()
        except Exception as err:
            self.log.error(
                "Something went wrong with forking cmd %s: %s" % (cmd, err))

        if returnout:
            return ec, out
        else:
            if ec > 0:
                self.log.error(
                    "Cmd '%s' failed: exitcode %s, output %s" % (cmd, ec, out))


class master(mympi):

    def __init__(self):
        mympi.__init__(self, nolog=False)

    def setfn(self, fn, remove=True):
        self.fn = fn


class slave(mympi):

    def __init__(self):
        mympi.__init__(self, nolog=False)

if __name__ == '__main__':
    import getopt
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ds")
    except getopt.GetoptError, err:
        print str(err)  # will print something like "option -a not recognized"
        sys.exit(2)

    serial = False
    debug = False
    for o, a in opts:
        if o in ['-s']:
            serial = True
        if o in ['-d']:
            debug = True

    setdebugloglevel(debug)

    m = mympi(serial=serial)

    try:
        fn = os.path.join(getshared(), 'test2')
    except KeyError as err:
        # TODO: replace prints with fancylogger
        print str(err) + 'is not set'
        sys.exit(3)

    m.setfn(fn)

    if serial:
        m.read()
    else:
        data = '%s' % m.rank
        m.write(data*250)
        m.comm.Barrier()
        m.read()
