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

Classes to generate pairs 

TODO:
 - faster generation of random pairs: 
  - have each rank generate a set of pairs, using as seed the main seed + the rank
  - exchange (alltoall?) the generated pairs
"""

import sys
import os
import re
import copy

import numpy as n
import logging


class pair(object):

    def __init__(self, rng=None, seed=None, pairid=None):

        self.log = logging.getLogger()   

        self.rng = None
        self.origrng = None

        self.cpumap = None
        self.origmap = None
        self.revmap = None

        self.pairid = None

        self.mode = self.__class__.__name__

        if rng:
            self.setrng(rng)
        if isinstance(pairid, int):
            self.setpairid(pairid)

        self.offset = 0

    def setpairid(self, pairid):
        if isinstance(pairid, int):
            self.pairid = pairid
            self.log.debug("Id is %s" % pairid)
        else:
            self.log.error("No valid id given: %s (%s)", pairid, type(pairid))

    def setnr(self, nr=None):
        if not nr:
            nr = int(rng/2)+1
        self.nr = nr
        self.log.debug("Number of samples: %s", nr)

    def setrng(self, rng, start=0, step=1):
        if isinstance(rng, int):
            self.rng = range(start, rng, step)
        elif isinstance(rng, list):
            self.rng = rng[start::step]
        else:
            self.log.error("setrng: rng is neither int or list: %s (%s)", rng, type(rng))

        if not self.origrng:
            # the first time
            self.origrng = copy.deepcopy(self.rng)
        self.log.debug("setrng: size %s rng %s (srcrng %s %s)", len(self.rng), self.rng, rng, type(rng))

    def filterrng(self):
        """makes sure the length of rng is even"""

        if len(self.rng) == 0:
            self.rng = [self.pairid, -1]
            self.log.info('filterrng: 0 number of rng provided. Adding self %s and -1.', self.pairid)
        elif len(self.rng) % 2 == 1:
            self.rng += [-2]
            self.log.info('filterrng: odd number of rng provided. Adding -2.')

    def setcpumap(self, cpumapin, rngfilter=None, mapfilter=None):

        """
        setcpumap: rng -> list of features (eg nodename)
        - what to do with ids that have no properties: add them to a default group or not
        """

        if cpumapin:
            if not self.origmap:
                self.origmap = copy.deepcopy(cpumapin)
        else:
            if self.origmap:
                cpumapin = copy.deepcopy(self.origmap)
            else:
                self.log.error("setcpumap: no map or origmap found")


        self.cpumap = {}
        if mapfilter:
            self.log.debug("setcpumap: mapfilter %s" % mapfilter)
            try:
                reg = re.compile(r""+mapfilter)
            except Exception as err:
                self.log.error("setcpumap: problem with mapfilter %s:%s", mapfilter, err)
        for k, els in cpumapin.items():
            if isinstance(els, list):
                newl = els
            else:
                newl = [els]

            self.cpumap[k] = []
            for el in newl:
                if mapfilter and not reg.search(el):
                    continue
                self.cpumap[k].append(el)

        self.log.debug("setcpumap: map is %s (orig: %s)", self.cpumap, cpumapin)

        #Reverse map
        self.revmap = {}
        for k, l in self.cpumap.items():
            for p in l:
                if not self.revmap.has_key(p):
                    self.revmap[p] = []
                if k in self.revmap[p]:
                    self.log.error("setcpumap: already found id %s in revmap for property %s: %s", k, p, self.revmap)
                else:
                    self.revmap[p].append(k)
        self.log.debug("setcpumap: revmap is %s", self.revmap)

        if not rngfilter:
            return

        """
        Collect relevant ids
        - then either include or exclude them
        - if this id has no property: do nothing at all
        """
        self.log.debug("setcpumap: rngfilter %s", rngfilter)
        try:
            props = self.cpumap[self.pairid]
        except:
            props = []
            self.log.debug("No props found for id %s", self.pairid)
        ids = []
        for p in props:
            for x in self.revmap[p]:
                if (x in self.rng) and (x not in ids):
                    ids.append(x)

        ids.sort()
        self.log.debug("setcpumap: props %s ids %s", props, ids)
        if rngfilter == 'incl':
            # use only these ids to make pairs
            self.setrng(ids)
        elif rngfilter == 'excl':
            """
            This does not do what it's supposed to
            - better not use it like this
            """
            new = []
            for x in self.rng:
                if not x in ids:
                    new.append(x)
            if not self.pairid in new:
                new.append(self.pairid)
            new.sort()

            self.setrng(new)
        elif rngfilter == 'groupexcl':
            # do nothing
            self.log.debug('setcpumap: rngfilter %s: do nothing', rngfilter)
            pass
        else:
            self.log.error('setcpumap: unknown rngfilter %s', rngfilter)

    def makepairs(self):
        """
        For a given set of ranks, create pairs
        - shuffle ranks + make pairs 
        - repeat nr times 
        """
        # new run the filter
        self.filterrng()

        res = n.ones((self.nr, 2), int)*-1

        if isinstance(self.pairid, int) and (not self.pairid in self.rng):
            self.log.debug("makepairs: %s not in list of ranks", self.pairid)
            return res

        a = n.array(self.rng)
        for i in xrange(self.nr):
            res[i] = self.new(a, i)

        self.log.debug("makepairs %s returns\n%s", self.pairid, res.transpose())
        return res

    def new(self):
        self.log.error("New not implemented for mode %s", self.mode)


class shift(pair):

    """
    A this moment, this doesn't do a lot
    """

    def new(self, x, iteration):
        # iteration as shift
        b = n.roll(x, self.offset+iteration).reshape(len(self.rng)/2, 2)
        try:
            res = b[n.where(b == self.pairid)[0][0]]
        except Exception as err:
            self.log.error("new: failed to pick element for id %s from %s", self.pairid, b)
        return res


class shuffle(pair):

    def new(self, x, iteration):

        #x = array of rng
        n.random.shuffle(x)

        #convert to matrix with height len(self.rng)/2 and width 2
        b = x.reshape(len(self.rng)/2, 2)

        try:
            #n.where(b == self.pairid)[0] returns a list of indices of every element in b that equals pairid
            #b[n.where(b == self.pairid)[0][0]] is the first element of b that equals pairid
            res = b[n.where(b == self.pairid)[0][0]]
        except Exception as err:
            self.log.error("new: failed to pick element for id %s from %s", self.pairid, b)
        return res


class groupexcl(pair):

    def new(self, x, iteration):
        
        y = x.copy()
        while y.size > 0:
            n.random.shuffle(y)
            luckyid = y[0]

            try:
                props = self.cpumap[luckyid]
            except:
                props = []
                self.log.debug("new: No props found for id %s", luckyid)
            ids = []
            for p in props:
                for x in self.revmap[p]:
                    if (x in y) and (x not in ids):
                        ids.append(x)
            neww = []
            for x in y:
                if x == luckyid:
                    continue
                if not x in ids:
                    neww.append(x)
            neww.sort()

            if len(neww) == 0:
                otherluckyid = -1
            else:
                z = n.array(neww)
                n.random.shuffle(z)
                otherluckyid = z[0]

            self.log.debug("new: id %s: Found other luckyid %s for luckyid %s", self.pairid, otherluckyid, luckyid)
            if self.pairid in [luckyid, otherluckyid]:
                return n.array([luckyid, otherluckyid])
            else:
                for iidd in [luckyid, otherluckyid]:
                    y = n.delete(y, n.where(y == iidd)[0])


class hwloc(shuffle):

    def makepairs(self):
        """
        Cycle through all core ids
        - restore origrng and reapply map with new hwloc filter
        - repeat ad nauseam

        This assumes that all cpus have same hwloc info
        """
        # from origmap, get all hwloc values
        hwlocs = []
        for vs in self.origmap.values():
            for v in vs:
                if v.startswith('hwloc'):
                    hwlocs.append(v)
        hwlocs.sort()
        self.log.debug("makepairs: hwlocs %s" % hwlocs)

        res = n.ones((self.nr, 2), int)*-1

        if isinstance(self.pairid, int) and (not self.pairid in self.rng):
            self.log.debug("makepairs: %s not in list of ranks", self.pairid)
            return res

        hwlocid = 0

        subgroup = 10
        for i in xrange(self.nr/subgroup):
            # restore rng
            self.rng = copy.deepcopy(self.origrng)

            # remap
            self.setcpumap(None, rngfilter='incl', mapfilter=hwlocs[hwlocid])

            self.filterrng()

            a = n.array(self.rng)
            for j in xrange(subgroup):
                res[i*subgroup+j] = self.new(a, i*subgroup+j)

            hwlocid = (hwlocid+1) % (len(hwlocs))

        self.log.debug("makepairs %s returns\n%s", self.pairid, res.transpose())
        return res
