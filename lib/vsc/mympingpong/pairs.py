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


class Pair(object):

    def __init__(self, seed=None, rng=None, pairid=None, logger=None):

        self.log = logger

        self.rng = None
        self.origrng = None

        self.seed=None
        self.nextseed=None

        self.cpumap = None
        self.origmap = None
        self.revmap = None

        self.pairid = None

        self.mode = self.__class__.__name__

        if rng:
            self.setrng(rng)
        if isinstance(seed, int):
            self.setseed(seed)
        if isinstance(pairid, int):
            self.setpairid(pairid)

        self.offset = 0

    def setseed(self,seed=None):
        if type(seed) == int:
            n.random.seed(seed)
            self.seed=seed
            self.nextseed=n.random.random_integers(10000000)
            self.log.debug("Seed is %s. Nextseed is %s"%(self.seed, self.nextseed))
        else:
            self.log.debug("Seed: nothing done: %s (%s)"%(seed,type(seed)))


    def setpairid(self, pairid):
        if isinstance(pairid, int):
            self.pairid = pairid
            self.log.debug("PAIRS: Id is %s" % pairid)
        else:
            self.log.error("No valid id given: %s (%s)", pairid, type(pairid))

    def setnr(self, nr=None):
        if not nr:
            nr = int(rng/2)+1
        self.nr = nr
        self.log.debug("PAIRS: Number of samples: %s", nr)

    def setrng(self, rng, start=0, step=1):
        """
        set self.rng

        Arguments:
        rng: what self.rng will be set to, can be an int or a list
        start: if rng is a list and start is given, slice rng starting from here
        step: if rng is a list and step is given, only write every $step element to self.rng
        """
        if isinstance(rng, int):
            self.rng = range(start, rng, step)
        elif isinstance(rng, list):
            self.rng = rng[start::step]
        else:
            self.log.error("setrng: rng is neither int or list: %s (%s)", rng, type(rng))

        if not self.origrng:
            # the first time
            self.origrng = copy.deepcopy(self.rng)
        self.log.debug("PAIRS: setrng: size %s rng %s (srcrng %s %s)", len(self.rng), self.rng, rng, type(rng))

    def filterrng(self):
        """makes sure rng has an even & nonzero amount of elements"""

        if len(self.rng) == 0:
            self.rng = [self.pairid, -1]
            self.log.info('filterrng: 0 number of rng provided. Adding self %s and -1.', self.pairid)
        elif len(self.rng) % 2 == 1:
            self.rng += [-2]
            self.log.info('filterrng: odd number of rng provided. Adding -2.')

    def setcpumap(self, cpumapin, rngfilter=None, mapfilter=None):
        """set the cpumap and the revmap, apply filters when necessary"""

        if cpumapin:
            if not self.origmap:
                self.origmap = copy.deepcopy(cpumapin)
        else:
            if self.origmap:
                cpumapin = copy.deepcopy(self.origmap)
            else:
                self.log.error("setcpumap: no map or origmap found")

        if mapfilter:
            self.cpumap = applymapfilter(mapfilter)
        else:
            self.cpumap = cpumapin

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
        self.log.debug("PAIRS: setcpumap: revmap is %s", self.revmap)

        if rngfilter:
           self.applyrngfilter(rngfilter) 

    def applymapfilter(self,dictin,mapfilter):
        """
        filter out the keyvalue pairs in dictin that contain $mapfilter
        mapfilter is currently never used, so this block can be discarded if the feature is scrapped
        """

        dictout = {}

        self.log.debug("PAIRS: applymapfilter: mapfilter %s" % mapfilter)
        try:
            reg = re.compile(r""+mapfilter)
        except Exception as err:
            self.log.error("applymapfilter: problem with mapfilter %s:%s", mapfilter, err)

        for k, els in dictin.items():
            if isinstance(els, list):
                newl = els
            else:
                newl = [els]

            dictout[k] = []
            for el in newl:
                if mapfilter and not reg.search(el):
                    continue
                dictout[k].append(el)

        self.log.debug("PAIRS: applymapfilter: map is %s (orig: %s)", dictout, dictin)   
        return dictout 
     
    def applyrngfilter(self,rngfilter):
        """
        Collect relevant ids
        - then either include or exclude them
        - if this id has no property: do nothing at all
        """

        self.log.debug("PAIRS: applyrngfilter: rngfilter %s", rngfilter)
        try:
            props = self.cpumap[self.pairid]
        except:
            props = []
            self.log.debug("PAIRS: No props found for id %s", self.pairid)

        ids = []
        for p in props:
            for x in self.revmap[p]:
                if (x in self.rng) and (x not in ids):
                    ids.append(x)
        ids.sort()
        self.log.debug("PAIRS: applyrngfilter: props %s ids %s", props, ids)

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
            self.log.debug('PAIRS: applyrngfilter: rngfilter %s: do nothing', rngfilter)
            pass
        else:
            self.log.error('PAIRS: applyrngfilter: unknown rngfilter %s', rngfilter)

    def makepairs(self):
        """
        For a given set of ranks, create pairs
        - shuffle ranks + make pairs 
        - repeat nr times 
        """
        self.filterrng()

        #creates a matrix of minus ones, with height = self.nr and width = 2
        res = n.ones((self.nr, 2), int)*-1

        if isinstance(self.pairid, int) and (not self.pairid in self.rng):
            self.log.debug("PAIRS: makepairs: %s not in list of ranks", self.pairid)
            return res

        rngarray = n.array(self.rng)
        for i in xrange(self.nr):
            res[i] = self.new(rngarray, i)

        self.log.debug("PAIRS: makepairs %s returns\n%s", self.pairid, res.transpose())
        return res

    def new(self):
        self.log.error("New not implemented for mode %s", self.mode)


class Shift(Pair):

    """
    A this moment, this doesn't do a lot
    """

    def new(self, rngarray, iteration):
        # iteration as shift
        b = n.roll(rngarray, self.offset+iteration).reshape(len(self.rng)/2, 2)
        try:
            res = b[n.where(b == self.pairid)[0][0]]
        except Exception as err:
            self.log.error("new: failed to pick element for id %s from %s", self.pairid, b)
        return res


class Shuffle(Pair):

    def new(self, rngarray, iteration):

        n.random.shuffle(rngarray)

        #convert to matrix with height len(self.rng)/2 and width 2
        b = rngarray.reshape(len(self.rng)/2, 2)

        try:
            #n.where(b == self.pairid)[0] returns a list of indices of the elements in b that equal pairid
            #b[n.where(b == self.pairid)[0][0]] is the first element of b that equals pairid
            res = b[n.where(b == self.pairid)[0][0]]
        except Exception as err:
            self.log.error("new: failed to pick element for id %s from %s", self.pairid, b)
        return res


class Groupexcl(Pair):

    def new(self, rngar, iteration):

        # reseed deterministically because the run of this function is not equal for all values of self.id
        self.setseed(self.nextseed)
        
        rngarray = rngar.copy()
        while rngarray.size > 0:
            n.random.shuffle(rngarray)
            luckyid = rngarray[0]

            try:
                props = self.cpumap[luckyid]
            except:
                props = []
                self.log.debug("PAIRS: new: No props found for id %s", luckyid)
            ids = []
            for p in props:
                for x in self.revmap[p]:
                    if (x in rngarray) and (x not in ids):
                        ids.append(x)
            neww = []
            for x in rngarray:
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

            self.log.debug(
                "PAIRS: new: id %s: Found other luckyid %s for luckyid %s", self.pairid, otherluckyid, luckyid)
            if self.pairid in [luckyid, otherluckyid]:
                return n.array([luckyid, otherluckyid])
            else:
                for iidd in [luckyid, otherluckyid]:
                    rngarray = n.delete(rngarray, n.where(rngarray == iidd)[0])


class Hwloc(Shuffle):

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
        self.log.debug("PAIRS: makepairs: hwlocs %s" % hwlocs)

        res = n.ones((self.nr, 2), int)*-1

        if isinstance(self.pairid, int) and (not self.pairid in self.rng):
            self.log.debug("PAIRS: makepairs: %s not in list of ranks", self.pairid)
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

        self.log.debug("PAIRS: makepairs %s returns\n%s", self.pairid, res.transpose())
        return res
