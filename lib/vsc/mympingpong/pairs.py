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

Classes to generate pairs

TODO:
 - faster generation of random pairs:
  - have each rank generate a set of pairs, using as seed the main seed + the rank
  - exchange (alltoall?) the generated pairs
"""

import copy
import re

import numpy as n

from vsc.utils.missing import get_subclasses


class Pair(object):

    def __init__(self, seed=None, rng=None, pairid=None, logger=None):

        self.log = logger

        self.seed = None
        self.nextseed = None

        self.rng = None
        self.origrng = None

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

    @staticmethod
    def pairfactory(pairmode, seed=None, rng=None, pairid=None, logger=None):
        """A factory for creating Pair objects"""

        logger.debug("in pairfactory with pairmode %s", pairmode)
        for cls in get_subclasses(Pair, include_base_class=True):
            if pairmode == cls.__name__.lower():
                return cls(seed, rng, pairid, logger)
        raise KeyError

    def setseed(self, seed=None):
        """set the seed for n.random"""

        if isinstance(seed, int):
            n.random.seed(seed)
            self.seed = seed
            self.nextseed = n.random.random_integers(10000000)
            self.log.debug("Seed is %s. Nextseed is %s", self.seed, self.nextseed)
        else:
            self.log.debug("Seed: nothing done: %s (%s)", seed, type(seed))

    def setpairid(self, pairid):
        if isinstance(pairid, int):
            self.pairid = pairid
            self.log.debug("pairs: Id is %s" % pairid)
        else:
            self.log.error("No valid id given: %s (%s)", pairid, type(pairid))

    def setnr(self, nr=None):
        if not nr:
            nr = int(self.rng / 2) + 1
        self.nr = nr
        self.log.debug("pairs: Number of samples: %s", nr)

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
            # origrng is empty, so this rng is the original rng
            self.origrng = copy.deepcopy(self.rng)
        self.log.debug("pairs: setrng: size %s rng %s (srcrng %s %s)", len(self.rng), self.rng, rng, type(rng))

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
            self.cpumap = self.applymapfilter(mapfilter)
        else:
            self.cpumap = cpumapin

        # Reverse map
        self.revmap = {}
        for ind, l in enumerate(self.cpumap):
            for p in l:
                if p not in self.revmap:
                    self.revmap[p] = []
                if ind in self.revmap[p]:
                    self.log.error("setcpumap: already found id %s in revmap for property %s: %s", ind, p, self.revmap)
                else:
                    self.revmap[p].append(ind)
        self.log.debug("pairs: setcpumap: revmap is %s", self.revmap)

        if rngfilter:
            self.applyrngfilter(rngfilter)

    def applymapfilter(self, dictin, mapfilter=None):
        """
        filter out the keyvalue pairs in dictin that contain $mapfilter
        mapfilter is currently never used, so this block can be discarded if the feature is scrapped
        """

        dictout = {}

        self.log.debug("pairs: applymapfilter: mapfilter %s" % mapfilter)
        try:
            reg = re.compile(r"" + mapfilter)
        except re.error as err:
            self.log.error("applymapfilter: problem with compiling the regex for mapfilter %s:%s", mapfilter, err)

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

        self.log.debug("pairs: applymapfilter: map is %s (orig: %s)", dictout, dictin)
        return dictout

    def applyrngfilter(self, rngfilter):
        """
        filter rng based on information from cpumap

        incl: only the ids in cpumap are kept in rng
        excl: (not correctly implemented) the ids in cpumap are removed from rng
        groupexcl: do nothing

        """

        self.log.debug("pairs: applyrngfilter: rngfilter %s", rngfilter)
        try:
            props = self.cpumap[self.pairid]
        except KeyError as _:
            props = []
            self.log.debug("pairs: No props found for id %s", self.pairid)

        ids = []
        for p in props:
            for x in self.revmap[p]:
                if (x in self.rng) and (x not in ids):
                    ids.append(x)
        ids.sort()
        self.log.debug("pairs: applyrngfilter: props %s ids %s", props, ids)

        if rngfilter == 'incl':
            # use only these ids to make pairs
            self.setrng(ids)
        elif rngfilter == 'excl':
            self.log.error("attempted to use %s rngfilter, which is not correctly implemented", rngfilter)
            new = []
            for x in self.rng:
                if x not in ids:
                    new.append(x)
            if self.pairid not in new:
                new.append(self.pairid)
            new.sort()

            self.setrng(new)
        elif rngfilter == 'groupexcl':
            # do nothing
            self.log.debug('pairs: applyrngfilter: rngfilter %s: do nothing', rngfilter)
            pass
        else:
            self.log.error('pairs: applyrngfilter: unknown rngfilter %s', rngfilter)

    def makepairs(self):
        """create an nr amount of pairs, using the new() function defined by $pairmode in mympingpong.py"""

        self.filterrng()

        # creates a matrix of minus ones, with height = self.nr and width = 2
        res = n.ones((self.nr, 2), int) * -1

        if isinstance(self.pairid, int) and (self.pairid not in self.rng):
            self.log.debug("pairs: makepairs: %s not in list of ranks", self.pairid)
            return res

        rngarray = n.array(self.rng)
        for i in xrange(self.nr):
            res[i] = self.new(rngarray, i)

        self.log.debug("pairs: makepairs %s returns\n%s", self.pairid, res.transpose())
        return res

    def new(self, rngarray, iteration):  # pylint: disable-msg=W0613
        self.log.error("New not implemented for mode %s", self.mode)


class Shift(Pair):
    """iterate through rng to find the next random number"""

    def new(self, rngarray, iteration):
        # shift through rngarray and convert to a matrix with height = len(self.rng)/2 and width = 2
        b = n.roll(rngarray, self.offset + iteration).reshape(len(self.rng) / 2, 2)

        try:
            # n.where(b == self.pairid)[0] returns a list of indices of the elements in b that equal pairid
            # b[n.where(b == self.pairid)[0][0]] is the first element of b that equals pairid
            res = b[n.where(b == self.pairid)[0][0]]
        except IndexError as _:
            self.log.error("new: failed to pick element for id %s from %s", self.pairid, b)
        return res


class Shuffle(Pair):
    """shuffle rng to find the next random number"""

    def new(self, rngarray, iteration):

        n.random.shuffle(rngarray)

        # convert to matrix with height len(self.rng)/2 and width 2
        b = rngarray.reshape(len(self.rng) / 2, 2)

        try:
            res = b[n.where(b == self.pairid)[0][0]]
        except IndexError as _:
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
            except IndexError as _:
                props = []
                self.log.debug("pairs: new: No props found for id %s", luckyid)
            ids = []
            for p in props:
                for x in self.revmap[p]:
                    if (x in rngarray) and (x not in ids):
                        ids.append(x)
            neww = []
            for x in rngarray:
                if x == luckyid:
                    continue
                if x not in ids:
                    neww.append(x)
            neww.sort()

            if len(neww) == 0:
                otherluckyid = -1
            else:
                z = n.array(neww)
                n.random.shuffle(z)
                otherluckyid = z[0]

            self.log.debug("pairs: new: id %s: Found other luckyid %s for luckyid %s",
                           self.pairid, otherluckyid, luckyid)
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
        self.log.debug("pairs: makepairs: hwlocs %s" % hwlocs)

        res = n.ones((self.nr, 2), int) * -1

        if isinstance(self.pairid, int) and (self.pairid not in self.rng):
            self.log.debug("pairs: makepairs: %s not in list of ranks", self.pairid)
            return res

        hwlocid = 0

        subgroup = 10
        for i in xrange(self.nr / subgroup):
            # restore rng
            self.rng = copy.deepcopy(self.origrng)

            # remap
            self.setcpumap(None, rngfilter='incl', mapfilter=hwlocs[hwlocid])

            self.filterrng()

            a = n.array(self.rng)
            for j in xrange(subgroup):
                res[i * subgroup + j] = self.new(a, i * subgroup + j)

            hwlocid = (hwlocid + 1) % (len(hwlocs))

        self.log.debug("pairs: makepairs %s returns\n%s", self.pairid, res.transpose())
        return res
