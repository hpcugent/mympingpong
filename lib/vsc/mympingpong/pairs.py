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

import sys,os,re,copy

import numpy as n
import logging


class pair:
    def __init__(self,rng=None,seed=None,id=None):
        self.log=logging.getLogger()

        self.seed=None

        self.rng=None
        self.origrng=None
        
        self.rngsize=None

        self.map=None
        self.origmap=None
        
        self.revmap=None
        self.nextseed=None

        self.id=None
        
        self.mode=self.__class__.__name__

        if rng:
            self.addrng(rng)
        if type(seed) == int:
            self.setseed(seed)
        if type(id) == int:
            self.addid(id)

        self.offset=0

    def setseed(self,seed=None):
        if type(seed) == int:
            n.random.seed(seed)
            self.seed=seed
            self.nextseed=n.random.random_integers(10000000)
            self.log.debug("Seed is %s. Nextseed is %s"%(self.seed, self.nextseed))
        else:
            self.log.debug("Seed: nothing done: %s (%s)"%(seed,type(seed)))

    def addrng(self,rng,start=0,step=1):
        if type(rng) == int:
            self.rng=range(start,rng,step)
        elif type(rng) == list:
            self.rng=rng[start::step]
        else:
            self.log.error("addrng: rng is neither int or list: %s (%s)"%(rng,type(rng)))
        
        self.rngsize=len(self.rng)
        if not self.origrng:
            ## the first time
            self.origrng=copy.deepcopy(self.rng)
        self.log.debug("addrng: size %s rng %s (srcrng %s %s)"%(self.rngsize,self.rng,rng,type(rng)))

    def filterrng(self):
        if self.rngsize == 0:
            self.rngsize=2
            self.rng=[self.id,-1]
            self.log.info('filterrng: 0 number of rng provided. Adding self %s and -1.'%self.id)
        elif self.rngsize%2 == 1:
            self.rng+=[-2]
            self.rngsize+=1
            self.log.info('filterrng: odd number of rng provided. Adding -1.')

    def addmap(self,map,rngfilter=None,mapfilter=None):
        """
        Add map: rng -> list of features (eg nodename)
        - what to do with ids that have no properties: add them to a default group or not
        """
        if map:
            if not self.origmap:
                ## only first time copy
                self.origmap=copy.deepcopy(map)
        else:
            if self.origmap:
                map=copy.deepcopy(self.origmap)
            else:
                self.log.error("addmap: no map or origmap found")
        
        self.map={}
        if mapfilter:
            self.log.debug("addmap: mapfilter %s"%mapfilter)
            try:
                reg=re.compile(r""+mapfilter)
            except Exception, err:
                self.log.error("addmap: problem with mapfilter %s:%s"%(mapfilter,err))
        for k,els in map.items():
            if type(els) == list:
                newl=els
            else:
                newl=[els]

            self.map[k]=[]
            for el in newl:
                if mapfilter and not reg.search(el): continue
                self.map[k].append(el)
                
        self.log.debug("addmap: map is %s (orig: %s)"%(self.map,map))
        """
        Reverse map
        """
        self.revmap={}
        for k,l in self.map.items():
            for p in l:
                if not self.revmap.has_key(p):
                    self.revmap[p]=[]
                if k in self.revmap[p]:
                    self.log.error("addmap: already found id %s in revmap for property %s: %s"%(k,p,self.revmap))
                else:
                    self.revmap[p].append(k)
        self.log.debug("addmap: revmap is %s"%(self.revmap))

        if not rngfilter:
            return

        """
        Collect relevant ids
        - then either include or exclude them
        - if this id has no property: do nothing at all
        """
        self.log.debug("addmap: rngfilter %s"%rngfilter)
        try:
            props=self.map[self.id]
        except:
            props=[]
            self.log.debug("No props found for id %s"%self.id)
        ids=[]
        for p in props:
            for id in self.revmap[p]:
                if (id in self.rng) and (id not in ids):
                    ids.append(id)
        
        ids.sort()
        self.log.debug("addmap: props %s ids %s"%(props,ids))
        if rngfilter == 'incl':
            ## use only these ids to make pairs
            self.addrng(ids)
        elif rngfilter == 'excl':
            """
            This does not do what it's supposed to
            - better not use it like this
            """
            new=[]
            for id in self.rng:
                if not id in ids:
                    new.append(id)
            if not self.id in new:
                new.append(self.id)
            new.sort()
            
            self.addrng(new)
        elif rngfilter == 'groupexcl':
            ## do nothing
            self.log.debug('addmap: rngfilter %s: do nothing'%rngfilter)
            pass
        else:
            self.log.error('addmap: unknown rngfilter %s'%rngfilter)

    def addid(self,id):
        if type(id) == int:
            self.id=id
            self.log.debug("Id is %s"%id)
        else:
            self.log.error("No valid id given: %s (%s)"%(id, type(id)))

    def addnr(self,nr=None):
        if not nr:
            nr=int(rng/2)+1
        self.nr=nr
        self.log.debug("Number of samples: %s"%nr)
    
    def makepairs(self):
        """
        For a given set of ranks, create pairs
        - shuffle ranks + make pairs 
        - repeat nr times 
        """
        ## new run the filter
        self.filterrng()

        res=n.ones((self.nr,2),int)*-1

        if (type(self.id) == int) and (not self.id in self.rng):
            self.log.debug("makepairs: %s not in list of ranks"%self.id)
            return res
        
        a=n.array(self.rng)
        for i in xrange(self.nr):
            res[i]=self.new(a,i)
    
        self.log.debug("makepairs %s returns\n%s"%(self.id,res.transpose()))
        return res

    def new(self):
        self.log.error("New not implemented for mode %s"%self.mode)

class shift(pair):
    """
    A this moment, this doesn't do a lot
    """
    def new(self,x,iter):
        ## iter as shift
        b=n.roll(x,self.offset+iter).reshape(self.rngsize/2,2)
        try:
            res=b[n.where(b==self.id)[0][0]]
        except Exception, err:
            self.log.error("new: failed to pick element for id %s from %s"%(self.id,b))
        return res

class shuffle(pair):
    def new(self,x,iter):
        n.random.shuffle(x)
        b=x.reshape(self.rngsize/2,2)
        try:
            res=b[n.where(b==self.id)[0][0]]
        except Exception, err:
            self.log.error("new: failed to pick element for id %s from %s"%(self.id,b))
        return res
    
class groupexcl(pair):
    def new(self,x,iter):
        ## reseed deterministically because the run of this function is not equal for all values of self.id
        self.setseed(self.nextseed)

        y=x.copy()
        while y.size > 0:
            n.random.shuffle(y)
            luckyid=y[0]

            try:
                props=self.map[luckyid]
            except:
                props=[]
                self.log.debug("new: No props found for id %s"%luckyid)
            ids=[]
            for p in props:
                for id in self.revmap[p]:
                    if (id in y) and (id not in ids):
                        ids.append(id)
            neww=[]
            for id in y:
                if id == luckyid: continue
                if not id in ids:
                    neww.append(id)
            neww.sort()

            if len(neww) == 0:
                otherluckyid=-1
            else:
                z=n.array(neww)
                n.random.shuffle(z)
                otherluckyid=z[0]

            self.log.debug("new: id %s: Found other luckyid %s for luckyid %s"%(self.id,otherluckyid,luckyid))
            if self.id in [luckyid,otherluckyid]:
                return n.array([luckyid,otherluckyid])
            else:
                for iidd in [luckyid,otherluckyid]:
                    y=n.delete(y,n.where(y==iidd)[0])
            
class hwloc(shuffle):
    def makepairs(self):
        """
        Cycle through all core ids
        - restore origrng and reapply map with new hwloc filter
        - repeat ad nauseam
        
        This assumes that all cpus have same hwloc info
        """
        ## from origmap, get all hwloc values
        hwlocs=[]
        for vs in self.origmap.values():
            for v in vs:
                if v.startswith('hwloc'):
                    hwlocs.append(v)
        hwlocs.sort()
        self.log.debug("makepairs: hwlocs %s"%hwlocs)
        

        res=n.ones((self.nr,2),int)*-1

        if (type(self.id) == int) and (not self.id in self.rng):
            self.log.debug("makepairs: %s not in list of ranks"%self.id)
            return res
        
        hwlocid=0
        
        subgroup=10
        for i in xrange(self.nr/subgroup):
            ## restore rng
            self.rng=copy.deepcopy(self.origrng)
            self.rngsize=len(self.rng)
            ## remap
            self.addmap(None,rngfilter='incl',mapfilter=hwlocs[hwlocid])
            
            self.filterrng()
            
            a=n.array(self.rng)
            for j in xrange(subgroup):
                res[i*subgroup+j]=self.new(a,i*subgroup+j)
                
            hwlocid=(hwlocid+1)%(len(hwlocs))
            
    
        self.log.debug("makepairs %s returns\n%s"%(self.id,res.transpose()))
        return res