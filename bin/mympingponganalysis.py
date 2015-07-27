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

Generate plots from output from mympingpong.py
"""

import bisect
import sys
from math import sqrt

import h5py
import matplotlib as mp
import matplotlib.pyplot as ppl
import matplotlib.cm as cm
import numpy as n
from matplotlib.colorbar import Colorbar, make_axes

from vsc.utils.generaloption import simple_option


class PingPongAnalysis(object):

    def __init__(self, logger, latencyscale, latencymask):
        self.log = logger

        self.data = None
        self.count = None
        self.fail = None
        self.nodemap = None

        # use multiplication of 10e6 (ie microsec)
        self.scaling = 1e6

        self.meta = None

        self.cmap = None

        self.latencyscale = latencyscale
        self.latencymask = latencymask


    def collecthdf5(self, fn):
        """collects metatags, failures, counters and timingdata from fn"""
        f = h5py.File(fn, 'r')

        self.meta = dict(f.attrs.items())
        self.log.debug("collect meta: %s" % self.meta)     

        if self.meta['failed']:
            self.fail = f['fail'][:]
            self.log.debug("collect fail: %s" % self.fail)

        # http://stackoverflow.com/a/118508
        self.count = n.ma.array(f['data'][...,0])
        self.log.debug("collect count: %s" % self.count)

        data = f['data'][...,1]
        data = data*self.scaling
        data = data/n.where(self.count == 0, 1, self.count)
        self.data = n.ma.array(data)
        self.log.debug("collect data: %s" % data)

        f.close()

    def addtext(self, meta, sub, fig):
        self.log.debug("addtext")

        sub.set_axis_off()

        # build a rectangle in axes coords
        left, width = .1, .9
        bottom, height = .1, .9
        right = left + width
        top = bottom + height

        cols = 3
        tags = self.meta.keys()
        nrmeta = len(tags)
        while nrmeta % cols != 0:
            nrmeta += 1
            tags.append(None)
        layout = n.array(tags).reshape(nrmeta/cols, cols)

        for r in xrange(nrmeta/cols):
            for c in xrange(cols):
                m = layout[r][c]
                if not (m and meta.has_key(m)):
                    continue
                val = meta[m]
                unit = ' sec' if m == 'timing' else ''
                sub.text(left+c*width/cols, bottom+r*height/(nrmeta/cols), "%s: %s%s" %
                         (m, val, unit), horizontalalignment='left', verticalalignment='top', transform=sub.transAxes)

    def addlatency(self, data, sub, fig, latencymask):
        """make and show the main latency graph"""
        maskeddata = n.ma.masked_outside(n.ma.masked_equal(data, 0), latencymask[0], latencymask[1])

        vmin = self.latencyscale[0] if self.latencyscale[0] else maskeddata.min()
        vmax = self.latencyscale[1] if self.latencyscale[1] else maskeddata.max()

        cax = sub.imshow(maskeddata, cmap=self.cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
        fig.colorbar(cax)

        axlim = sub.axis()
        sub.axis(n.append(axlim[0:2], axlim[2::][::-1]))
        sub.set_title(r'Latency ($\mu s$)')

        return vmin, vmax

    def addhistogram(self, data, sub, fig1, vmin, vmax):
        """make and show the histogram"""
        bins = 50

        # filter out zeros and data that is too small or too large to show with the selected scaling
        d = n.ma.masked_outside(data.ravel(),1/self.scaling, 1.0*self.scaling)
        (nn, binedges, patches) = sub.hist(n.ma.compressed(d), bins=bins)

        # We don't want the very first binedge
        binedges = binedges[1:]

        lscale = self.latencyscale
        lmask = self.latencymask

        lscale0_ind = bisect.bisect(binedges,lscale[0])
        lscale1_ind = bisect.bisect(binedges,lscale[1])
        lmask0_ind = bisect.bisect(binedges,lmask[0])
        lmask1_ind = bisect.bisect(binedges,lmask[1])
        vmin_ind = bisect.bisect(binedges,vmin)
        vmax_ind = bisect.bisect(binedges,vmax)

        self.log.debug("got indices: lscale0: %s, lscale1: %s, lmask0: %s, lmask1: %s, vmin: %s, vmax: %s", lscale0_ind, lscale1_ind, lmask0_ind, lmask1_ind, vmin_ind, vmax_ind)

        # color every bin according to its corresponding cmapvalue from the latency graph
        # if the bin is masked or falls outside the cmap interval it is colored grey.
        # if latencyscale has been set, color the bins outside the interval with their corresponding colors instead
        colors = [(0.5,0.5,0.5,1)]*bins

        for i in range(vmax_ind-vmin_ind):
            colors[vmin_ind + i] = self.cmap(1.*i/(vmax_ind-vmin_ind))

        if lscale[0] or lscale[1]:
            begin = lmask[0] if lmask[0] else 0
            begin_ind = bisect.bisect(binedges,begin)
            end = lmask[1] if lmask[1] != sys.maxint else bins
            end_ind = bisect.bisect(binedges,end)
            self.log.debug("got beginning and end: %s, %s",begin,end)

            if begin < lscale[0]:
                colors = [self.cmap(0) if i>=begin_ind and i<lscale0_ind else c for i,c in enumerate(colors)]
            if lscale[1] < end:
                colors = [ self.cmap(1.0) if i>=lscale1_ind and i<end_ind else c for i,c in enumerate(colors)]

        if lmask[0] or lmask[1] != sys.maxint:
            if lmask[0] > vmin:
                colors = [(0.5,0.5,0.5,1) if i<lmask0_ind else c for i,c in enumerate(colors)]
            if vmax > lmask[1]:
                colors = [(0.5,0.5,0.5,1) if i>=lmask1_ind else c for i,c in enumerate(colors)]
 
        for color, patch in zip(colors,patches):
            patch.set_facecolor(color)

    def addsamplesize(self, count, sub, fig):
        self.log.debug("addcount")

        maskedcount = n.ma.masked_where(count == 0, count)
        cax = sub.imshow(maskedcount, cmap=self.cmap, interpolation='nearest', vmin = 0)
        cb = fig.colorbar(cax)

        axlim = sub.axis()
        sub.axis(n.append(axlim[0:2], axlim[2::][::-1]))
        sub.set_title('Pair samples (#)')

    def plot(self, data=None, count=None, meta=None):
        self.log.debug("plot")
        if not data:
            data = self.data
        if not count:
            count = self.count
        if not meta:
            meta = self.meta

        # enable LaTeX processing. Internal mathtext should work fine too
        # mp.rcParams['text.usetex']=True
        mp.rcParams['mathtext.fontset'] = 'custom'

        # set colormap
        self.cmap = ppl.get_cmap('jet')
        self.cmap.set_bad(color='grey', alpha=0.25)

        # scale for ISO Ax
        figscale = sqrt(2)
        # A4: 210 mm width
        # 1 millimeter = 0.0393700787 inch
        mmtoin = 0.0393700787
        figwa4 = 210*mmtoin
        figha4 = figwa4*figscale
        fig1 = ppl.figure(figsize=(figwa4, figha4))
        fig1.show()

        def shrink(rec, s=None):
            if not s:
                s = 0.1
            l, b, w, h = rec

            nl = l+w*s/2
            nb = b+h*s/2
            nw = (1-s)*w
            nh = (1-s)*h

            ans = [nl, nb, nw, nh]
            return ans

        texth = 0.1
        subtext = fig1.add_axes(shrink([0, 1-texth, 1, texth]))
        self.addtext(meta, subtext, fig1)

        datah = 1/figscale
        subdata = fig1.add_axes(shrink([0, 1-texth-datah, 1, datah]))
        vmin, vmax = self.addlatency(data, subdata, fig1, self.latencymask)

        histw = 0.7
        subhist = fig1.add_axes(shrink([0, 0, histw, 1-datah-texth], 0.3), vmin, vmax)
        self.addhistogram(data, subhist, fig1, vmin, vmax)

        subcount = fig1.add_axes(shrink([1-histw, 0, histw, 1-datah-texth], 0.3))
        self.addsamplesize(count, subcount, fig1)

        fig1.canvas.draw()

        ppl.show()


if __name__ == '__main__':

    # dict = {longopt:(help_description,type,action,default_value,shortopt),}
    options = {
        'input': ('set the inputfile', str, 'store', 'test2', 'f'),
        'latencyscale': ('set the minimum and maximum of the latency graph colorscheme',
             'strtuple', 'store', ('0','0'), 's'
             ),
        'latencymask': ('set the interval of the data that should be plotted in the latency graph, so  \
            any datapoints that falls outside this interval will not be plotted. The colorscheme min and max \
            will correspond to respectively the lowest and highest value in the remaining data-array',
            'strtuple', 'store', ('0','2147483647'), 'm'
            ),
    }

    go = simple_option(options)

    lscale = (
        float(go.options.latencyscale[0]),
        float(go.options.latencyscale[1]),
        )

    lmask = (
        float(go.options.latencymask[0]),
        float(go.options.latencymask[1]),
        ) 

    ppa = PingPongAnalysis(go.log, lscale, lmask)
    ppa.collecthdf5(go.options.input)

    ppa.plot()
