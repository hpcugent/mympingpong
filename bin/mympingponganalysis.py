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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import numpy as n
from matplotlib.colorbar import Colorbar, make_axes


from vsc.utils.generaloption import simple_option


class PingPongAnalysis(object):

    def __init__(self, logger, latencyscale, latencymask):
        self.log = logger

        self.data = None
        self.count = None
        self.fail = None
        self.consistency = None

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

        self.consistency = n.ma.array(f['data'][...,2])
        self.log.debug("collect consistency: %s" % self.consistency)

        f.close()

    def setticks(self, nrticks, length, sub):
        """make and set evenly spaced ticks for the subplot, that excludes zero and max"""
        ticks = [0] * nrticks
        for i in range(nrticks):
            ticks[i] = round((i+1) * length/ (nrticks+1))

        sub.set_xticks(ticks)
        sub.set_yticks(ticks)

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

    def addlatency(self, data, sub, fig):
        """make and show the main latency graph"""
        maskeddata = n.ma.masked_equal(data, 0)

        if self.latencymask[0] is not None and self.latencymask[1] is not None:
            maskeddata = n.ma.masked_outside(maskeddata, self.latencymask[0], self.latencymask[1])

        vmin = self.latencyscale[0] if self.latencyscale[0] is not None else maskeddata.min()
        vmax = self.latencyscale[1] if self.latencyscale[1] is not None else maskeddata.max()

        cax = sub.imshow(maskeddata, cmap=self.cmap, interpolation='nearest', vmin=vmin, vmax=vmax, origin='lower')
        fig.colorbar(cax)
        self.setticks(7, n.size(data,0), sub)
        sub.set_title(r'Latency ($\mu s$)')

        return vmin, vmax

    def addhistogram(self, data, sub, fig1, vextrema):
        """make and show the histogram"""

        bins = 50 #amount of bins in the histogram. 50 is a good default
        defaultcolor = (0.5,0.5,0.5,1)

        # filter out zeros and data that is too small or too large to show with the selected scaling
        d = n.ma.masked_outside(data.ravel(),1/self.scaling, 1.0*self.scaling)
        (nn, binedges, patches) = sub.hist(n.ma.compressed(d), bins=bins)

        # We don't want the very first binedge
        binedges = binedges[1:]

        lscale = self.latencyscale
        lmask = self.latencymask

        # color every bin according to its corresponding cmapvalue from the latency graph
        # if the bin is masked or falls outside the cmap interval it is colored grey.
        # if latencyscale has been set, color the bins outside the interval with their corresponding colors instead
        vmin_ind = bisect.bisect(binedges,vextrema[0])
        vmax_ind = bisect.bisect(binedges,vextrema[1])
        colorrange = vmax_ind-vmin_ind

        colors = [defaultcolor]*vmin_ind + [self.cmap(1.*i/colorrange) for i in range(colorrange)] + [defaultcolor]*(bins-vmax_ind)

        if lscale[0] is not None or lscale[1] is not None:

            begin = lmask[0] if lmask[0] else 0
            end = lmask[1] if lmask[1] else binedges[-1]
            self.log.debug("got beginning and end: %s, %s",begin,end)

            if begin < lscale[0]:
                begin_ind = bisect.bisect(binedges,begin)                
                lscale0_ind = bisect.bisect(binedges,lscale[0])
                colors = self.overwritecolors(self.cmap(0), colors, begin_ind, lscale0_ind )
            if lscale[1] < end:
                end_ind = bisect.bisect(binedges,end)
                lscale1_ind = bisect.bisect(binedges,lscale[1])
                colors = self.overwritecolors(self.cmap(1.0), colors, lscale1_ind, end_ind )

        if lmask[0] is not None or lmask[1] is not None:
            if lmask[0] > vextrema[0]:
                lmask0_ind = bisect.bisect(binedges,lmask[0])
                colors = self.overwritecolors(defaultcolor, colors, end=lmask0_ind)
            if vextrema[1] > lmask[1]:
                lmask1_ind = bisect.bisect(binedges,lmask[1])
                colors = self.overwritecolors(defaultcolor, colors, begin=lmask1_ind)
 
        for color, patch in zip(colors,patches):
            patch.set_facecolor(color)

    def overwritecolors(self, color, colors, begin=0.0, end=float('inf')):
        """will overwrite all elements in the colors array in interval [begin,end] with color"""
        self.log.debug("overwriting %s to %s with %s",begin,end,color)
        return [color if i>=begin and i<end else c for i,c in enumerate(colors)]    

    def addsamplesize(self, count, sub, fig):
        self.log.debug("add a sample size graph to the plot")

        maskedcount = n.ma.masked_where(count == 0, count)
        cax = sub.imshow(maskedcount, cmap=self.cmap, interpolation='nearest', vmin = 0, origin='lower')
        cb = fig.colorbar(cax)
        self.setticks(3, n.size(count,0), sub)
        sub.set_title('Pair samples (#)')

    def addconsistency(self, consistency, sub, fig):
        self.log.debug("addcount")

        maskedconsistency= n.ma.masked_where(consistency == 0, consistency)
        cax = sub.imshow(maskedconsistency, cmap=self.cmap, interpolation='nearest', vmin = 0, origin='lower')
        cb = fig.colorbar(cax)
        self.setticks(3, n.size(consistency,0), sub)
        sub.set_title('standard deviation')


    def plot(self, latencymask):
        self.log.debug("plot")

        mp.rcParams.update({'font.size': 15})

        # set colormap
        self.cmap = plt.get_cmap('jet')
        self.cmap.set_bad(color='grey', alpha=0.25)

        fig1 = plt.figure(figsize=(32,18), dpi=60)

        gs1 = gridspec.GridSpec(1, 1)
        gs1.update(left=0.02, right=0.54, wspace=0.05)

        vmin, vmax = self.addlatency(self.data, plt.subplot(gs1[:, :]), fig1, latencymask)

        gs2 = gridspec.GridSpec(7, 3)
        gs2.update(left=0.55, right=0.98, wspace=0.1, hspace=0.4)

        ax3 = plt.subplot(gs2[0:1, :])
        self.addtext(self.meta, ax3, fig1)

        ax4 = plt.subplot(gs2[1:3, :])
        self.addhistogram(self.data, ax4, fig1, vmin, vmax)
        ax5 = plt.subplot(gs2[3:5, 0])
        self.addsamplesize(self.count, ax5, fig1)
        ax6 = plt.subplot(gs2[3:5, 1])
        self.addconsistency(self.consistency, ax6, fig1)

        """
        ax7 = plt.subplot(gs2[3:5, 2])
        ax7.set_title("ax7")

        ax8 = plt.subplot(gs2[5:7, 0])
        ax8.set_title("ax8")

        ax9 = plt.subplot(gs2[5:7, 1])
        ax9.set_title("ax9")

        ax10 = plt.subplot(gs2[5:7, 2])
        ax10.set_title("ax10")
        """

        fig1.canvas.draw()

        plt.show()


if __name__ == '__main__':

    # dict = {longopt:(help_description,type,action,default_value,shortopt),}
    options = {
        'input': ('set the inputfile', str, 'store', 'test2', 'f'),
        'latencyscale': ('set the minimum and maximum of the latency graph colorscheme',
             'strtuple', 'store', None, 's'
             ),
        'latencymask': ('set the interval of the data that should be plotted in the latency graph, so'
            'any datapoints that falls outside this interval will not be plotted. The colorscheme min and max'
            'will correspond to respectively the lowest and highest value in the remaining data-array',
            'strtuple', 'store', None, 'm'
            ),
    }

    go = simple_option(options)

    lscale = map(float, go.options.latencyscale) if go.options.latencyscale else (None,None)
    lmask = map(float, go.options.latencymask) if go.options.latencymask else (None,None)

    ppa = PingPongAnalysis(go.log, lscale, lmask)
    ppa.collecthdf5(go.options.input)

    ppa.plot()
