#
# Copyright 2016-2016 Ghent University
#
# This file is part of mympingpong,
# originally created by the HPC team of Ghent University (http://ugent.be/hpc/en),
# with support of Ghent University (http://ugent.be/hpc),
# the Flemish Supercomputer Centre (VSC) (https://vscentrum.be/nl/en),
# the Hercules foundation (http://www.herculesstichting.be/in_English)
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
import os
import tempfile

import vsc.utils.run
import vsc.mympingpong.tools
from vsc.install.testing import TestCase

# mock the one imported in the tools module, not run_simple itself
runs_orig = vsc.mympingpong.tools.run_simple
_parse_orig = vsc.mympingpong.tools._parse_hwloc_xml


class ToolsTest(TestCase):
    """Test tools"""

    def setUp(self):
        """restore/set orignal functions mocked in test_hwlocmap"""
        vsc.mympingpong.tools.run_simple = runs_orig
        vsc.mympingpong.tools._parse_hwloc_xml = _parse_orig
        super(ToolsTest, self).setUp()

    def test_hwlocmap(self):
        """Test hwlocmap"""

        xmlout = "%s/test.xml.%s" % (tempfile.gettempdir(), os.getpid())

        global called
        called = {'runs': 0, '_parse': 0}

        def runs(cmd):
            global called
            called['runs'] += 1
            print 'runs', called
            self.assertEqual(cmd, vsc.mympingpong.tools.HWLOC_LS_XML_TEMPLATE % xmlout,
                             msg='command %s passed to run_simple')
            return 1, 2 # not relevant, just needs to return 2 things

        def _parse(xml_fn):
            global called
            called['_parse'] += 1
            print '_parse', called
            return 'randomstring'

        vsc.mympingpong.tools.run_simple = runs
        vsc.mympingpong.tools._parse_hwloc_xml = _parse

        self.assertEqual('randomstring', vsc.mympingpong.tools.hwlocmap(),
                         msg='hwlocmap returned output from _parse_hwloc_xml')

        self.assertEqual(called, {'runs': 1, '_parse': 1},
                         msg='simple_run and _parse both called once by hwlocmap, total %s' % called)


    def test_parse_hwloc_xml(self):
        """Going to test _parse_hwloc_xml"""
        basedir = os.path.dirname(__file__)

        # SB with proper bios and sequential numbered cores (numa == socket)
        xmlout = os.path.join(basedir, 'data', 'sb_hwloc_c8220-1.5-3.el6_5.xml')
        hmap = vsc.mympingpong.tools._parse_hwloc_xml(xmlout)

        gen_map = {}
        aPU = 0
        for sk in range(2): # 2 socket
            for cr in range(8): # 8 cores per socket
                gen_map[aPU] = "socket %s core %s abscore %s numa %s" % (sk, cr, aPU, sk)
                aPU += 1
        self.assertEqual(hmap, gen_map, msg='SB sequential hwlocmap %s is equal to generated map %s' % (hmap, gen_map))

        # SB with alternating numbered cores (numa == socket)
        xmlout = os.path.join(basedir, 'data', 'sb_hwloc_r720-1.5-3.el6_5.xml')
        hmap = vsc.mympingpong.tools._parse_hwloc_xml(xmlout)

        gen_map = {}
        for sk in range(2): # 2 socket
            aPU = sk*1
            for cr in range(8): # 8 cores per socket
                gen_map[aPU] = "socket %s core %s abscore %s numa %s" % (sk, cr, aPU, sk)
                aPU += 2
        self.assertEqual(hmap, gen_map, msg='SB alternating hwlocmap %s is equal to generated map %s' % (hmap, gen_map))

        # haswell with cod has non-sequential numbering
        xmlout = os.path.join(basedir, 'data', 'haswell_cod_hwloc-1.10.1-2.el7.centos.xml')
        hmap = vsc.mympingpong.tools._parse_hwloc_xml(xmlout)

        gen_map = {}
        offset = 12 # half the number of cores
        for sk in range(2): # 2 socket
            for num in range(2): # 2 numa per socket
                numa = sk*2 + num
                for dom in range(2):
                    # oh boy
                    # cores in steps of 2, 
                    # there are also no cores 6/7, seems like a binned 16 core
                    # so core id is num*8 instead of num*6
                    cr = num*8 + dom
                    aPU = numa*3 + dom*offset
                    for scr in range(3): # only 3 cores per sequential group
                        gen_map[aPU] = "socket %s core %s abscore %s numa %s" % (sk, cr, aPU, numa)
                        cr += 2
                        aPU += 1
        self.assertEqual(hmap, gen_map, msg='haswell cod hwlocmap %s is equal to generated map %s' % (hmap, gen_map))
