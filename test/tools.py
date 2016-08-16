#
# Copyright 2016-2016 Ghent University
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
from mock import patch
import os
import tempfile

import vsc.mympingpong.tools
from vsc.install.testing import TestCase

class ToolsTest(TestCase):
    """Test tools"""

    def test_hwlocmap(self):
        """Test hwlocmap"""

        xmlout = "/some/random/filename"
        fh = 123

        with patch('vsc.mympingpong.tools._parse_hwloc_xml', return_value='randomstring') as p_h_x:
            with patch('vsc.mympingpong.tools.run_simple', return_value=(1,2)) as r_s:
                with patch('tempfile.mkstemp', return_value=(fh, xmlout)) as mkstemp:
                    with patch('os.close') as close:
                        with patch('os.remove') as remove:
                            self.assertEqual('randomstring', vsc.mympingpong.tools.hwlocmap(),
                                             msg='hwlocmap returned output from _parse_hwloc_xml')

        # Assert run_simple called and with args
        # Assert _parse_hwloc_xml is called
        r_s.assert_called_with(vsc.mympingpong.tools.HWLOC_LS_XML_TEMPLATE % xmlout)
        p_h_x.assert_called_with(xmlout)
        mkstemp.assert_called_with(prefix="hwloc-xml-", suffix=".xml")
        close.assert_called_with(fh)
        remove.assert_called_with(xmlout)

    def get_xmlout(self, filename):
        """Return parsed hwloc xml output"""
        basedir = os.path.dirname(__file__)
        xmlout = os.path.join(basedir, 'data', filename)
        hmap = vsc.mympingpong.tools._parse_hwloc_xml(xmlout)
        return hmap

    def test_parse_hwloc_xml_sb_seq(self):
        """
        Going to test _parse_hwloc_xml with hwloc from
        sandy bridge with proper bios and sequential numbered cores (numa == socket)
        """
        hmap = self.get_xmlout('sb_hwloc_c8220-1.5-3.el6_5.xml')

        gen_map = {}
        aPU = 0
        for sk in range(2): # 2 socket
            for cr in range(8): # 8 cores per socket
                gen_map[aPU] = "socket %s core %s abscore %s numa %s" % (sk, cr, aPU, sk)
                aPU += 1
        self.assertEqual(hmap, gen_map, msg='SB sequential hwlocmap %s is equal to generated map %s' % (hmap, gen_map))

    def test_parse_hwloc_xml_sb_alt(self):
        """
        Going to test _parse_hwloc_xml with hwloc from
        sandy bridge with alternating numbered cores (numa == socket)
        """
        hmap = self.get_xmlout('sb_hwloc_r720-1.5-3.el6_5.xml')

        gen_map = {}
        for sk in range(2): # 2 socket
            aPU = sk*1
            for cr in range(8): # 8 cores per socket
                gen_map[aPU] = "socket %s core %s abscore %s numa %s" % (sk, cr, aPU, sk)
                aPU += 2
        self.assertEqual(hmap, gen_map, msg='SB alternating hwlocmap %s is equal to generated map %s' % (hmap, gen_map))

    def test_parse_hwloc_xml_haswell_cod(self):
        """
        Going to test _parse_hwloc_xml with hwloc from
        intel haswell with COD and non-sequential numbering
        """
        hmap = self.get_xmlout('haswell_cod_hwloc-1.10.1-2.el7.centos.xml')

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

    def test_parse_hwloc_xml_broadwell(self):
        """
        Going to test _parse_hwloc_xml with hwloc from broadwell
        """
        hmap = self.get_xmlout('broadwell_hwloc-1.11.3-6.el7.centos.x86_64.xml')

        gen_map = {}
        aPU = 0
        for sk in range(2): # 2 socket
            for num in range(2): # 2 numa per socket
                numa = sk*2 + num
                for scr in range(8): # 7 cores per numa domain, but core 7 doesn't exist
                    if scr == 7:
                        continue
                    cr = 8*num + scr
                    gen_map[aPU] = "socket %s core %s abscore %s numa %s" % (sk, cr, aPU, numa)
                    aPU += 1
        self.assertEqual(hmap, gen_map, msg='broadwell hwlocmap %s is equal to generated map %s' % (hmap, gen_map))
