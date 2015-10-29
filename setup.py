#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
# Copyright 2009-2012 Ghent University
# Copyright 2009-2012 Stijn De Weirdt
# Copyright 2012 Andy Georges
#
# This file is part of VSC-tools,
# originally created by the HPC team of Ghent University (http://ugent.be/hpc/en),
# with support of Ghent University (http://ugent.be/hpc),
# the Flemish Supercomputer Centre (VSC) (https://vscentrum.be/nl/en),
# the Hercules foundation (http://www.herculesstichting.be/in_English)
# and the Department of Economy, Science and Innovation (EWI) (http://www.ewi-vlaanderen.be/en).
#
# http://github.com/hpcugent/VSC-tools
#
# VSC-tools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation v2.
#
# VSC-tools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with VSC-tools. If not, see <http://www.gnu.org/licenses/>.
#
"""
Setup for mympingpong
"""
from vsc.install.shared_setup import action_target, sdw, vsc_install_scripts

VERSION = '0.6.1'

PACKAGE = {
    'name': 'mympingpong',
    'provides': 'mympingpong = %s' % VERSION,
    'install_requires': ['vsc-base >= 1.8.6', 'numpy >= 1.8.2', 'matplotlib >= 1.3.1'],
    'version': VERSION,
    'author': [sdw],
    'maintainer': [sdw],
    'packages': ['vsc.mympingpong', 'vsc'],
    'scripts': ['bin/mympingpong.py', 'bin/mympingponganalysis.py'],
    'cmdclass': {
        "install_scripts": vsc_install_scripts,
    },
}

if __name__ == '__main__':
    action_target(PACKAGE, extra_sdist=['shared_setup_mympirun.py'])
