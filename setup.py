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
# the Flemish Supercomputer Centre (VSC) (https://www.vscentrum.be),
# the Flemish Research Foundation (FWO) (http://www.fwo.be/en)
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

import sys

from vsc.install.shared_setup import action_target, kh, sdw

PACKAGE = {
    'name': 'mympingpong',
    'version': '0.8.1',
    'install_requires': [
        'vsc-install >= 0.17.1',
        'vsc-base >= 3.1.4',
        'numpy >= 1.8.2',
        'matplotlib >= 1.3.1',
        'lxml',
        'h5py',
        'mpi4py',  # should be patched version, which includes support for timed pingpong
    ],
    'tests_require': [
        'mock',
        'nose',
    ],
    'author': [sdw, kh],
    'maintainer': [sdw, kh],
}

if sys.version_info < (2, 7):
    # matplotlib dropped support for python < 2.7 in version 2.0.0
    idx = [i for i, x in enumerate(PACKAGE['install_requires']) if x.startswith('matplotlib')]
    PACKAGE['install_requires'][idx[0]] += ', < 2.0.0'

    # numpy also dropped support for python < 2.7 in version 1.11.0
    idx = [i for i, x in enumerate(PACKAGE['install_requires']) if x.startswith('numpy')]
    PACKAGE['install_requires'][idx[0]] += ', < 1.11.0'
    idx = [i for i, x in enumerate(PACKAGE['setup_requires']) if x.startswith('numpy')]
    PACKAGE['setup_requires'][idx[0]] += ', < 1.11.0'

if __name__ == '__main__':
    action_target(PACKAGE)
