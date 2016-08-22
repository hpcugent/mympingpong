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

from vsc.install.shared_setup import action_target, sdw

PACKAGE = {
    'name': 'mympingpong',
    'version': '0.7.1',
    'install_requires': [
        'vsc-base >= 2.4.16',
        'numpy >= 1.8.2',
        'matplotlib >= 1.3.1',
        'lxml',
        'h5py',
        'mpi4py < 2.0.0',  # the patched one to run, for analysis, this is ok (and not used)
    ],
    # Workaround from
    # https://github.com/numpy/numpy/issues/2434#issuecomment-65252402
    # and
    # https://github.com/h5py/h5py/issues/535#issuecomment-79158166
    'setup_requires': [
        'mock',
        'numpy >= 1.8.2',
        'nose',
    ],
    'author': [sdw],
    'maintainer': [sdw],
}

# matplotlib dropped support for python < 2.7 in version 2.0.0
if sys.version_info < (2, 7):
    idx = [i for i, x in enumerate(PACKAGE['install_requires']) if x.startswith('matplotlib')]
    PACKAGE['install_requires'][idx[0]] += ', < 2.0.0'

if __name__ == '__main__':
    action_target(PACKAGE)
