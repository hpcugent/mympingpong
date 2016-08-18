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
Module with general tools

@author: Stijn De Weirdt (Ghent University)
"""
import logging
import os
import tempfile

from lxml.etree import ElementTree as etree
from vsc.utils.run import run_simple


HWLOC_LS = "hwloc-ls"
HWLOC_LS_XML_TEMPLATE = HWLOC_LS + " --force --output-format xml %s"


def hwlocmap():
    """
    Generate and parse output from hwloc-ls

    Returns a dict that maps the absolute Processor Unit ID to its socket-id and its core-id
    """
    # Only need a filename
    (fh, xmlout) = tempfile.mkstemp(prefix="hwloc-xml-", suffix=".xml")
    os.close(fh)

    run_simple(HWLOC_LS_XML_TEMPLATE % xmlout)

    parsed = _parse_hwloc_xml(xmlout)

    os.remove(xmlout)

    return parsed


def _parse_hwloc_xml(xml_fn):
    """
    Generate the mapping between absolute Processor Unit ID to its socket-id and its core-id

    xml_fn is filename for xml file with hwloc-ls output in xml format
    """
    # parse xmloutput
    base = etree().parse(xml_fn).getroottree()

    # gather all interesting elements and their paths
    # SB has numa -> socket -> core -> pu
    # haswell has socket -> numa -> core -> pu
    # broadwell has package -> numa -> core -> pu
    # track elements by unique xpath

    elements = {}
    for typ in ['Package', 'Socket', 'NUMANode', 'Core', 'PU']:
        xpath = './/object[@type="%s"]' % typ
        elements[typ] = dict([(base.getpath(el), int(el.get('os_index', -1))) for el in base.findall(xpath)])

    # there should be either socket or package
    # if package, rename to socket
    if len(elements['Package']):
        elements['Socket'] = elements['Package']

    def find_parent_element(typ, path):
        """
        Look for typ that is parent of path in elements
        log error if more than one is found (it really shouldn't)
        return first value
        """
        # keep p and v for debugging when more than one is found
        found = [(p, v) for p, v in elements[typ].items() if path.startswith(p)]
        if len(found) > 1:
            logging.error("Found more than one %s for child path %s: %s" % (typ, path, found))
        elif len(found) == 0:
            logging.error("Found none %s for child path %s" % (typ, path))
            return None
        # only return value
        return found[0][1]

    res = {}
    for path, pu in elements['PU'].items():
        core = find_parent_element('Core', path)
        socket = find_parent_element('Socket', path)
        numa = find_parent_element('NUMANode', path)
        text = "socket %s core %s abscore %s numa %s" % (socket, core, pu, numa)
        res[pu] = text

    logging.debug("result map: %s", res)
    return res
