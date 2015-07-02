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
@author:Stijn De Weirdt (Ghent University)

30/01/2010 SDW UGent-VSC
logging from easybuild buildLog
- extended for MPI 26/04/2010

TODO: replace any usage with vsc.utils.fancylogger and remove this module completely
"""

import logging
import sys
import os

try:
    from mpi4py import MPI
    suffix = '%s/%s' % (MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size())
except:
    suffix = ''

fm = '%(asctime)s %(name)s %(levelname)s ' + suffix + ' %(message)s '


class myError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class newLog(logging.Logger):
    rais = True

    def error(self, msg, *args, **kwargs):
        newmsg = "err %s %s" % (self.findCaller(), msg)
        logging.Logger.error(self, msg, *args, **kwargs)
        if self.rais:
            raise myError(newmsg)

    def exception(self, msg, *args):
        # don't rasie the exception from within error
        newmsg = "exc %s %s" % (self.findCaller(), msg)
        self.rais = False
        #logging.Logger.error(self,*((msg,) + args), **{'exc_info': 1})
        logging.Logger.exception(self, msg, *args)
        self.rais = True
        if self.rais:
            raise myError(newmsg)

# redirect standard handler of rootlogger to /dev/null
logging.basicConfig(level=logging.ERROR,
                    format=fm,
                    filename='/dev/null'
                    )

logging.setLoggerClass(newLog)

formatter = logging.Formatter(fm)

knownhandlers = []

# new dirty trick for global cross module variable
import __builtin__
try:
    # will do nowthing if it exists
    __builtin__.maindebug
except:
    __builtin__.maindebug = True


def setdebugloglevel(deb):
    __builtin__.maindebug = deb


def initLog(name=None, typ=None):
    if not typ:
        typ = name or 'UNKNOWN'

    log = logging.getLogger(name)

    debug = __builtin__.maindebug

    if debug:
        defaultloglevel = logging.DEBUG
    else:
        defaultloglevel = logging.INFO

    exe = 'hand=logging.StreamHandler(sys.stdout)'
    pref = "name_%s_typ_%s_%s" % (name, typ, exe)
    if not pref in knownhandlers:
        exec(exe)
        hand.setFormatter(formatter)

        log.addHandler(hand)
        knownhandlers.append(pref)

    tmp = logging.getLogger(typ)

    tmp.setLevel(defaultloglevel)

    # init message
    from socket import gethostname
    tmp.debug("Log initialised with name %s host %s debug %s" % (name, gethostname(), debug))

    return tmp


if __name__ == '__main__':
    log = initLog()
    log.info("Test info")

    log.info("Testing debug")
    log.debug("Test debug")

    log.info("Testing error")
    log.error("Test error")
    log.info("Error tested. YOU SHOULD NOT SEE THIS")
