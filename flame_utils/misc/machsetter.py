#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np

from flame import Machine

_LOGGER = logging.getLogger(__name__)


def machine_setter(_latfile=None, _machine=None, _handle_name=None):
    """ set flame machine, prefer *_latfile*

    :return: FLAME machine object
    """
    if _latfile is not None:
        try:
            m = Machine(open(_latfile, 'r'))
        except:
            if _machine is None:
                _LOGGER.error("{}: Failed to initialize flame machine".format(
                    _handle_name))
                return None
            else:
                _LOGGER.warning("{}: Failed to initialize flame machine, "
                                "use _machine instead".format(_handle_name))
                m = _machine
    else:
        if _machine is None:
            return None
        else:
            m = _machine
    return m


def is_zeros_states(s):
    """ test if flame machine states is all zeros

    Returns
    -------
    True or False
        True if is all zeros, else False
    """
    return np.alltrue(getattr(s, 'moment0') == np.zeros([7, 1]))
