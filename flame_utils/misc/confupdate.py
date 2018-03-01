#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from flame import Machine

import numpy as np

_LOGGER = logging.getLogger(__name__)


def conf_update(machine):
    """Update machine conf() by using current settings.

    :return: FLAME machine object
    """

    m = machine
    try:
        mconf = m.conf()
    except:
        _LOGGER.error("Failed to load FLAME machine object.")
        return None
    
    mc_src = m.conf(m.find(type='source')[0])
    
    for i in range(len(m)):
        elem_i = m.conf(i)
        ename, etype = elem_i['name'], elem_i['type']
        ki = elem_i.keys()
        elem_k = set(ki).difference(mc_src.keys())
        if etype == 'source':
            elem_k.add('vector_variable')
            elem_k.add('matrix_variable')
        if etype == 'stripper':
            elem_k.add('IonChargeStates')
            elem_k.add('NCharge')
        
        for k in elem_k:
            mconf['elements'][i][k] = m.conf(i)[k]
    
    new_m = Machine(mconf)
    
    return new_m


