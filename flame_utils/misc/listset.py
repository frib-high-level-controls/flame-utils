#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging


def _flatten(nnn):
    """ flatten recursively defined list or tuple

    :param nnn: recursively defined list or tuple
    :return: a generator object

    :Example:

    >>> l0 = [1,2,3,[4,5],[6,[7,8,[9,10,['x',['y']]]]]]
    >>> l1 = list(_flatten(l0))
    >>> print(l1)
    [1,2,3,4,5,6,7,8,9,10,'x','y']
    """
    for nn in nnn:
        if isinstance(nn, (list, tuple)):
            for n in flatten(nn):
                yield (n)
        else:
            yield (nn)


def flatten(nnn):
    """ flatten recursively defined list or tuple

    :param nnn: recursively defined list or tuple
    :return: flattened list
    
    :Example:

    >>> l0 = [1,2,3,[4,5],[6,[7,8,[9,10,['x',['y']]]]]]
    >>> l1 = flatten(l0)
    >>> print(l1)
    [1,2,3,4,5,6,7,8,9,10,'x','y']
    """
    return list(_flatten(nnn))


def get_intersection(**kws):
    """Get the intersection of all input keyword parameters, ignore
    empty list or tuple.

    Returns
    -------
    res : list
    
    Examples
    --------
    >>> a, b, c = [], [], []
    >>> print(get_intersection(a=a,b=b,c=c))
    []
    >>> a, b, c = [1], [2], []
    >>> print(get_intersection(a=a,b=b,c=c))
    []
    >>> a, b, c = [1,2], [2], []
    >>> print(get_intersection(a=a,b=b,c=c))
    [2]
    >>> a, b, c = [1,2], [2], [2,3]
    >>> print(get_intersection(a=a,b=b,c=c))
    [2]
    >>> a, b, c = [1,2], [3,4], [2,3]
    >>> print(get_intersection(a=a,b=b,c=c))
    []
    """
    s = set()
    for k in kws:
        v = kws.get(k, [])
        if s == set() or v == []:
            s = s.union(v)
        else:
            s = s.intersection(v)
    return list(s)

