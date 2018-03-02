#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Operations about FLAME machine elements, lattice is a special element.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import re
import flame

from collections import Counter

from flame_utils.misc import machine_setter
from flame_utils.misc import flatten
from flame_utils.misc import get_intersection
from flame_utils.misc import conf_update

__authors__ = "Tong Zhang"
__copyright__ = "(c) 2016-2017, Facility for Rare Isotope beams, " \
                "Michigan State University"
__contact__ = "Tong Zhang <zhangt@frib.msu.edu>"

_LOGGER = logging.getLogger(__name__)

STRIPPER_PROP_KEYS = ['IonChargeStates', 'NCharge']
SOURCE_PROP_KEYS = ['IonEk', 'IonEs', 'NCharge', 'IonChargeStates'] # field names, not including S[i], P[i]


def get_all_types(latfile=None, _machine=None):
    """Get all unique types from a FLAME machine or lattice file.

    Parameters
    ----------
    latfile:
        FLAME lattice file.
    _machine:
        FLAME machine object.

    Returns
    -------
    list
        None if failed, or list of valid element types' string names.
    """
    m = machine_setter(latfile, _machine, 'get_all_types')
    if m is None:
        return None

    mconf = m.conf()
    mconfe = mconf['elements']
    return list(set([i.get('type') for i in mconfe]))


def get_all_names(latfile=None, _machine=None):
    """Get all uniqe names from a FLAME machine or lattice file.

    Parameters
    ----------
    latfile : str
        FLAME lattice file.
    _machine :
        FLAME machine object.

    Returns
    -------
    str or None
        None if failed, or list of valid element types' string names.
    """
    m = machine_setter(latfile, _machine, 'get_all_names')
    if m is None:
        return None

    mconf = m.conf()
    mconfe = mconf['elements']
    return list(set([i.get('name') for i in mconfe]))


def inspect_lattice(latfile=None, out=None, _machine=None):
    """Inspect FLAME lattice file, print a lattice information report,
    if failed, print nothing.

    Parameters
    ----------
    latfile :
        FLAME lattice file.
    out :
        output stream, stdout by default.
    _machine :
        FLAME machine object.

    Returns
    -------
    None
        None if failed, or print information.

    Examples
    --------
    >>> from flame import Machine
    >>> from phantasy import flameutils
    >>> latfile = 'lattice/test.lat'
    >>> m = Machine(open(latfile, 'r'))
    >>> flameutils.inspect_lattice(_machine=m)
    Inspecting lattice: <machine>
    ==============================
    TYPE        COUNT   PERCENTAGE
    ------------------------------
    SOURCE       1       0.08
    STRIPPER     1       0.08
    QUADRUPOLE   40      3.22
    BPM          75      6.04
    SOLENOID     78      6.28
    SBEND        80      6.44
    RFCAVITY     117     9.42
    ORBTRIM      120     9.66
    DRIFT        730    58.78
    >>> # pass the latfile parameter
    >>> flameutils.inspect_lattice(latfile=latfile)
    Inspecting lattice: test.lat
    ==============================
    TYPE        COUNT   PERCENTAGE
    ------------------------------
    SOURCE       1       0.08
    STRIPPER     1       0.08
    QUADRUPOLE   40      3.22
    BPM          75      6.04
    SOLENOID     78      6.28
    SBEND        80      6.44
    RFCAVITY     117     9.42
    ORBTRIM      120     9.66
    DRIFT        730    58.78
    >>>
    >>> ## write inspection message to other streams
    >>> # write to file
    >>> fout = open('test.out', 'w')
    >>> flameutils.inspect_lattice(latfile=latfile, out=fout)
    >>> fout.close()
    >>>
    >>> # write to string
    >>> from StringIO import StringIO
    >>> sio = StringIO()
    >>> flameutils.inspect_lattice(latfile=latfile, out=sio)
    >>> retstr = sio.getvalue()
    """
    if latfile is None:
        latfile = "<machine>"  # data from machine, not lattice file
    m = machine_setter(latfile, _machine, 'inspect_lattice')
    if m is None:
        return None

    mconf = m.conf()
    mconfe = mconf['elements']
    msize = len(mconfe)
    type_cnt = Counter([i.get('type') for i in mconfe])
    etable = [(t, n, n / msize) for (t, n) in sorted(type_cnt.items(), key=lambda x: x[1])]

    out = sys.stdout if out is None else out
    print("Inspecting lattice: %s" % os.path.basename(latfile), file=out)
    print("=" * 30, file=out)
    print("{0:<11s} {1:<7s} {2:<10s}".format("TYPE", "COUNT", "PERCENTAGE"), file=out)
    print("-" * 30, file=out)
    for (t, n, p) in etable:
        outstr = "{t:<12s} {n:<5d} {p:^8.2f}".format(t=t.upper(), n=n, p=p * 100)
        print(outstr, file=out)


def get_element(latfile=None, index=None, name=None, type=None, **kws):
    """Inspect FLAME lattice element, return properties.

    Parameters
    ----------
    latfile : str
        FLAME lattice file.
    index : int
        (list of) Element index(s).
    name : str
        (list of) Element name(s).
    type : str
        (list of) Element type(s).

    Keyword Arguments
    -----------------
    _machine :
        FLAME machine object.
    _pattern : str
        Regex to search element name.

    Returns
    -------
    res : list of dict or []
        List of dict of properties or empty list

    Note
    ----
    If more than one optional paramters (index, name, type, _pattern) are
    provided, only return element that meets all these definitions.

    Examples
    --------
    >>> from flame import Machine
    >>> from phantasy import flameutils
    >>> latfile = 'lattice/test.lat'
    >>> ename = 'LS1_CA01:CAV4_D1150'
    >>> e = flameutils.get_element(name=ename, latfile=latfile)
    >>> print(e)
    [{'index': 27, 'properties': {'aper': 0.017, 'name': 'LS1_CA01:CAV4_D1150',
      'f': 80500000.0, 'cavtype': '0.041QWR', 'L': 0.24, 'phi': 325.2,
      'scl_fac': 0.819578, 'type': 'rfcavity'}}]
    >>> # use multiple filters, e.g. get all BPMs in the first 20 elements
    >>> e = flameutils.get_element(_machine=m, index=range(20), type='bpm')
    >>> print(e)
    [{'index': 18, 'properties': {'name': 'LS1_CA01:BPM_D1144', 'type': 'bpm'}},
     {'index': 5, 'properties': {'name': 'LS1_CA01:BPM_D1129', 'type': 'bpm'}}]
    >>> # all these filters could be used together, return [] if found nothing
    >>>
    >>> # get names by regex
    >>> e = flameutils.get_element(_machine=m, _pattern='FS1_BBS:DH_D2394_1.?')
    >>> print(e)
    [{'index': 1092, 'properties': {'L': 0.104065, 'aper': 0.07,
      'bg': 0.191062, 'name': 'FS1_BBS:DH_D2394_1', 'phi': 4.5, 'phi1': 7.0,
      'phi2': 0.0, 'type': 'sbend'}},
     {'index': 1101, 'properties': {'L': 0.104065, 'aper': 0.07,
      'bg': 0.191062, 'name': 'FS1_BBS:DH_D2394_10', 'phi': 4.5, 'phi1': 0.0,
      'phi2': 7.0, 'type': 'sbend'}}]

    Warning
    -------
    Invalid element names or type names will be ignored.

    See Also
    --------
    get_index_by_name, get_index_by_type
    :func:`.get_intersection` : Get the intersection of input valid list/tuple.
    """
    _machine = kws.get('_machine', None)
    m = machine_setter(latfile, _machine, 'get_element')
    if m is None:
        return None

    if index is not None:
        if not isinstance(index, (list, tuple)):
            idx_from_index = index,
        else:
            idx_from_index = index
    else:
        idx_from_index = []

    names = []
    # name pattern
    _name_pattern, _name_list = kws.get('_pattern'), None
    if _name_pattern is not None:
        _name_list = get_names_by_pattern(pattern=_name_pattern, _machine=m)
    if _name_list is not None:
        names = _name_list

    if name is not None:
        if isinstance(name, str):
            names.append(name)
        elif isinstance(name, list):
            names.extend(name)

    if names:
        idx_from_name = list(flatten(get_index_by_name(names, _machine=m, rtype='list')))
    else:
        idx_from_name = []

    if type is not None:
        idx_from_type = list(flatten(get_index_by_type(type, _machine=m, rtype='list')))
    else:
        idx_from_type = []

    ele_idx = get_intersection(index=idx_from_index, name=idx_from_name, type=idx_from_type)

    if ele_idx == []:
        _LOGGER.warning("get_element: Nothing to get, invalid filtering.")
        return []
    else:
        mconf = m.conf()
        mks = mconf.keys()
        share_keys = [k for k in mks if k not in ("elements", "name")]
        retval = []
        for i in ele_idx:
            elem = m.conf(i)
            elem_k = set(elem.keys()).difference(share_keys)
            if elem.get('type') == 'stripper':
                [elem_k.add(k) for k in STRIPPER_PROP_KEYS]
            elif elem.get('type') == 'source':
                ndim_charge = len(elem.get('NCharge'))
                p = elem.get('vector_variable', None)
                s = elem.get('matrix_variable', None)
                for v in (s, p):
                    if v is not None:
                        [elem_k.add('{0}{1}'.format(v, i)) for i in range(ndim_charge)]
                [elem_k.add(k) for k in SOURCE_PROP_KEYS]
            elem_p = {k: elem.get(k) for k in elem_k}
            retval.append({'index': i, 'properties': elem_p})
        return retval


def get_index_by_type(type='', latfile=None, rtype='dict', _machine=None):
    """Get element(s) index by type(s).

    Parameters
    ----------
    type : str or list of str
        Single element type name or list[tuple] of element type names.
    latfile :
        FLAME lattice file, preferred.
    rtype : str
        Return type, 'dict' (default) or 'list'.
    _machine :
        FLAME machine object.

    Returns
    -------
    ind : dict or list
        Dict, key is type name, value if indice list of each type name,
        list, of indices list, with the order of type.

    Note
    ----
    If *rtype* is ``list``, list of list would be returned instead of a dict,
    ``flatten()`` function could be used to flatten the list.

    Examples
    --------
    >>> from flame import Machine
    >>> from phantasy import flameutils
    >>> latfile = 'lattice/test.lat'
    >>> m = Machine(open(latfile, 'r'))
    >>> types = 'stripper'
    >>> print(flameutils.get_index_by_type(type=types, latfile=latfile))
    {'stripper': [891]}
    >>> print(flameutils.get_index_by_type(type=types, _machine=m))
    {'stripper': [891]}
    >>> types = ['stripper', 'source']
    >>> print(flameutils.get_index_by_type(type=types, latfile=latfile))
    {'source': [0], 'stripper': [891]}
    >>> # return a list instead of dict
    >>> print(flameutils.get_index_by_type(type=types, latfile=latfile, rtype='list'))
    [[891], [0]]

    See Also
    --------
    :func:`.flatten` : flatten recursive list.
    """
    m = machine_setter(latfile, _machine, 'get_index_by_type')
    if m is None:
        return None

    if not isinstance(type, (list, tuple)):
        type = type,

    if rtype == 'dict':
        return {t: m.find(type=t) for t in type}
    else:  # list
        return [m.find(type=t) for t in type]


def get_index_by_name(name='', latfile=None, rtype='dict', _machine=None):
    """Get index(s) by name(s).

    Parameters
    ----------
    name : str or list of str
        Single element name or list[tuple] of element names
    latfile :
        FLAME lattice file, preferred.
    rtype : str
        Return type, 'dict' (default) or 'list'.
    _machine :
        FLAME machine object.

    Returns
    -------
    ind : dict or list
        dict of element indices, key is name, value is index,
        list of element indices list

    Note
    ----
    If *rtype* is ``list``, list of list would be returned instead of a dict,
    ``flatten()`` function could be used to flatten the list.

    Examples
    --------
    >>> from flame import Machine
    >>> from phantasy import flameutils
    >>> latfile = 'lattice/test.lat'
    >>> m = Machine(open(latfile, 'r'))
    >>> names = 'LS1_CA01:SOL1_D1131_1'
    >>> print(flameutils.get_index_by_name(name=names, latfile=latfile))
    {'LS1_CA01:SOL1_D1131_1': [8]}
    >>> print(flameutils.get_index_by_name(name=names, _machine=m))
    {'LS1_CA01:SOL1_D1131_1': [8]}
    >>> names = ['LS1_CA01:SOL1_D1131_1', 'LS1_CA01:CAV4_D1150',
    >>>          'LS1_WB01:BPM_D1286', 'LS1_CA01:BPM_D1144']
    >>> print(flameutils.get_index_by_name(name=names, latfile=latfile))
    {'LS1_CA01:SOL1_D1131_1': [8], 'LS1_WB01:BPM_D1286': [154],
     'LS1_CA01:BPM_D1144': [18], 'LS1_CA01:CAV4_D1150': [27]}
    >>> # return a list instead of dict
    >>> print(flameutils.get_index_by_name(name=names, latfile=latfile, rtype='list'))
    [[8], [27], [154], [18]]

    See Also
    --------
    :func:`.flatten` : flatten recursive list.
    """
    m = machine_setter(latfile, _machine, 'get_index_by_name')
    if m is None:
        return None

    if not isinstance(name, (list, tuple)):
        name = name,
    if rtype == 'dict':
        return {n: m.find(name=n) for n in name}
    else:
        return [m.find(name=n) for n in name]


def get_names_by_pattern(pattern='.*', latfile=None, _machine=None):
    """Get element names by regex defined by *pattern*.

    Parameters
    ----------
    pattern : str
        Regex to search element name.
    latfile :
        FLAME lattice file, preferred.
    _machine :
        FLAME machine object.

    Returns
    -------
    names : List
        List of element names, if not found, return None.
    """
    m = machine_setter(latfile, _machine, 'get_names_by_pattern')
    if m is None:
        return None

    econf = m.conf().get('elements')
    rp = re.compile(pattern)
    m_names = [e.get('name') for e in econf if rp.search(e.get('name'))]
    if m_names != []:
        return m_names
    else:
        return None


def insert_element(machine=None, index=None, element=None):
    """Insert new element to the machine.

    Parameters
    ----------
    machine :
        FLAME machine object.
    index :
        Insert element before the index (or element name).
    element :
        Lattice element dictionary. e.g. {'name':xxx, 'type':yyy, 'L':zzz}

    Returns
    -------
    machine : FLAME machine object.
    """
    if machine is None:
        return None

    try:
        m = conf_update(machine)
        mconf = m.conf()
    except:
        _LOGGER.error("Failed to load FLAME machine object.")
        return None

    if index is not None and element is not None:
        if isinstance(index, (str, unicode)):
            index = m.find(name=index)[0]
        mconf['elements'].insert(index, element)

    new_m = flame.Machine(mconf)
    return new_m
