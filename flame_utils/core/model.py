#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simulation/modeling with FLAME.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from flame import Machine

from .state import MachineStates
from .element import get_element
from .element import get_all_types
from .element import get_all_names
from .element import get_index_by_type
from .element import get_index_by_name

from flame_utils.misc import is_zeros_states
from flame_utils.misc import machine_setter
from flame_utils.io import collect_data

_LOGGER = logging.getLogger(__name__)


def propagate(machine=None, mstates=None, from_element=None, to_element=None, monitor=None, **kws):
    """Propagate ``MachineStates``.

    Parameters
    ----------
    machine :
        FLAME machine object.
    mstates :
        MachineStates object.
    from_element : int
        Element index of start point, if not set, will be the first element.
    to_element : int
        Element index of end point, if not set, will be the last element.
    monitor : list
        List of element indice selected as states monitors, if set -1, will be
        a list of only last element.

    Keyword Arguments
    -----------------
    latfile : str
        FLAME lattice file.

    Returns
    -------
    tuple
        None if failed, else tuple of ``(r,ms)``, where ``r`` is list of
        results at each monitor points, ``ms`` is ``MachineStates`` object
        after the last monitor point.

    See Also
    --------
    MachineStates : FLAME Machine state class created for ``MomentMatrix``
        type.
    """
    _latfile = kws.get('latfile', None)
    _machine = machine
    _m = machine_setter(_latfile, _machine, 'propagate')
    if _m is None:
        return None
    if mstates is None:
        s0 = _m.allocState({})
        ms = MachineStates(s0)
    else:
        ms = mstates

    vstart = 0 if from_element is None else from_element
    vend = len(_m) - 1 if to_element is None else to_element
    obs = [vend] if monitor is -1 else monitor

    vmax = vend - vstart + 1
    s = ms.mstates
    r = _m.propagate(s, start=vstart, max=vmax, observe=obs)
    ms.mstates = s
    return r, ms


def configure(machine=None, econf=None, **kws):
    """Configure FLAME machine.

    Parameters
    ----------
    machine :
        FLAME machine object.
    econf : (list of) dict
        Element configuration(s).

    Keyword Arguments
    -----------------
    latfile : str
        FLAME lattice file.
    c_idx : int
        Element index.
    c_dict : dict
        Configuration dict.


    Returns
    -------
    m : new FLAME machine object
        None if failed, else new machine object.

    Note
    ----
    If wanna configure FLAME machine by conventional way, then *c_idx* and
    *c_dict* could be used, e.g. reconfigure one corrector of ``m``:
    ``configure(machine=m, c_idx=10, c_dict={'theta_x': 0.001})``
    which is just the same as: ``m.reconfigure(10, {'theta_x': 0.001})``.

    Examples
    --------
    >>> from flame import Machine
    >>> from phantasy import flameutils
    >>>
    >>> latfile = 'test.lat'
    >>> m = Machine(open(latfile, 'r'))
    >>>
    >>> # reconfigure one element
    >>> e1 = flameutils.get_element(_machine=m, index=1)[0]
    >>> print(e1)
    {'index': 1, 'properties': {'L': 0.072, 'aper': 0.02,
     'name': 'LS1_CA01:GV_D1124', 'type': 'drift'}}
    >>> e1['properties']['aper'] = 0.04
    >>> m = flameutils.configure(m, e1)
    >>> print(flameutils.get_element(_machine=m, index=1)[0])
    {'index': 1, 'properties': {'L': 0.072, 'aper': 0.04,
     'name': 'LS1_CA01:GV_D1124', 'type': 'drift'}}
    >>>
    >>> # reconfiugre more than one element
    >>> e_cor = flameutils.get_element(_machine=m, type='orbtrim')
    >>> # set all horizontal correctors with theta_x = 0.001
    >>> for e in e_cor:
    >>>     if 'theta_x' in e['properties']:
    >>>         e['properties']['theta_x'] = 0.001
    >>> m = flameutils.configure(m, e_cor)

    See Also
    --------
    get_element : Get FLAME lattice element configuration.
    """
    _latfile = kws.get('latfile', None)
    _machine = machine
    _m = machine_setter(_latfile, _machine, 'configure')
    if _m is None:
        return None

    _c_idx, _c_dict = kws.get('c_idx'), kws.get('c_dict')
    if _c_idx is not None and _c_dict is not None:
        _m.reconfigure(_c_idx, _c_dict)
    else:
        if not isinstance(econf, list):
            _m.reconfigure(econf['index'], econf['properties'])
        else:
            for e in econf:
                _m.reconfigure(e['index'], e['properties'])
    return _m


class ModelFlame(object):
    """General FLAME modeling class.

    Parameters
    ----------
    lat_file : str
        FLAME lattice file, if not set, None.

    Examples
    --------
    >>> from flame import Machine
    >>> from phantasy import flameutils
    >>>
    >>> latfile = "lattice/test.lat"
    >>> fm1 = flameutils.ModelFlame()
    >>> # manually initialization
    >>> fm1.latfile = latfile
    >>> m = Machine(latfile)
    >>> fm1.machine = m
    >>> fm1.mstates = m.allocState({})
    >>> # or by explicitly calling:
    >>> fm1.machine, fm1.mstates = fm1.init_machine(latfile)
    >>>
    >>> # initialize with valid lattice file
    >>> fm2 = flameutils.ModelFlame(lat_file=latfile)
    >>>
    >>> # (Recommanded) initialize with MachineStates
    >>> fm = flameutils.ModelFlame()
    >>> ms = flameutils.MachineStates(machine=m)
    >>> # now the attributes of ms could be arbitarily altered
    >>> fm.mstates = ms
    >>> fm.machine = m
    >>>
    >>> # run fm
    >>> obs = fm.get_index_by_type(type='bpm')['bpm']
    >>> r, s = fm.run(monitor=obs)
    >>>
    >>> # get result, storing as a dict, e.g. data
    >>> data = fm.collect_data(r, pos=True, x0=True, y0=True)

    See Also
    --------
    MachineStates : FLAME Machine state class created for ``MomentMatrix`` type.
    """

    def __init__(self, lat_file=None, **kws):
        self._lat_file = lat_file
        self._mach_ins, self._mach_states = self.init_machine(lat_file)

    @property
    def latfile(self):
        """str: FLAME lattice file name."""
        return self._lat_file

    @latfile.setter
    def latfile(self, fn):
        self._lat_file = fn

    @property
    def machine(self):
        """FLAME machine object."""
        return self._mach_ins

    @machine.setter
    def machine(self, m):
        self._mach_ins = m
        if self._mach_states is None:
            self._mach_states = m.allocState({})

    @property
    def mstates(self):
        """FLAME machine state, could be internal state or MachineStates.

        See Also
        --------
        MachineStates : FLAME Machine state class created for ``MomentMatrix``
            type.
        """
        return self._mach_states

    @mstates.setter
    def mstates(self, s):
        self._mach_states = s

    @staticmethod
    def init_machine(latfile):
        """Initialize FLAME machine.

        Parameters
        ----------
        latfile :
            FLAME lattice file.

        Returns
        -------
        res : tuple
            Tuple of ``(m, s)``, where ``m`` is FLAME machine instance,
            and ``s`` is initial machine states.
        """
        try:
            with open(latfile, 'r') as f:
                m = Machine(f)
            s = m.allocState({})
            m.propagate(s, 0, 1)
            _LOGGER.info("ModelFlame: Initialization succeeded.")
            return m, s
        except:
            _LOGGER.warning("ModelFlame: Lattice file is not valid, do it manually.")
            return None, None

    def get_element(self, name=None, index=None, type=None, **kws):
        """Element inspection, get properties.

        Returns
        -------
        res : list of dict
            List of dict of properties or empty list.

        See Also
        --------
        get_element : Get element from FLAME machine object.
        """
        elem_list = get_element(_machine=self._mach_ins,
                                name=name, index=index, type=type,
                                **kws)
        return elem_list

    def inspect_lattice(self):
        """Inspect FLAME machine and print out information.

        See Also
        --------
        inspect_lattice : Inspect FLAME lattice file, print a brief report.
        """
        inspect_lattice(_machine=self._mach_ins)

    def get_all_types(self):
        """Get all uniqe element types.

        Returns
        -------
        res : list of str
            List of element type names

        See Also
        --------
        get_all_types : Get all unique types from a FLAME machine.
        """
        return get_all_types(_machine=self._mach_ins)

    def get_all_names(self):
        """Get all uniqe element names.

        Returns
        -------
        res : list of str
            List of element names.

        See Also
        --------
        get_all_names : Get all uniqe names from a FLAME machine.
        """
        return get_all_names(_machine=self._mach_ins)

    def get_index_by_type(self, type='', rtype='dict'):
        """Get element(s) index by type(s).

        Parameters
        ----------
        type : str or list of str
            Single element type name or list[tuple] of element type names.
        rtype : str
            Return type, 'dict' (default) or 'list'.

        Returns
        -------
        ind : dict or list
            Dict, key is type name, value if indice list of each type name,
            list, of indices list, with the order of type.

        See Also
        --------
        get_index_by_type : Get element(s) index by type(s).
        """
        return get_index_by_type(type=type, rtype=rtype, _machine=self._mach_ins)

    def get_index_by_name(self, name='', rtype='dict'):
        """Get index(s) by name(s).

        Parameters
        ----------
        name : list or tuple of str
            Single element name or list[tuple] of element names.
        rtype : str
            Return type, 'dict' (default) or 'list'.

        Returns
        -------
        dict or list
            Dict of element indices, key is name, value is index,
            list of element indices list.

        See Also
        --------
        get_index_by_name : Get index(s) by element name(s).
        """
        return get_index_by_name(name=name, _machine=self._mach_ins, rtype=rtype)

    def run(self, mstates=None, from_element=None, to_element=None, monitor=None):
        """Simulate model.

        Parameters
        ----------
        mstates :
            FLAME machine states object, also could be :class:`MachineStates`
            object if not set, will use the one from ``ModelFlame`` object
            itself, usually is created at the initialization stage,
            see :func:`init_machine()`.
        from_element : int
            Element index of start point, if not set, will be the first element
            if not set, will be 0 for zero states, or 1.
        to_element : int
            Element index of end point, if not set, will be the last element.
        monitor : list[int]
            List of element indice selected as states monitors, if set -1,
            will be a list of only last element.

        Returns
        -------
        tuple
            Tuple of ``(r,s)``, where ``r`` is list of results at each monitor
            points, ``s`` is ``MachineStates`` object after the last monitor
            point.

        Warning
        -------
        This method does not change the input *mstates*, while ``propagate``
        changes.

        See Also
        --------
        MachineStates : FLAME Machine state class created for ``MomentMatrix``
            type.
        propagate : Propagate ``MachineStates`` within defined FLAME machine
            object.
        """
        m = self._mach_ins
        if mstates is None:
            # s = m.allocState({})
            s = self._mach_states.clone()
        else:
            s = mstates.clone()

        if is_zeros_states(s):
            vstart = 0 if from_element is None else from_element
        else:
            vstart = 1 if from_element is None else from_element
        vend = len(m) - 1 if to_element is None else to_element
        obs = [vend] if monitor is -1 else monitor

        vmax = vend - vstart + 1
        if isinstance(s, MachineStates):
            r, s = propagate(m, s, from_element=vstart, to_element=vend, monitor=obs)
        else:
            r = m.propagate(s, start=vstart, max=vmax, observe=obs)
        r = self.convert_results(r)
        return r, s

    @staticmethod
    def convert_results(mres, **kws):
        """Convert all machine states of results generated by :func:`run()`
        method to be ``MachineStates`` object.

        Parameters
        ----------
        mres : list of tuple
            List of propagation results.

        Returns
        -------
        list of tuple
            Tuple of ``(r,s)``, where ``r`` is list of results at each monitor
            points, ``s`` is ``MachineStates`` object after the last monitor
            point.
        """
        return [(i, MachineStates(s)) for (i, s) in mres]

    @staticmethod
    def collect_data(result, **kws):
        """collect data of interest from propagation results

        Parameters
        ----------
        result :
            propagation results with ``MachineStates`` object

        See Also
        --------
        collect_data : Get data of interest from results.
        """
        return collect_data(result, **kws)

    def configure(self, econf):
        """Configure FLAME model.

        Parameters
        ----------
        econf : (list of) dict
            Element configuration(s), see :func:`get_element`.

        See Also
        --------
        configure : Configure FLAME machine.
        get_element : Get FLAME lattice element configuration.
        """
        m = configure(self._mach_ins, econf)
        self._mach_ins = m

