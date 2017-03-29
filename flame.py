#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module contains classes/functions to serve as utils for the
potential requirment of modeling accelerator with FLAME code.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import re
import sys
from collections import Counter

import numpy as np
from flame import Machine

from phantasy.library.misc import miscutils

__authors__ = "Tong Zhang"
__copyright__ = "(c) 2016, Facility for Rare Isotope beams, Michigan State University"
__contact__ = "Tong Zhang <zhangt@frib.msu.edu>"

_LOGGER = logging.getLogger(__name__)


def generate_latfile(machine, latfile=None, out=None):
    """Generate lattice file for the usage of FLAME code.

    Parameters
    ----------
    machine :
        FLAME machine object.
    latfile :
        File name for generated lattice file.
    out :
        New stream paramter, file stream, would be preferred.

    Returns
    -------
    filename : str
        None if failed to generate lattice file, or the out file name.

    Note
    ----
    - If *latfile* and *out* are not defined, will print all output to screen;
    - If *latfile* and *out* are all defined, *out* stream is preferred;
    - For other cases, choose one that is defined.

    Examples
    --------
    >>> from flame import Machine
    >>> latfile = 'test.lat'
    >>> m = Machine(open(latfile))
    >>> outfile1 = generate_latfile(m, 'out1.lat')
    >>> m.reconfigure(80, {'theta_x': 0.1})
    >>> outfile2 = generate_latfile(m, 'out2.lat')
    >>> # recommand new way
    >>> fout = open('out.lat', 'w')
    >>> generate_latfile(m, out=fout)

    Note
    ----
    Parameter *out* can also be ``StringIO``, and get string by ``getvalue()``.

    Warning
    -------
    To get element configuration only by ``m.conf(i)`` method,
    where ``m`` is ``flame.Machine`` object, ``i`` is element index,
    when some re-configuring operation is done, ``m.conf(i)`` will be update,
    but ``m.conf()["elements"]`` remains with the initial values.
    """
    m = machine
    try:
        mconf = m.conf()
        mks = mconf.keys()
    except:
        print("Failed to load FLAME machine object.")
        return None

    try:
        mconf_ks = mconf.keys()
        [mconf_ks.remove(i) for i in ['elements', 'name'] if i in mconf_ks]

        #
        lines = []
        for k in mconf_ks:
            v = mconf[k]
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if isinstance(v, str):
                v = '"{0}"'.format(v)
            line = '{k} = {v};'.format(k=k, v=v)
            lines.append(line)

        mconfe = mconf['elements']

        # element configuration
        elem_num = len(mconfe)
        elem_name_list = []
        for i in range(0, elem_num):
            elem_i = m.conf(i)
            ename, etype = elem_i['name'], elem_i['type']
            if ename in elem_name_list:
                continue
            elem_name_list.append(ename)
            ki = elem_i.keys()
            elem_k = set(ki).difference(mks)
            if etype == 'stripper':
                elem_k.add('IonChargeStates')
                elem_k.add('NCharge')
            p = []
            for k, v in elem_i.items():
                if k in elem_k and k not in ['name', 'type']:
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    if isinstance(v, str):
                        v = '"{0}"'.format(v)
                    p.append('{k} = {v}'.format(k=k, v=v))
            pline = ', '.join(p)

            line = '{n}: {t}, {p}'.format(n=ename, t=etype, p=pline)

            line = line.strip(', ') + ';'
            lines.append(line)

        dline = '(' + ', '.join(([e['name'] for e in mconfe])) + ')'

        blname = mconf['name']
        lines.append('{0}: LINE = {1};'.format(blname, dline))
        lines.append('USE: {0};'.format(blname))
    except:
        print("Failed to generate lattice file.")
        return None

    all_lines = '\n'.join(lines)
    try:
        if latfile is None and out is None:
            sout = sys.stdout
        elif out is None:
            sout = open(latfile, 'w')
        else:
            sout = out
        print(all_lines, file=sout)
    except:
        print("Failed to write to %s" % latfile)
        return None

    try:
        retval = sout.name
    except:
        retval = 'string'

    return retval


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
    _m = miscutils.machine_setter(_latfile, _machine, 'configure')
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


def convert_results(mres, **kws):
    """Convert all machine states of results generated by :func:`run()` method
    to be ``MachineStates`` object.

    Parameters
    ----------
    mres : list of tuple
        List of propagation results.

    Returns
    -------
    list of tuple
        Tuple of ``(r,s)``, where ``r`` is list of results at each monitor
        points, ``s`` is ``MachineStates`` object after the last monitor point.
    """
    return [(i, MachineStates(s)) for (i, s) in mres]


def collect_data(result, **kws):
    """Collect data of interest from propagation results.

    Parameters
    ----------
    result : 
        Propagation results with ``MachineStates`` object.

    Keyword Arguments
    -----------------
    pos : float
        Longitudinally propagating position, [m].
    ref_beta : float
        Speed in the unit of light velocity in vacuum of reference charge
        state, Lorentz beta.
    ref_bg : float
        Multiplication of beta and gamma of reference charge state.
    ref_gamma : float
        Relativistic energy of reference charge state, Lorentz gamma.
    ref_IonEk : float
        Kinetic energy of reference charge state, [eV/u].
    ref_IonEs : float
        Rest energy of reference charge state, [eV/u].
    ref_IonQ : int
        Macro particle number of reference charge state.
    ref_IonW : float
        Total energy of reference charge state, [eV/u],
        i.e. :math:`W = E_s + E_k`.
    ref_IonZ : float
        Reference charge state, measured by charge to mass ratio, e.g.
        :math:`^{33^{+}}_{238}U: Q[33]/A[238]`.
    ref_phis : float
        Absolute synchrotron phase of reference charge state, [rad].
    ref_SampleIonK : float
        Wave-vector in cavities with different beta values of reference charge
        state.
    beta : Array
        Speed in the unit of light velocity in vacuum of all charge states,
        Lorentz beta.
    bg : Array
        Multiplication of beta and gamma of all charge states.
    gamma : Array
        Relativistic energy of all charge states, Lorentz gamma.
    IonEk : Array
        Kinetic energy of all charge states, [eV/u].
    IonEs : Array
        Rest energy of all charge states, [eV/u].
    IonQ : Array
        Macro particle number of all charge states.
    IonW : Array
        Total energy of all charge states, [eV/u], i.e. :math:`W = E_s + E_k`.
    IonZ : Array
        All charge states, measured by charge to mass ratio
    phis : Array
        Absolute synchrotron phase of all charge states, [rad]
    SampleIonK : Array
        Wave-vector in cavities with different beta values of all charge
        states.
    x0 : Array
        X centroid for all charge states, [mm].
    y0 : Array
        Y centroid for all charge states, [mm].
    xp0 : Array
        X centroid divergence for all charge states, [rad].
    yp0 : Array
        Y centroid divergence for all charge states, [rad].
    phi0 : Array
        Longitudinal beam length, measured in RF frequency for all charge
        states, [rad].
    dEk0 : Array
        Kinetic energy deviation w.r.t. reference charge state, for all charge
        states, [MeV/u].
    x0_rms : Array
        General rms beam envelope for x, [mm].
    y0_rms : Array
        General rms beam envelope for y, [mm].
    xp0_rms : Array
        General rms beam envelope for x', [rad].
    yp0_rms : Array
        General rms beam envelope for y', [rad].
    phi0_rms : Array
        General rms beam envelope for :math:`\phi`, [rad].
    dEk0_rms : Array
        General rms beam envelope for :math:`\delta E_k`, [MeV/u].
    x0_env : Array
        Weight average of all charge states for x', [rad].
    y0_env : Array
        Weight average of all charge states for y, [mm].
    xp0_env : Array
        Weight average of all charge states for x', [rad].
    yp0_env : Array
        Weight average of all charge states for y', [rad].
    phi0_env : Array
        Weight average of all charge states for :math:`\phi`, [mm].
    dEk0_env : Array
        Weight average of all charge states for :math:`\delta E_k`, [MeV/u].
    moment0_env : Array
        Weight average of centroid for all charge states, array of
        ``[x, x', y, y', phi, dEk, 1]``, with the units of
        ``[mm, rad, mm, rad, rad, MeV/u, 1]``.
    moment0 : Array
        Centroid for all charge states, array of ``[x, x', y, y', phi, dEk, 1]``.
    moment0_rms : Array
        RMS beam envelope, part of statistical results from ``moment1``.
    moment1 : Array
        Correlation tensor of all charge states, for each charge state.

    Returns
    -------
    dict
        Dict of ``{k1:v1, k2,v2...}``, keys are from keyword parameters.

    Note
    ----
    Set the data of interest with ``k=True`` as input will return the defined
    ``k`` value.

    Examples
    --------
    >>> # get x0 and y0 array
    >>> collect_data(r, x0=True, y0=True)

    See Also
    --------
    MachineStates : FLAME Machine state class created for ``MomentMatrix`` type.
    """
    valid_keys = [k for k, v in kws.items() if v is not None]
    try:
        return {ik: np.array([getattr(s, ik) for (i, s) in result]) for ik in valid_keys}
    except:
        result = convert_results(result)
        return {ik: np.array([getattr(s, ik) for (i, s) in result]) for ik in valid_keys}


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
    _m = miscutils.machine_setter(_latfile, _machine, 'propagate')
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
    m = miscutils.machine_setter(latfile, _machine, 'get_all_types')
    if m is None: return None

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
    m = miscutils.machine_setter(latfile, _machine, 'get_all_names')
    if m is None: return None

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
    >>> from cStringIO import StringIO
    >>> sio = StringIO()
    >>> flameutils.inspect_lattice(latfile=latfile, out=sio)
    >>> retstr = sio.getvalue()
    """
    if latfile is None:
        latfile = "<machine>"  # data from machine, not lattice file
    m = miscutils.machine_setter(latfile, _machine, 'inspect_lattice')
    if m is None: return None

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
    m = miscutils.machine_setter(latfile, _machine, 'get_element')
    if m is None: return None

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
        idx_from_name = list(miscutils.flatten(get_index_by_name(names, _machine=m, rtype='list')))
    else:
        idx_from_name = []

    if type is not None:
        idx_from_type = list(miscutils.flatten(get_index_by_type(type, _machine=m, rtype='list')))
    else:
        idx_from_type = []

    ele_idx = miscutils.get_intersection(index=idx_from_index, name=idx_from_name, type=idx_from_type)

    if ele_idx == []:
        _LOGGER.warning("get_element: Nothing to inspect")
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
                elem_k.add('IonChargeStates')
                elem_k.add('NCharge')
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
    m = miscutils.machine_setter(latfile, _machine, 'get_index_by_type')
    if m is None: return None

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
    m = miscutils.machine_setter(latfile, _machine, 'get_index_by_name')
    if m is None: return None

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
    m = miscutils.machine_setter(latfile, _machine, 'get_names_by_pattern')
    if m is None: return None

    econf = m.conf().get('elements')
    rp = re.compile(pattern)
    m_names = [e.get('name') for e in econf if rp.search(e.get('name'))]
    if m_names != []:
        return m_names
    else:
        return None
    

def is_zeros_states(s):
    """ test if flame machine states is all zeros

    Returns
    -------
    True or False
        True if is all zeros, else False
    """
    return np.alltrue(getattr(s, 'moment0') == np.zeros([7, 1]))


class MachineStates(object):
    """Class for general FLAME machine states

    all attributes of states:

     - ``pos``,
     - ``ref_beta``, ``ref_bg``, ``ref_gamma``, ``ref_IonEk``, ``ref_IonEs``,
       ``ref_IonQ``, ``ref_IonW``, ``ref_IonZ``, ``ref_phis``,
       ``ref_SampleIonK``,
     - ``beta``, ``bg``, ``gamma``, ``IonEk``, ``IonEs``, ``IonQ``, ``IonW``,
       ``IonZ``, ``phis``, ``SampleIonK``,
     - ``moment0``, ``moment0_rms``, ``moment0_env``, ``moment1``
     - ``phis``, ``phi0``, ``phi0_env``, ``phi0_rms``,
     - ``x0``, ``x0_env``, ``x0_rms``, ``xp0``, ``xp0_env``, ``xp0_rms``,
     - ``y0``, ``y0_env``, ``y0_rms``, ``yp0``, ``yp0_env``, ``yp0_rms``,
     - ``last_caviphi0`` (since version 1.1.1)

    Warning
    -------
    1. These attributes are only valid for the case of ``sim_type`` being
       defined as ``MomentMatrix``, which is de facto the exclusive option
       used at FRIB.
    2. If the attribute is an array, new array value should be assigned
       instead of by element indexing way, e.g.

       >>> ms = MachineStates(s)
       >>> print(ms.moment0)
       array([[ -7.88600000e-04],
              [  1.08371000e-05],
              [  1.33734000e-02],
              [  6.67853000e-06],
              [ -1.84773000e-04],
              [  3.09995000e-04],
              [  1.00000000e+00]])
       >>> # the right way to just change the first element of the array
       >>> m_tmp = ms.moment0
       >>> m_tmp[0] = 0
       >>> ms.moment0 = m_tmp
       >>> print(ms.moment0)
       array([[  0.00000000e+00],
              [  1.08371000e-05],
              [  1.33734000e-02],
              [  6.67853000e-06],
              [ -1.84773000e-04],
              [  3.09995000e-04],
              [  1.00000000e+00]])
       >>> # this way does work: ms.moment0[0] = 0


    Parameters
    ----------
    s : 
        machine states object.

    Keyword Arguments
    -----------------
    mstates : 
        flame machine states object, priority: high
    machine : 
        flame machine object, priority: middle
    latfile : 
        flame lattice file name, priority: low

    Note
    ----
    If more than one keyword parameters are provided, 
    the selection policy follows the priority from high to low.
    
    Warning
    -------
    If only ``s`` is assigned with all-zeros states (usually created by 
    ``allocState({})`` method), then attention should be paid, since this
    machine states only can propagate from the first element, i.e. ``SOURCE``
    (``from_element`` parameter of ``run()`` or ``propagate()`` should be 0),
    or errors happen; the better initialization should be passing one of
    keyword parameters of ``machine`` and ``latfile`` to initialize the
    states to be significant for the ``propagate()`` method.
    """
    def __init__(self, s=None, **kws):
        _mstates = kws.get('mstates', None)
        _machine = kws.get('machine', None)
        _latfile = kws.get('latfile', None)
        self._states = None

        if s is None:
            if _mstates is not None:
                self._states = _mstates.clone()
            else:
                _m = miscutils.machine_setter(_latfile, _machine, 'MachineStates')
                if _m is not None:
                    self._states = _m.allocState({})
        else:
            self._states = s

        if self._states is not None:
            if is_zeros_states(self._states):
                _m = miscutils.machine_setter(_latfile, _machine, 'MachineStates')
                if _m is not None:
                    _m.propagate(self._states, 0, 1)
                else:
                    _LOGGER.warning(
                        "MachineStates: The initial machine states is 0, true values could be obtained with more information.")

    @property
    def mstates(self):
        """flame._internal.State: FLAME Machine states object"""
        return self._states

    @mstates.setter
    def mstates(self, s):
        self._states = s

    @property
    def pos(self):
        """float: longitudinally propagating position, [m]"""
        return getattr(self._states, 'pos')

    @pos.setter
    def pos(self, x):
        setattr(self._states, 'pos', x)

    @property
    def ref_beta(self):
        """float: speed in the unit of light velocity in vacuum of reference
        charge state, Lorentz beta"""
        return getattr(self._states, 'ref_beta')

    @ref_beta.setter
    def ref_beta(self, x):
        setattr(self._states, 'ref_beta', x)

    @property
    def ref_bg(self):
        """float: multiplication of beta and gamma of reference charge state"""
        return getattr(self._states, 'ref_bg')

    @ref_bg.setter
    def ref_bg(self, x):
        setattr(self._states, 'ref_bg', x)

    @property
    def ref_gamma(self):
        """float: relativistic energy of reference charge state, Lorentz gamma"""
        return getattr(self._states, 'ref_gamma')

    @ref_gamma.setter
    def ref_gamma(self, x):
        setattr(self._states, 'ref_gamma', x)

    @property
    def ref_IonEk(self):
        """float: kinetic energy of reference charge state, [eV/u]
        """
        return getattr(self._states, 'ref_IonEk')

    @ref_IonEk.setter
    def ref_IonEk(self, x):
        setattr(self._states, 'ref_IonEk', x)

    @property
    def ref_IonEs(self):
        """float: rest energy of reference charge state, [eV/u]
        """
        return getattr(self._states, 'ref_IonEs')

    @ref_IonEs.setter
    def ref_IonEs(self, x):
        setattr(self._states, 'ref_IonEs', x)

    @property
    def ref_IonQ(self):
        """int: macro particle number of reference charge state
        """
        return getattr(self._states, 'ref_IonQ')

    @ref_IonQ.setter
    def ref_IonQ(self, x):
        setattr(self._states, 'ref_IonQ', x)

    @property
    def ref_IonW(self):
        """float: total energy of reference charge state, [eV/u],
        i.e. :math:`W = E_s + E_k`"""
        return getattr(self._states, 'ref_IonW')

    @ref_IonW.setter
    def ref_IonW(self, x):
        setattr(self._states, 'ref_IonW', x)

    @property
    def ref_IonZ(self):
        """float: reference charge state, measured by charge to mass ratio,
        e.g. :math:`^{33^{+}}_{238}U: Q[33]/A[238]`"""
        return getattr(self._states, 'ref_IonZ')

    @ref_IonZ.setter
    def ref_IonZ(self, x):
        setattr(self._states, 'ref_IonZ', x)

    @property
    def ref_phis(self):
        """float: absolute synchrotron phase of reference charge state,
        [rad]"""
        return getattr(self._states, 'ref_phis')

    @ref_phis.setter
    def ref_phis(self, x):
        setattr(self._states, 'ref_phis', x)

    @property
    def ref_SampleIonK(self):
        """float: wave-vector in cavities with different beta values of
        reference charge state"""
        return getattr(self._states, 'ref_SampleIonK')

    @ref_SampleIonK.setter
    def ref_SampleIonK(self, x):
        setattr(self._states, 'ref_SampleIonK', x)

    @property
    def beta(self):
        """Array: speed in the unit of light velocity in vacuum of all charge
        states, Lorentz beta"""
        return getattr(self._states, 'beta')

    @beta.setter
    def beta(self, x):
        setattr(self._states, 'beta', x)

    @property
    def bg(self):
        """Array: multiplication of beta and gamma of all charge states"""
        return getattr(self._states, 'bg')

    @bg.setter
    def bg(self, x):
        setattr(self._states, 'bg', x)

    @property
    def gamma(self):
        """Array: relativistic energy of all charge states, Lorentz gamma"""
        return getattr(self._states, 'gamma')

    @gamma.setter
    def gamma(self, x):
        setattr(self._states, 'gamma', x)

    @property
    def IonEk(self):
        """Array: kinetic energy of all charge states, [eV/u]"""
        return getattr(self._states, 'IonEk')

    @IonEk.setter
    def IonEk(self, x):
        setattr(self._states, 'IonEk', x)

    @property
    def IonEs(self):
        """Array: rest energy of all charge states, [eV/u]"""
        return getattr(self._states, 'IonEs')

    @IonEs.setter
    def IonEs(self, x):
        setattr(self._states, 'IonEs', x)

    @property
    def IonQ(self):
        """Array: macro particle number of all charge states

        Note
        ----
        This is what ``NCharge`` means in the FLAME lattice file.
        """
        return getattr(self._states, 'IonQ')

    @IonQ.setter
    def IonQ(self, x):
        setattr(self._states, 'IonQ', x)

    @property
    def IonW(self):
        """Array: total energy of all charge states, [eV/u],
        i.e. :math:`W = E_s + E_k`"""
        return getattr(self._states, 'IonW')

    @IonW.setter
    def IonW(self, x):
        setattr(self._states, 'IonW', x)

    @property
    def IonZ(self):
        """Array: all charge states, measured by charge to mass ratio

        Note
        ----
        This is what ``IonChargeStates`` means in the FLAME lattice file.
        """
        return getattr(self._states, 'IonZ')

    @IonZ.setter
    def IonZ(self, x):
        setattr(self._states, 'IonZ', x)

    @property
    def phis(self):
        """Array: absolute synchrotron phase of all charge states, [rad]"""
        return getattr(self._states, 'phis')

    @phis.setter
    def phis(self, x):
        setattr(self._states, 'phis', x)

    @property
    def SampleIonK(self):
        """Array: wave-vector in cavities with different beta values of all
        charge states"""
        return getattr(self._states, 'SampleIonK')

    @SampleIonK.setter
    def SampleIonK(self, x):
        setattr(self._states, 'SampleIonK', x)

    @property
    def moment0_env(self):
        """Array: weight average of centroid for all charge states, array of
        ``[x, x', y, y', phi, dEk, 1]``, with the units of
        ``[mm, rad, mm, rad, rad, MeV/u, 1]``.

        Note
        ----
        The physics meanings for each column are:

            * ``x``: x position in transverse plane;
            * ``x'``: x divergence;
            * ``y``: y position in transverse plane;
            * ``y'``: y divergence;
            * ``phi``: longitudinal beam length, measured in RF frequency;
            * ``dEk``: kinetic energy deviation w.r.t. reference charge state;
            * ``1``: should be always 1, for the convenience of handling 
              corrector (i.e. ``orbtrim`` element)
        """
        return getattr(self._states, 'moment0_env')

    @moment0_env.setter
    def moment0_env(self, x):
        setattr(self._states, 'moment0_env', x)

    @property
    def moment0_rms(self):
        """Array: rms beam envelope, part of statistical results from
        ``moment1``.

        Note
        ----
        The square of ``moment0_rms`` should be equal to the diagonal
        elements of ``moment1``.

        See Also
        --------
        moment1 : correlation tensor of all charge states
        """
        return getattr(self._states, 'moment0_rms')

    @property
    def moment0(self):
        """Array: centroid for all charge states, array of
        ``[x, x', y, y', phi, dEk, 1]``"""
        return getattr(self._states, 'moment0')

    @moment0.setter
    def moment0(self, x):
        setattr(self._states, 'moment0', x)

    @property
    def moment1(self):
        r"""Array: correlation tensor of all charge states, for each charge
        state, the correlation matrix could be written as:

        .. math::

           \begin{array}{ccccccc}
               \color{red}{\left<x \cdot x\right>} & \left<x \cdot x'\right> & \left<x \cdot y\right> & \left<x \cdot y'\right> & \left<x \cdot \phi\right> & \left<x \cdot \delta E_k\right> & 0 \\
               \left<x'\cdot x\right> & \color{red}{\left<x'\cdot x'\right>} & \left<x'\cdot y\right> & \left<x'\cdot y'\right> & \left<x'\cdot \phi\right> & \left<x'\cdot \delta E_k\right> & 0 \\
               \left<y \cdot x\right> & \left<y \cdot x'\right> & \color{red}{\left<y \cdot y\right>} & \left<y \cdot y'\right> & \left<y \cdot \phi\right> & \left<y \cdot \delta E_k\right> & 0 \\
               \left<y'\cdot x\right> & \left<y'\cdot x'\right> & \left<y'\cdot y\right> & \color{red}{\left<y'\cdot y'\right>} & \left<y'\cdot \phi\right> & \left<y'\cdot \delta E_k\right> & 0 \\
               \left<\phi \cdot x\right> & \left<\phi \cdot x'\right> & \left<\phi \cdot y\right> & \left<\phi \cdot y'\right> & \color{red}{\left<\phi \cdot \phi\right>} & \left<\phi \cdot \delta E_k\right> & 0 \\
               \left<\delta E_k  \cdot x\right> & \left<\delta E_k  \cdot x'\right> & \left<\delta E_k  \cdot y\right> & \left<\delta E_k  \cdot y'\right> & \left<\delta E_k  \cdot \phi\right> & \color{red}{\left<\delta E_k  \cdot \delta E_k\right>} & 0 \\
               0                    & 0                     & 0                    & 0                     & 0                       & 0                      & 0
           \end{array}
        """
        return getattr(self._states, 'moment1')

    @moment1.setter
    def moment1(self, x):
        setattr(self._states, 'moment1', x)

    @property
    def x0(self):
        """Array: x centroid for all charge states, [mm]"""
        return self._states.moment0[0]

    @property
    def xp0(self):
        """Array: x centroid divergence for all charge states, [rad]"""
        return self._states.moment0[1]

    @property
    def y0(self):
        """Array: y centroid for all charge states, [mm]"""
        return self._states.moment0[2]

    @property
    def yp0(self):
        """Array: y centroid divergence for all charge states, [rad]"""
        return self._states.moment0[3]

    @property
    def phi0(self):
        """Array: longitudinal beam length, measured in RF frequency for all
        charge states, [rad]"""
        return self._states.moment0[4]

    @property
    def dEk0(self):
        """Array: kinetic energy deviation w.r.t. reference charge state,
        for all charge states, [MeV/u]"""
        return self._states.moment0[5]

    @property
    def x0_env(self):
        """Array: weight average of all charge states for ``x``, [mm]"""
        return self._states.moment0_env[0]

    @property
    def xp0_env(self):
        """Array: weight average of all charge states for ``x'``, [rad]"""
        return self._states.moment0_env[1]

    @property
    def y0_env(self):
        """Array: weight average of all charge states for ``y``, [mm]"""
        return self._states.moment0_env[2]

    @property
    def yp0_env(self):
        """Array: weight average of all charge states for ``y'``, [rad]"""
        return self._states.moment0_env[3]

    @property
    def phi0_env(self):
        """Array: weight average of all charge states for :math:`\phi`,
        [mm]"""
        return self._states.moment0_env[4]

    @property
    def dEk0_env(self):
        """Array: weight average of all charge states for :math:`\delta E_k`,
        [MeV/u]"""
        return self._states.moment0_env[5]

    @property
    def x0_rms(self):
        """Array: general rms beam envelope for ``x``, [mm]"""
        return self._states.moment0_rms[0]

    @property
    def xp0_rms(self):
        """Array: general rms beam envelope for ``x'``, [rad]"""
        return self._states.moment0_rms[1]

    @property
    def y0_rms(self):
        """Array: general rms beam envelope for ``y``, [mm]"""
        return self._states.moment0_rms[2]

    @property
    def yp0_rms(self):
        """Array: general rms beam envelope for ``y'``, [rad]"""
        return self._states.moment0_rms[3]

    @property
    def phi0_rms(self):
        """Array: general rms beam envelope for :math:`\phi`, [mm]"""
        return self._states.moment0_rms[4]

    @property
    def dEk0_rms(self):
        """Array: general rms beam envelope for :math:`\delta E_k`, [MeV/u]"""
        return self._states.moment0_rms[5]
    
    @property
    def last_caviphi0(self):
        """float: Last RF cavity's driven phase, [deg]"""
        try:
            ret = self._states.last_caviphi0
        except:
            print("python-flame version should be at least 1.1.1")
            ret = None
        return ret

    def clone(self):
        """Return a copy of machine states
        """
        return MachineStates(self._states.clone())

    def __repr__(self):
        try:
            moment0_env = ','.join(["{0:.6g}".format(i) for i in self.moment0_env])
            return "State: moment0 mean=[7]({})".format(moment0_env)
        except AttributeError:
            return "Incompleted initializaion."
