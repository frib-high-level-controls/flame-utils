#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Abstracted FLAME beam state class.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import flame
import logging
import numpy as np

try:
    basestring
except NameError:
    basestring = str

from flame_utils.misc import is_zeros_states
from flame_utils.misc import machine_setter
from flame_utils.misc import alias

__authors__ = "Tong Zhang"
__copyright__ = "(c) 2016-2017, Facility for Rare Isotope beams, " \
                "Michigan State University"
__contact__ = "Tong Zhang <zhangt@frib.msu.edu>"


_LOGGER = logging.getLogger(__name__)

KEY_MAPPING = {
        'IonChargeStates': 'IonZ',
        'IonEk': 'ref_IonEk',
        'IonEs': 'ref_IonEs',
        'NCharge': 'IonQ',
}


@alias
class BeamState(object):
    """FLAME beam state, from which simulated results could be retrieved.

    All attributes of states:

     - ``pos``,
     - ``ref_beta``, ``ref_bg``, ``ref_gamma``, ``ref_IonEk``, ``ref_IonEs``,
       ``ref_IonQ``, ``ref_IonW``, ``ref_IonZ``, ``ref_phis``,
       ``ref_SampleIonK``,
     - ``beta``, ``bg``, ``gamma``, ``IonEk``, ``IonEs``, ``IonQ``, ``IonW``,
       ``IonZ``, ``phis``, ``SampleIonK``,
     - ``moment0`` (``cenvector_all``), ``moment0_rms`` (``rmsvector``), ``moment0_env`` (``cenvector``),
     - ``moment1`` (``beammatrix_all``), ``moment1_env`` (``beammatrix``),
     - ``dEk0`` (``dEkcen_all``, ``zpcen_all``), ``dEk0_env`` (``dEkcen``, ``zpcen``),
       ``dEkrms_all`` (``zprms_all``), ``dEk0_rms`` (``dEkrms``, ``zprms``)
     - ``phi0`` (``phicen_all``, ``zcen_all``), ``phi0_env`` (``phicen``, ``zcen``),
       ``phirms_all`` (``zrms_all``), ``phi0_rms`` (``phirms``, ``zrms``)
     - ``x0`` (``xcen_all``), ``x0_env`` (``xcen``), ``xrms_all``, ``x0_rms`` (``xrms``)
     - ``xp0`` (``xpcen_all``), ``xp0_env`` (``xpcen``), ``xprms_all``, ``xp0_rms`` (``xprms``)
     - ``y0`` (``ycen_all``), ``y0_env`` (``ycen``), ``yrms_all``, ``y0_rms`` (``yrms``)
     - ``yp0`` (``ypcen_all``), ``yp0_env`` (``ypcen``), ``yprms_all``, ``yp0_rms`` (``yprms``)
     - ``last_caviphi0``
     - ``xemittance_all`` (``xeps_all``), ``xemittance`` (``xeps``),
       ``xnemittance_all`` (``xepsn_all``), ``xnemittance`` (``xepsn``)
     - ``yemittance_all`` (``yeps_all``), ``yemittance`` (``yeps``),
       ``ynemittance_all`` (``yepsn_all``), ``ynemittance`` (``yepsn``)
     - ``zemittance_all`` (``zeps_all``), ``zemittance`` (``zeps``),
       ``znemittance_all`` (``zepsn_all``), ``znemittance`` (``zepsn``)
     - ``xtwiss_beta_all`` (``xtwsb_all``), ``xtwiss_beta`` (``xtwsb``),
       ``xtwiss_alpha_all`` (``xtwsa_all``),  ``xtwiss_alpha`` (``xtwsa``)
     - ``ytwiss_beta_all`` (``ytwsb_all``), ``ytwiss_beta`` (``ytwsb``),
       ``ytwiss_alpha_all`` (``ytwsa_all``),  ``ytwiss_alpha`` (``ytwsa``)
     - ``ztwiss_beta_all`` (``ztwsb_all``), ``ztwiss_beta`` (``ztwsb``),
       ``ztwiss_alpha_all`` (``ztwsa_all``),  ``ztwiss_alpha`` (``ztwsa``)
     - ``couple_xy_all`` (``cxy_all``), ``couple_xy`` (``cxy``),
       ``couple_xpy_all`` (``cxpy_all``), ``couple_xpy`` (``cxpy``),
       ``couple_xyp_all`` (``cxyp_all``), ``couple_xyp`` (``cxyp``),
       ``couple_xpyp_all`` (``cxpyp_all``), ``couple_xpyp`` (``cxpyp``)

    Warning
    -------
    1. These attributes are only valid for the case of ``sim_type`` being
       defined as ``MomentMatrix``, which is de facto the exclusive option
       used at FRIB.
    2. If the attribute is an array, new array value should be assigned
       instead of by element indexing way, e.g.

       >>> bs = BeamState(s)
       >>> print(bs.moment0)
       array([[ -7.88600000e-04],
              [  1.08371000e-05],
              [  1.33734000e-02],
              [  6.67853000e-06],
              [ -1.84773000e-04],
              [  3.09995000e-04],
              [  1.00000000e+00]])
       >>> # the right way to just change the first element of the array
       >>> m_tmp = bs.moment0
       >>> m_tmp[0] = 0
       >>> bs.moment0 = m_tmp
       >>> print(bs.moment0)
       array([[  0.00000000e+00],
              [  1.08371000e-05],
              [  1.33734000e-02],
              [  6.67853000e-06],
              [ -1.84773000e-04],
              [  3.09995000e-04],
              [  1.00000000e+00]])
       >>> # while this way does not work: ms.moment0[0] = 0


    Parameters
    ----------
    s :
        FLAME state object, created by `allocState()`.

    Keyword Arguments
    -----------------
    bmstate :
        BeamState object, priority: high
    machine :
        FLAME machine object, priority: middle
    latfile :
        FLAME lattice file name, priority: low

    Note
    ----
    If more than one keyword parameters are provided,
    the selection policy follows the priority from high to low.

    Warning
    -------
    If only ``s`` is assigned with all-zeros states (usually created by
    ``allocState({})`` method), then please note that this state can only
    propagate from the first element, i.e. ``SOURCE``
    (``from_element`` parameter of ``run()`` or ``propagate()`` should be 0),
    or errors happen; the better initialization should be passing one of
    keyword parameters of ``machine`` and ``latfile`` to initialize the
    state to be significant for the ``propagate()`` method.
    """
    _aliases = {
        'xcen_all': 'x0',
        'ycen_all': 'y0',
        'xpcen_all': 'xp0',
        'ypcen_all': 'yp0',
        'zcen_all': 'phi0',
        'zpcen_all': 'dEk0',
        'phicen_all': 'phi0',
        'dEkcen_all': 'dEk0',
        'xrms': 'x0_rms',
        'yrms': 'y0_rms',
        'xprms': 'xp0_rms',
        'yprms': 'yp0_rms',
        'zrms': 'phi0_rms',
        'zprms': 'dEk0_rms',
        'phirms': 'phi0_rms',
        'dEkrms': 'dEk0_rms',
        'zrms_all': 'phirms_all',
        'zprms_all': 'dEkrms_all',
        'xcen': 'x0_env',
        'ycen': 'y0_env',
        'xpcen': 'xp0_env',
        'ypcen': 'yp0_env',
        'zcen': 'phi0_env',
        'zpcen': 'dEk0_env',
        'phicen': 'phi0_env',
        'dEkcen': 'dEk0_env',
        'cenvector': 'moment0_env',
        'cenvector_all': 'moment0',
        'rmsvector': 'moment0_rms',
        'beammatrix_all': 'moment1',
        'beammatrix': 'moment1_env',
        'xeps': 'xemittance',
        'yeps': 'yemittance',
        'zeps': 'zemittance',
        'xeps_all': 'xemittance_all',
        'yeps_all': 'yemittance_all',
        'zeps_all': 'zemittance_all',
        'xepsn': 'xnemittance',
        'yepsn': 'ynemittance',
        'zepsn': 'znemittance',
        'xepsn_all': 'xnemittance_all',
        'yepsn_all': 'ynemittance_all',
        'zepsn_all': 'znemittance_all',
        'xtwsb': 'xtwiss_beta',
        'ytwsb': 'ytwiss_beta',
        'ztwsb': 'ztwiss_beta',
        'xtwsb_all': 'xtwiss_beta_all',
        'ytwsb_all': 'ytwiss_beta_all',
        'ztwsb_all': 'ztwiss_beta_all',
        'xtwsa': 'xtwiss_alpha',
        'ytwsa': 'ytwiss_alpha',
        'ztwsa': 'ztwiss_alpha',
        'xtwsa_all': 'xtwiss_alpha_all',
        'ytwsa_all': 'ytwiss_alpha_all',
        'ztwsa_all': 'ztwiss_alpha_all',
        'cxy': 'couple_xy',
        'cxpy': 'couple_xpy',
        'cxyp': 'couple_xyp',
        'cxpyp': 'couple_xpyp',
        'cxy_all': 'couple_xy_all',
        'cxpy_all': 'couple_xpy_all',
        'cxyp_all': 'couple_xyp_all',
        'cxpyp_all': 'couple_xpyp_all',
    }
    def __init__(self, s=None, **kws):
        _bmstate = kws.get('bmstate', None)
        _machine = kws.get('machine', None)
        _latfile = kws.get('latfile', None)
        self._states = None

        if s is None:
            if _bmstate is not None:
                self.state = _bmstate
            else:
                _m = machine_setter(_latfile, _machine, 'BeamState')
                if _m is not None:
                    self._states = _m.allocState({})
        else:
            self._states = s

        if self._states is not None:
            if is_zeros_states(self._states):
                _m = machine_setter(_latfile, _machine, 'BeamState')
                if _m is not None:
                    _m.propagate(self._states, 0, 1)
                else:
                    _LOGGER.warning(
                    "BeamState: " \
                     "Zeros initial states, get true values by " \
                     "parameter '_latfile' or '_machine'.")

    @property
    def state(self):
        """flame._internal.State: FLAME state object, also could be
        initialized with BeamState object"""
        return self._states

    @state.setter
    def state(self, s):
        if isinstance(s, flame._internal.State):
            self._states = s.clone()
        elif isinstance(s, BeamState):
            self._states = s.clone().state

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
        charge state, Lorentz beta, [1]"""
        return getattr(self._states, 'ref_beta')

    @ref_beta.setter
    def ref_beta(self, x):
        setattr(self._states, 'ref_beta', x)

    @property
    def ref_bg(self):
        """float: multiplication of beta and gamma of reference charge state, [1]"""
        return getattr(self._states, 'ref_bg')

    @ref_bg.setter
    def ref_bg(self, x):
        setattr(self._states, 'ref_bg', x)

    @property
    def ref_gamma(self):
        """float: relativistic energy of reference charge state, Lorentz gamma, [1]"""
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
        """int: macro particle number of reference charge state, [1]
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
        e.g. :math:`^{33^{+}}_{238}U: Q(33)/A(238)`, [Q/A]"""
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
        reference charge state, [rad]"""
        return getattr(self._states, 'ref_SampleIonK')

    @ref_SampleIonK.setter
    def ref_SampleIonK(self, x):
        setattr(self._states, 'ref_SampleIonK', x)

    @property
    def beta(self):
        """Array: speed in the unit of light velocity in vacuum of all charge
        states, Lorentz beta, [1]"""
        return getattr(self._states, 'beta')

    @beta.setter
    def beta(self, x):
        setattr(self._states, 'beta', x)

    @property
    def bg(self):
        """Array: multiplication of beta and gamma of all charge states, [1]"""
        return getattr(self._states, 'bg')

    @bg.setter
    def bg(self, x):
        setattr(self._states, 'bg', x)

    @property
    def gamma(self):
        """Array: relativistic energy of all charge states, Lorentz gamma, [1]"""
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
        charge states, [rad]"""
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

        - ``x``: x position in transverse plane;
        - ``x'``: x divergence;
        - ``y``: y position in transverse plane;
        - ``y'``: y divergence;
        - ``phi``: longitudinal beam length, measured in RF frequency;
        - ``dEk``: kinetic energy deviation w.r.t. reference charge state;
        - ``1``: should be always 1, for the convenience of handling
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
    def moment1_env(self):
        """Array: averaged correlation tensor of all charge states"""
        return getattr(self._states, 'moment1_env')

    @moment1_env.setter
    def moment1_env(self, x):
        setattr(self._states, 'moment1_env', x)

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
        [rad]"""
        return self._states.moment0_env[4]

    @property
    def dEk0_env(self):
        """Array: weight average of all charge states for :math:`\delta E_k`,
        [MeV/u]"""
        return self._states.moment0_env[5]

    @property
    def xrms_all(self):
        """Array: general rms beam envelope for ``x`` of all charge states, [mm]"""
        return np.sqrt(self._states.moment1[0, 0, :])

    @property
    def xprms_all(self):
        """Array: general rms beam envelope for ``x'`` of all charge states, [rad]"""
        return np.sqrt(self._states.moment1[1, 1, :])

    @property
    def yrms_all(self):
        """Array: general rms beam envelope for ``y`` of all charge states, [mm]"""
        return np.sqrt(self._states.moment1[2, 2, :])

    @property
    def yprms_all(self):
        """Array: general rms beam envelope for ``y'`` of all charge states, [rad]"""
        return np.sqrt(self._states.moment1[3, 3, :])

    @property
    def phirms_all(self):
        """Array: general rms beam envelope for :math:`\phi` of all charge states, [rad]"""
        return np.sqrt(self._states.moment1[4, 4, :])

    @property
    def dEkrms_all(self):
        """Array: general rms beam envelope for :math:`\delta E_k` of all charge states, [MeV/u]"""
        return np.sqrt(self._states.moment1[5, 5, :])

    @property
    def x0_rms(self):
        """float: general rms beam envelope for ``x``, [mm]"""
        return self._states.moment0_rms[0]

    @property
    def xp0_rms(self):
        """float: general rms beam envelope for ``x'``, [rad]"""
        return self._states.moment0_rms[1]

    @property
    def y0_rms(self):
        """float: general rms beam envelope for ``y``, [mm]"""
        return self._states.moment0_rms[2]

    @property
    def yp0_rms(self):
        """float: general rms beam envelope for ``y'``, [rad]"""
        return self._states.moment0_rms[3]

    @property
    def phi0_rms(self):
        """float: general rms beam envelope for :math:`\phi`, [rad]"""
        return self._states.moment0_rms[4]

    @property
    def dEk0_rms(self):
        """float: general rms beam envelope for :math:`\delta E_k`, [MeV/u]"""
        return self._states.moment0_rms[5]

    @property
    def last_caviphi0(self):
        """float: Last RF cavity's driven phase, [deg]"""
        try:
            ret = self._states.last_caviphi0
        except AttributeError:
            print("python-flame version should be at least 1.1.1")
            ret = None
        return ret

    def clone(self):
        """Return a copy of Beamstate object.
        """
        return BeamState(self._states.clone())

    def __repr__(self):
        try:
            moment0_env = ','.join(["{0:.6g}".format(i) for i in self.moment0_env])
            return "BeamState: moment0 mean=[7]({})".format(moment0_env)
        except AttributeError:
            return "Incompleted initializaion."

    @property
    def xemittance(self):
        """float: weight average of geometrical x emittance, [mm-mrad]"""
        return np.sqrt(np.linalg.det(self._states.moment1_env[0:2, 0:2]))*1e3

    @property
    def yemittance(self):
        """float: weight average of geometrical y emittance, [mm-mrad]"""
        return np.sqrt(np.linalg.det(self._states.moment1_env[2:4, 2:4]))*1e3

    @property
    def zemittance(self):
        """float: weight average of geometrical z emittance, [rad-MeV/u]"""
        return np.sqrt(np.linalg.det(self._states.moment1_env[4:6, 4:6]))

    @property
    def xemittance_all(self):
        """Array: geometrical x emittance of all charge states, [mm-mrad]"""
        return np.array([np.sqrt(np.linalg.det(self._states.moment1[0:2, 0:2, i]))*1e3 for i in range(len(self.bg))])

    @property
    def yemittance_all(self):
        """Array: geometrical y emittance of all charge states, [mm-mrad]"""
        return np.array([np.sqrt(np.linalg.det(self._states.moment1[2:4, 2:4, i]))*1e3 for i in range(len(self.bg))])

    @property
    def zemittance_all(self):
        """Array: geometrical z emittance of all charge states, [rad-MeV/u]"""
        return np.array([np.sqrt(np.linalg.det(self._states.moment1[4:6, 4:6, i])) for i in range(len(self.bg))])

    @property
    def xnemittance(self):
        """float: weight average of normalized x emittance, [mm-mrad]"""
        return self.ref_bg*self.xeps

    @property
    def ynemittance(self):
        """float: weight average of normalized y emittance, [mm-mrad]"""
        return self.ref_bg*self.yeps

    @property
    def znemittance(self):
        """float: weight average of normalized z emittance, [rad-MeV/u]"""
        return self.ref_bg*self.zeps

    @property
    def xnemittance_all(self):
        """float: normalized x emittance of all charge states, [mm-mrad]"""
        return self.ref_bg*self.xeps_all

    @property
    def ynemittance_all(self):
        """float: normalized y emittance of all charge states, [mm-mrad]"""
        return self.ref_bg*self.yeps_all

    @property
    def znemittance_all(self):
        """float: normalized z emittance of all charge states, [rad-MeV/u]"""
        return self.ref_bg*self.zeps_all

    @property
    def xtwiss_beta(self):
        """float: weight average of twiss beta x, [m/rad]"""
        return self._states.moment1_env[0, 0]/self.xeps

    @property
    def ytwiss_beta(self):
        """float: weight average of twiss beta y, [m/rad]"""
        return self._states.moment1_env[2, 2]/self.yeps

    @property
    def ztwiss_beta(self):
        """float: weight average of twiss beta z, [rad/MeV/u]"""
        return self._states.moment1_env[4, 4]/self.zeps

    @property
    def xtwiss_beta_all(self):
        """float: twiss beta x of all charge states, [m/rad]"""
        return self._states.moment1[0, 0, :]/self.xeps_all

    @property
    def ytwiss_beta_all(self):
        """float: twiss beta y of all charge states, [m/rad]"""
        return self._states.moment1[2, 2, :]/self.yeps_all

    @property
    def ztwiss_beta_all(self):
        """float: twiss beta z of all charge states, [rad/MeV/u]"""
        return self._states.moment1[4, 4, :]/self.zeps_all

    @property
    def xtwiss_alpha(self):
        """float: weight average of twiss alpha x, [1]"""
        return -self._states.moment1_env[0, 1]/self.xeps*1e3

    @property
    def ytwiss_alpha(self):
        """float: weight average of twiss alpha y, [1]"""
        return -self._states.moment1_env[2, 3]/self.yeps*1e3

    @property
    def ztwiss_alpha(self):
        """float: weight average of twiss alpha z, [1]"""
        return -self._states.moment1_env[4, 5]/self.zeps

    @property
    def xtwiss_alpha_all(self):
        """float: twiss alpha x of all charge states, [1]"""
        return -self._states.moment1[0, 1, :]/self.xeps_all*1e3

    @property
    def ytwiss_alpha_all(self):
        """float: twiss alpha y of all charge states, [1]"""
        return -self._states.moment1[2, 3, :]/self.yeps_all*1e3

    @property
    def ztwiss_alpha_all(self):
        """float: twiss alpha z of all charge states, [1]"""
        return -self._states.moment1[4, 5, :]/self.zeps_all

    @property
    def couple_xy(self):
        """float: weight average of normalized x-y coupling term, [1]"""
        return self.get_couple('x', 'y', cs=-1)

    @property
    def couple_xpy(self):
        """float: weight average of normalized xp-y coupling term, [1]"""
        return self.get_couple('xp', 'y', cs=-1)

    @property
    def couple_xyp(self):
        """float: weight average of normalized x-yp coupling term, [1]"""
        return self.get_couple('x', 'yp', cs=-1)

    @property
    def couple_xpyp(self):
        """float: weight average of normalized xp-yp coupling term, [1]"""
        return self.get_couple('xp', 'yp', cs=-1)

    @property
    def couple_xy_all(self):
        """float: normalized x-y coupling term of all charge states, [1]"""
        return np.array([self.get_couple('x', 'y', cs=i) for i in range(len(self.bg))])

    @property
    def couple_xpy_all(self):
        """float: normalized xp-y coupling term of all charge states, [1]"""
        return np.array([self.get_couple('xp', 'y', cs=i) for i in range(len(self.bg))])

    @property
    def couple_xyp_all(self):
        """float: normalized x-yp coupling term of all charge states, [1]"""
        return np.array([self.get_couple('x', 'yp', cs=i) for i in range(len(self.bg))])

    @property
    def couple_xpyp_all(self):
        """float: normalized xp-yp coupling term of all charge states, [1]"""
        return np.array([self.get_couple('xp', 'yp', cs=i) for i in range(len(self.bg))])

    def set_twiss(self, coor, alpha = None, beta = None, rmssize = None, emittance = None, nemittance = None, cs = 0):
        """Set moment1 matrix by using Twiss parameter.

        Parameters
        ----------
        coor : str
            Coordinate of the twiss parameter,　'x', 'y', or 'z'.
        alpha : float
            Twiss alpha, [1].
        beta : float
            Twiss beta, [m/rad] for 'x' and 'y', [rad/MeV/u] for 'z'.
        rmssize : float
            RMS size of the real space, [mm] of 'x' and 'y', [rad] for 'z'.
        emittance : float
            Geometrical (Unnormalized) emittance, [mm-mrad] for 'x' and 'y', [rad-MeV/u] for 'z'.
        nemittance : float
            Normalized emittance, [mm-mrad] for 'x' and 'y', [rad-MeV/u] for 'z'.
        cs : int
            Index of the charge state to set parameter.

        Note
        ----
        'nemittance' is ignored if both 'emittance' and 'nemittance' are input.
        """
        eps = emittance
        neps = nemittance
        if eps is None and neps is None:
            eps = getattr(self, coor + 'emittance_all')[cs]
        elif eps is not None and neps is not None:
            _LOGGER.warning("'neps' is ignored by 'eps' input.")

        if eps is None:
            gam = 1.0 + self.ref_IonEk/self.ref_IonEs
            bg  = np.sqrt(gam*gam - 1.0)
            eps = neps/bg

        if beta is None and rmssize is None:
            beta = getattr(self, coor + 'twiss_beta_all')[cs]
        elif beta is None and rmssize is not None:
            beta = rmssize*rmssize/eps
        elif beta is not None and rmssize is None:
            beta = float(beta)
        else:
            _LOGGER.error("Invalid twiss input. It support to input only beta OR rmssize.")
            return None

        alpha = getattr(self, coor + 'twiss_alpha_all')[cs] if alpha is None else alpha

        mat = self._states.moment1
        if coor == 'x':
            idx = [0, 1]
            jdx = [2, 3, 4, 5]
            cpt = [[self.get_couple(i, j, cs = cs) for i in idx] for j in jdx]
            mat[0, 0, cs] = beta*eps
            mat[0, 1, cs] = mat[1, 0, cs] = -alpha*eps*1e-3
            mat[1, 1, cs] = (1.0 + alpha*alpha)/beta*eps*1e-6
        elif coor == 'y':
            idx = [2, 3]
            jdx = [0, 1, 4, 5]
            cpt = [[self.get_couple(i, j, cs = cs) for i in idx] for j in jdx]
            mat[2, 2, cs] = beta*eps
            mat[2, 3, cs] = mat[3, 2, cs] = -alpha*eps*1e-3
            mat[3, 3, cs] = (1.0 + alpha*alpha)/beta*eps*1e-6
        elif coor == 'z':
            idx = [4, 5]
            jdx = [0, 1, 2, 3]
            cpt = [[self.get_couple(i, j, cs = cs) for i in idx] for j in jdx]
            mat[4, 4, cs] = beta*eps
            mat[4, 5, cs] = mat[5, 4, cs] = -alpha*eps
            mat[5, 5, cs] = (1.0 + alpha*alpha)/beta*eps
        else:
            _LOGGER.error("Invalid coordinate type. It must be 'x', 'y', or 'z'.")
            return None

        self._states.moment1 = mat
        for j, cp in zip(jdx, cpt):
            for i, v in zip(idx, cp):
                self.set_couple(i, j, v, cs = cs)

    @staticmethod
    def _couple_index(coor1, coor2):
        """Get index from coordinate information"""
        crd = {'x': 0, 'xp': 1, 'y': 2, 'yp': 3, 'z': 4, 'zp': 5}

        if isinstance(coor1, basestring) and isinstance(coor2, basestring):
            if not coor1 in crd or not coor2 in crd:
                _LOGGER.error("Invalid coordinate type. It must be 'x', 'xp', 'y', 'yp', 'z', or 'zp'. ")
                return None

            c1 = crd[coor1]
            c2 = crd[coor2]
        else:
            c1 = int(coor1)
            c2 = int(coor2)

        if c1 == c2:
            _LOGGER.error("Invalid coordinate type. Combination of " + str(coor1) + " and " + str(coor2) + " is not coupling term.")
            return None

        return  c1, c2

    def get_couple(self, coor1, coor2, cs=0):
        """Get normalized coupling term of moment1 matrix

        Parameters
        ----------
        coor1 : str
            First Coordinate of the coupling term, 'x', xp, 'y', 'yp', 'z', or 'zp'.
        coor2 : str
            Second Coordinate of the coupling term, 'x', xp, 'y', 'yp', 'z', or 'zp'.
        cs : int
            Index of the charge state (-1 for weight average of all charge states).
        Returns
        -------
        term : float
            Normalized coupling term of ``coor1`` and ``coor2`` of ``cs``-th charge state.
        """
        r = self._couple_index(coor1, coor2)
        if r is not None:
            c1, c2 = r
        else:
            return None

        if cs == -1:
            mat = self._states.moment1_env
        else:
            mat = self._states.moment1[:, :, cs]
        fac = np.sqrt(mat[c1, c1]*mat[c2, c2])
        term = mat[c1, c2]/fac if fac != 0.0 else 0.0

        return term

    def set_couple(self, coor1, coor2, value=0.0, cs = 0):
        """Set normalized coupling term of moment1 matrix

        Parameters
        ----------
        coor1 : str
            First Coordinate of the coupling term, 'x', xp, 'y', 'yp', 'z', or 'zp'.
        coor2 : str
            Second Coordinate of the coupling term, 'x', xp, 'y', 'yp', 'z', or 'zp'.
        value : float
            Normalized coupling term, (-1 ~ +1) [1].
        cs : int
            Index of the charge state.
        """
        r = self._couple_index(coor1, coor2)
        if r is not None:
            c1, c2 = r
        else:
            return None

        mat = self._states.moment1
        fac = np.sqrt(mat[c1, c1, cs]*mat[c2, c2, cs])
        mat[c1, c2, cs] = mat[c2, c1, cs] = value*fac

        self._states.moment1 = mat

def generate_source(state, sconf=None):
    """Generate/Update FLAME source element from FLAME beam state object.

    Parameters
    ----------
    state :
        BeamState object, (also accept FLAME internal State object).
    sconf : dict
        Configuration of source element, if None, generate new one from state.

    Returns
    -------
    ret : dict
        FLAME source element configuration.

    Warning
    -------
    All zeros state may not produce reasonable result, for this case, the
    recommended way is to create a `BeamState` object with `latfile` or
    `machine` keyword parameter, e.g. `s = BeamState(s0, machine=m)`, then
    use `s` as the input of `generate_source`.

    See Also
    --------
    get_element : Get element from FLAME machine or lattice.
    """
    if sconf is not None:
        sconf_indx = sconf['index']
        sconf_prop = sconf['properties']

    else:
        sconf_indx = 0
        sconf_prop = {
                'name': 'S',
                'type': 'source',
                'matrix_variable': 'S',
                'vector_variable': 'P'
        }
    # update properties
    for k, v in KEY_MAPPING.items():
        sconf_prop[k] = getattr(state, v)
    # vector/matrix variables
    p = sconf_prop.get('vector_variable', None)
    s = sconf_prop.get('matrix_variable', None)
    for i in range(len(state.IonZ)):
        sconf_prop['{0}{1}'.format(p, i)] = state.moment0[:, i]
        sconf_prop['{0}{1}'.format(s, i)] = state.moment1[:, :, i].flatten()

    return {'index': sconf_indx, 'properties': sconf_prop}
