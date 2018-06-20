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

    - Reference beam information

        - :py:func:`pos <flame_utils.core.state.BeamState.pos>`,
          :py:func:`ref_IonZ <flame_utils.core.state.BeamState.ref_IonZ>`,
          :py:func:`ref_IonQ <flame_utils.core.state.BeamState.ref_IonQ>`,
          :py:func:`ref_IonEk <flame_utils.core.state.BeamState.ref_IonEk>`,
          :py:func:`ref_IonEs <flame_utils.core.state.BeamState.ref_IonEs>`,
          :py:func:`ref_IonW <flame_utils.core.state.BeamState.ref_IonW>`,
          :py:func:`ref_phis <flame_utils.core.state.BeamState.ref_phis>`,
          :py:func:`ref_beta <flame_utils.core.state.BeamState.ref_beta>`,
          :py:func:`ref_gamma <flame_utils.core.state.BeamState.ref_gamma>`,
          :py:func:`ref_bg <flame_utils.core.state.BeamState.ref_bg>`,
          :py:func:`ref_SampleIonK <flame_utils.core.state.BeamState.ref_SampleIonK>`,
          :py:func:`last_caviphi0 <flame_utils.core.state.BeamState.last_caviphi0>` (since version 1.1.1)

    - Actual beam information

        - :py:func:`IonZ <flame_utils.core.state.BeamState.IonZ>`,
          :py:func:`IonQ <flame_utils.core.state.BeamState.IonQ>`,
          :py:func:`IonEk <flame_utils.core.state.BeamState.IonEk>`,
          :py:func:`IonEs <flame_utils.core.state.BeamState.IonEs>`,
          :py:func:`IonW <flame_utils.core.state.BeamState.IonW>`,
          :py:func:`phis <flame_utils.core.state.BeamState.phis>`,
          :py:func:`beta <flame_utils.core.state.BeamState.beta>`,
          :py:func:`gamma <flame_utils.core.state.BeamState.gamma>`,
          :py:func:`bg <flame_utils.core.state.BeamState.bg>`,
          :py:func:`SampleIonK <flame_utils.core.state.BeamState.SampleIonK>`

        - :py:func:`moment0 <flame_utils.core.state.BeamState.moment0>`
          (:py:func:`cenvector_all <flame_utils.core.state.BeamState.cenvector_all>`),
          :py:func:`moment0_env <flame_utils.core.state.BeamState.moment0_env>`
          (:py:func:`cenvector <flame_utils.core.state.BeamState.cenvector>`),
          :py:func:`moment0_rms <flame_utils.core.state.BeamState.moment0_rms>`
          (:py:func:`rmsvector <flame_utils.core.state.BeamState.rmsvector>`),
          :py:func:`moment1 <flame_utils.core.state.BeamState.moment1>`
          (:py:func:`beammatrix_all <flame_utils.core.state.BeamState.beammatrix_all>`),
          :py:func:`moment1_env <flame_utils.core.state.BeamState.moment1_env>`
          (:py:func:`beammatrix <flame_utils.core.state.BeamState.beammatrix>`)

        - :py:func:`x0 <flame_utils.core.state.BeamState.x0>`
          (:py:func:`xcen_all <flame_utils.core.state.BeamState.xcen_all>`),
          :py:func:`x0_env <flame_utils.core.state.BeamState.x0_env>`
          (:py:func:`xcen <flame_utils.core.state.BeamState.xcen>`),
          :py:func:`xrms_all <flame_utils.core.state.BeamState.xrms_all>`,
          :py:func:`x0_rms <flame_utils.core.state.BeamState.x0_rms>`
          (:py:func:`xrms <flame_utils.core.state.BeamState.xrms>`)

        - :py:func:`xp0 <flame_utils.core.state.BeamState.xp0>`
          (:py:func:`xpcen_all <flame_utils.core.state.BeamState.xpcen_all>`),
          :py:func:`xp0_env <flame_utils.core.state.BeamState.xp0_env>`
          (:py:func:`xpcen <flame_utils.core.state.BeamState.xpcen>`),
          :py:func:`xprms_all <flame_utils.core.state.BeamState.xprms_all>`,
          :py:func:`xp0_rms <flame_utils.core.state.BeamState.xp0_rms>`
          (:py:func:`xprms <flame_utils.core.state.BeamState.xprms>`)

        - :py:func:`y0 <flame_utils.core.state.BeamState.y0>`
          (:py:func:`ycen_all <flame_utils.core.state.BeamState.ycen_all>`),
          :py:func:`y0_env <flame_utils.core.state.BeamState.y0_env>`
          (:py:func:`ycen <flame_utils.core.state.BeamState.ycen>`),
          :py:func:`yrms_all <flame_utils.core.state.BeamState.yrms_all>`,
          :py:func:`y0_rms <flame_utils.core.state.BeamState.y0_rms>`
          (:py:func:`yrms <flame_utils.core.state.BeamState.yrms>`)

        - :py:func:`yp0 <flame_utils.core.state.BeamState.yp0>`
          (:py:func:`ypcen_all <flame_utils.core.state.BeamState.ypcen_all>`),
          :py:func:`yp0_env <flame_utils.core.state.BeamState.yp0_env>`
          (:py:func:`ypcen <flame_utils.core.state.BeamState.ypcen>`),
          :py:func:`yprms_all <flame_utils.core.state.BeamState.yprms_all>`,
          :py:func:`yp0_rms <flame_utils.core.state.BeamState.yp0_rms>`
          (:py:func:`yprms <flame_utils.core.state.BeamState.yprms>`)

        - :py:func:`phi0 <flame_utils.core.state.BeamState.phi0>`
          (:py:func:`phicen_all <flame_utils.core.state.BeamState.phicen_all>`),
          :py:func:`phi0_env <flame_utils.core.state.BeamState.phi0_env>`
          (:py:func:`phicen <flame_utils.core.state.BeamState.phicen>`),
          :py:func:`phirms_all <flame_utils.core.state.BeamState.phirms_all>`,
          :py:func:`phi0_rms <flame_utils.core.state.BeamState.phi0_rms>`
          (:py:func:`phirms <flame_utils.core.state.BeamState.phirms>`)

        - :py:func:`dEk0 <flame_utils.core.state.BeamState.dEk0>`
          (:py:func:`dEkcen_all <flame_utils.core.state.BeamState.dEkcen_all>`),
          :py:func:`dEk0_env <flame_utils.core.state.BeamState.dEk0_env>`
          (:py:func:`dEkcen <flame_utils.core.state.BeamState.dEkcen>`),
          :py:func:`dEkrms_all <flame_utils.core.state.BeamState.dEkrms_all>`,
          :py:func:`dEk0_rms <flame_utils.core.state.BeamState.dEk0_rms>`
          (:py:func:`dEkrms <flame_utils.core.state.BeamState.dEkrms>`)


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
        'phicen_all': 'phi0',
        'dEkcen_all': 'dEk0',
        'xrms': 'x0_rms',
        'yrms': 'y0_rms',
        'xprms': 'xp0_rms',
        'yprms': 'yp0_rms',
        'phirms': 'phi0_rms',
        'dEkrms': 'dEk0_rms',
        'xcen': 'x0_env',
        'ycen': 'y0_env',
        'xpcen': 'xp0_env',
        'ypcen': 'yp0_env',
        'phicen': 'phi0_env',
        'dEkcen': 'dEk0_env',
        'cenvector': 'moment0_env',
        'cenvector_all': 'moment0',
        'rmsvector': 'moment0_rms',
        'beammatrix_all': 'moment1',
        'beammatrix': 'moment1_env',
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
        """float: longitudinal reference beam position, [m]"""
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
        """float: Lorentz factor of reference charge state, Lorentz gamma, [1]"""
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
        """float: total energy of reference charge state, [eV/u]

        i.e. :math:`W = E_s + E_k`"""
        return getattr(self._states, 'ref_IonW')

    @ref_IonW.setter
    def ref_IonW(self, x):
        setattr(self._states, 'ref_IonW', x)

    @property
    def ref_IonZ(self):
        """float: reference charge state, measured by charge to mass ratio, [1]

        e.g. :math:`^{33^{+}}_{238}U: Q[33]/A[238]`"""
        return getattr(self._states, 'ref_IonZ')

    @ref_IonZ.setter
    def ref_IonZ(self, x):
        setattr(self._states, 'ref_IonZ', x)

    @property
    def ref_phis(self):
        """float: absolute synchrotron phase of reference charge state, [rad]"""
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
        """Array: Lorentz factor of all charge states, Lorentz gamma, [1]"""
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
        """Array: macro particle number of all charge states, [1]

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
        """Array: all charge states, measured by charge to mass ratio, [1]

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
        """Array: weighted averages of beam centroids for all charge states

        defined as ``[x, x', y, y', phi, dEk, 1]``, with the units of
        ``[mm, rad, mm, rad, rad, MeV/u, 1]``.

        Note
        ----
        The physics meanings for each column are:

        - ``x``: x position in transverse plane
        - ``x'``: x momentum
        - ``y``: y position in transverse plane
        - ``y'``: y momentum
        - ``phi``: synchrotron phase deviation, measured in RF frequency;
        - ``dEk``: kinetic energy deviation w.r.t. reference charge state;
        - ``1``: should be always 1, for the convenience of handling
          dipole kick (i.e. ``orbtrim`` element)
        """
        return getattr(self._states, 'moment0_env')

    @moment0_env.setter
    def moment0_env(self, x):
        setattr(self._states, 'moment0_env', x)

    @property
    def moment0_rms(self):
        """Array: weighted averages of rms beam envelopes, part of statistical results
        from :py:func:`moment1 <flame_utils.core.state.BeamState.moment1>`.

        defined as ``[x, x', y, y', phi, dEk, 1]``, with the units of
        ``[mm, rad, mm, rad, rad, MeV/u, 1]``.

        Note
        ----
        The square of :py:func:`moment0_rms <flame_utils.core.state.BeamState.moment0_rms>`
        should be equal to the diagonal elements of :py:func:`moment1 
        <flame_utils.core.state.BeamState.moment1>`.

        See Also
        --------
        :py:func:`moment1 <flame_utils.core.state.BeamState.moment1>` :
        beam matrixes of all charge states
        """
        return getattr(self._states, 'moment0_rms')

    @property
    def moment0(self):
        """Array: beam centroids of all charge states

        defined as ``[x, x', y, y', phi, dEk, 1]``, with the units of
        ``[mm, rad, mm, rad, rad, MeV/u, 1]``.
        """
        return getattr(self._states, 'moment0')

    @moment0.setter
    def moment0(self, x):
        setattr(self._states, 'moment0', x)

    @property
    def moment1(self):
        r"""Array: beam matrixes of all charge states

        for each charge state, the beam matrix could be written as:

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
        """Array: weighted average of beam matrix for all charge states

        See Also
        --------
        :py:func:`moment1 <flame_utils.core.state.BeamState.moment1>` :
        beam matrixes of all charge states
        """
        return getattr(self._states, 'moment1_env')
        
    @moment1_env.setter
    def moment1_env(self, x):
        setattr(self._states, 'moment1_env', x)

    @property
    def x0(self):
        """Array: x centroids of all charge states, [mm]"""
        return self._states.moment0[0]

    @property
    def xp0(self):
        """Array: x momentums of all charge states, [rad]"""
        return self._states.moment0[1]

    @property
    def y0(self):
        """Array: y centroids of all charge states, [mm]"""
        return self._states.moment0[2]

    @property
    def yp0(self):
        """Array: y momentums of all charge states, [rad]"""
        return self._states.moment0[3]

    @property
    def phi0(self):
        """Array: synchrotron phase deviations (measured in RF) of all
        charge states, [rad]"""
        return self._states.moment0[4]

    @property
    def dEk0(self):
        """Array: kinetic energy deviations w.r.t. reference charge state,
        of all charge states, [MeV/u]"""
        return self._states.moment0[5]

    @property
    def x0_env(self):
        """Array: weight average of x centroids
        (:py:func:`x0 <flame_utils.core.state.BeamState.x0>`), [mm]"""
        return self._states.moment0_env[0]

    @property
    def xp0_env(self):
        """Array: weight average of x momentums
        (:py:func:`xp0 <flame_utils.core.state.BeamState.xp0>`), [rad]"""
        return self._states.moment0_env[1]

    @property
    def y0_env(self):
        """Array: weight average of all y centroids
        (:py:func:`y0 <flame_utils.core.state.BeamState.y0>`), [mm]"""
        return self._states.moment0_env[2]

    @property
    def yp0_env(self):
        """Array: weight average of y momentums
        (:py:func:`yp0 <flame_utils.core.state.BeamState.yp0>`), [rad]"""
        return self._states.moment0_env[3]

    @property
    def phi0_env(self):
        """Array: weight average of synchrotron phase deviations
        (:py:func:`phi0 <flame_utils.core.state.BeamState.phi0>`), [mm]"""
        return self._states.moment0_env[4]

    @property
    def dEk0_env(self):
        """Array: weight average of kinetic energy deviations
        (:py:func:`dEk0 <flame_utils.core.state.BeamState.dEk0>`), [MeV/u]"""
        return self._states.moment0_env[5]

    @property
    def xrms_all(self):
        """Array: rms beam envelopes for x position of all charge states, [mm]"""
        return np.sqrt(self._states.moment1[0, 0, :])

    @property
    def xprms_all(self):
        """Array: rms beam envelopes for x momentum of all charge states, [rad]"""
        return np.sqrt(self._states.moment1[1, 1, :])

    @property
    def yrms_all(self):
        """Array: rms envelopes for y position of all charge states, [mm]"""
        return np.sqrt(self._states.moment1[2, 2, :])

    @property
    def yprms_all(self):
        """Array: rms envelopes for y momentum of all charge states, [rad]"""
        return np.sqrt(self._states.moment1[3, 3, :])

    @property
    def phirms_all(self):
        """Array: rms envelopes for synchrotron phase deviation of all charge states, [rad]"""
        return np.sqrt(self._states.moment1[4, 4, :])

    @property
    def dEkrms_all(self):
        """Array: rms envelopes for kinetic energy deviation of all charge states, [MeV/u]"""
        return np.sqrt(self._states.moment1[5, 5, :])

    @property
    def x0_rms(self):
        """float: weighted average of rms envelopes for x position
        (:py:func:`xrms_all <flame_utils.core.state.BeamState.xrms_all>`), [mm]"""
        return self._states.moment0_rms[0]

    @property
    def xp0_rms(self):
        """float: weighted average of rms envelopes for x momentum
        (:py:func:`xprms_all <flame_utils.core.state.BeamState.xprms_all>`), [rad]"""
        return self._states.moment0_rms[1]

    @property
    def y0_rms(self):
        """float: weighted average of rms envelopes for y position,
        (:py:func:`yrms_all <flame_utils.core.state.BeamState.yrms_all>`), [mm]"""
        return self._states.moment0_rms[2]

    @property
    def yp0_rms(self):
        """float: weighted average of rms envelopes for y momentum
        (:py:func:`yprms_all <flame_utils.core.state.BeamState.yprms_all>`), [rad]"""
        return self._states.moment0_rms[3]

    @property
    def phi0_rms(self):
        """float: weighted average of rms envelopes for synchrotron phase deviation
        (:py:func:`phirms_all <flame_utils.core.state.BeamState.phirms_all>`), [rad]"""
        return self._states.moment0_rms[4]

    @property
    def dEk0_rms(self):
        """float: weighted average of rms envelopes for kinetic energy deviation
        (:py:func:`dEkrms_all <flame_utils.core.state.BeamState.dEkrms_all>`), [MeV/u]"""
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
