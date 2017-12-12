#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Abstracted FLAME machine state class.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging


from flame_utils.misc import machine_setter

__authors__ = "Tong Zhang"
__copyright__ = "(c) 2016-2017, Facility for Rare Isotope beams, " \
                "Michigan State University"
__contact__ = "Tong Zhang <zhangt@frib.msu.edu>"


_LOGGER = logging.getLogger(__name__)


class MachineStates(object):
    """Class for general FLAME machine states

    All attributes of states:

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
                _m = machine_setter(_latfile, _machine, 'MachineStates')
                if _m is not None:
                    self._states = _m.allocState({})
        else:
            self._states = s

        if self._states is not None:
            if _is_zeros_states(self._states):
                _m = machine_setter(_latfile, _machine, 'MachineStates')
                if _m is not None:
                    _m.propagate(self._states, 0, 1)
                else:
                    _LOGGER.warning(
                        "MachineStates: \
                        The initial machine states is 0, true values could be obtained with more information.")

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


def _is_zeros_states(s):
    """ test if flame machine states is all zeros

    Returns
    -------
    True or False
        True if is all zeros, else False
    """
    return np.alltrue(getattr(s, 'moment0') == np.zeros([7, 1]))

