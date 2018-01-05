#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module provides lattice picture from FLAME lattice file(.lat) or FLAME Machine object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from flame import Machine

import matplotlib.lines as lin
import matplotlib.patches as ptc
import matplotlib.pyplot as plt
import numpy as np


class PlotLat:
    """Lattice picture class from FLAME lattice file or FLAME Machine object

    Parameters
    ----------
    source : str or callable
        File path of the lattic file (str) or FLAME Machine object (callable)

    output : str (None), optional
        Output file name. If defined, the lattice plot is generated automatically.

    auto_scaling : bool (True), optional
        Flag for y-axis scaling by strength of the optical elements

    starting_offset : float (0.0), optional
        Position offset of starting point in the lattice file


    Class attributes
    ----------------
    types : dict
        Element type list of the lattice. Each element type contains
        on-off 'flag', plotting 'color', and y-axis 'scale'.

    """
    def __init__(self, source, output=None, auto_scaling=True , starting_offset=0.0, **kws):
        self._source = source
        self._auto_scaling = auto_scaling
        self._starting_offset = starting_offset

        if type(self._source) == str:
            with open(self._source, 'rb') as lat:
                self.M = Machine(lat)

        elif type(self._source) == Machine:
            self.M = self._source

        else:
            raise ValueError('source must be a file path of .lat or flame.Machine object')

        self.types = {'rfcavity':   {'flag':True, 'name':'rfcavity', 'color':'orange', 'scale':0.0},
                      'solenoid':   {'flag':True, 'name':'solenoid', 'color':'m',      'scale':0.0},
                      'quadrupole': {'flag':True, 'name':'quad',     'color':'blue',   'scale':0.0},
                      'sextupole':  {'flag':True, 'name':'sext',     'color':'purple', 'scale':0.0},
                      'sbend':      {'flag':True, 'name':'bend',     'color':'green',  'scale':0.0},
                      'equad':      {'flag':True, 'name':'e-quad',   'color':'navy',   'scale':0.0},
                      'edipole':    {'flag':True, 'name':'e-dipole', 'color':'lime',   'scale':0.0},
                      'bpm':        {'flag':True, 'name':'bpm',      'color':'red',    'scale':0.0},
                      'orbtrim':    {'flag':True, 'name':'corr',     'color':'black',  'scale':0.0},
                      'stripper':   {'flag':True, 'name':'stripper', 'color':'y',      'scale':0.0},
                      'marker':     {'flag':True, 'name':'pm',       'color':'c',      'scale':0.0}
                      }

        if self._auto_scaling:
            for i in range(len(self.M)):
                elem = self.M.conf(i)
                if elem['type'] in self.types.keys():
                    prv_scl = self.types[elem['type']]['scale']
                    tmp_scl = np.abs(self._get_scl(elem))

                    self.types[elem['type']]['scale'] = max(prv_scl,tmp_scl)

        if isinstance(output, (str, unicode)):
            self.generate()
            self.output(window=True, fname=output, **kws)


    def _get_scl(self, elem):
        """Get arbital strength of the optical element
        """

        scl = 0.0

        if elem['type'] == 'rfcavity':
            scl = elem['scl_fac']*np.cos(2.0*np.pi*elem['phi']/360.0)
        elif elem['type'] == 'solenoid':
            scl = elem['B']
        elif elem['type'] == 'quadrupole':
            scl = elem['B2']
        elif elem['type'] == 'sextuple':
            scl = elem['B3']
        elif elem['type'] == 'sbend':
            scl = elem['phi']
        elif elem['type'] == 'equad':
            scl = elem['V']/elem['radius']**2.0
        elif elem['type'] == 'edipole':
            scl = elem['phi']

        return scl


    def generate(self, start=None, end=None, xlim=None, aspect=5.0):
        """Generate matplotlib Axes class object from lattice file

        Parameter
        ---------
        start : int
            Index of the lattice start.

        end : int
            Index of the lattice end.

        xlim : list[2], optinal
            Plot range of the lattice.

        aspect : float (5.0), optional
            Aspect ratio of the picture.


        Class attribute
        ---------------
        axes : callable
            Axes class object of matplotlib.

        total_length : float
            Total length of the lattice.

        """

        self._fig = plt.figure()
        self.axes = self._fig.add_subplot(111)

        pos = self._starting_offset

        self.axes.add_line(lin.Line2D([-1,1e5], [0,0], color='gray'))

        indexes = range(len(self.M))[start:end]
        foundelm = []

        for i in indexes:
            elem = self.M.conf(i)

            try:
                dL = elem['L']
            except:
                dL = 0.0

            if elem['type'] in self.types.keys():
                info = self.types[elem['type']]

                if foundelm.count(elem['type']) == 0:
                    foundelm.append(elem['type'])
                    self.axes.fill_between([0,0],[0,0],[0,0], color=info['color'], label=info['name'])

                if info['flag']:
                    if dL != 0.0:

                        bp = 0.0

                        if info['scale'] != 0.0:
                            ht = self._get_scl(elem)/info['scale'] + 0.05
                        else:
                            ht = 1.0

                        if elem['type'] == 'rfcavity' or elem['type'] == 'solenoid':
                            bp = bp-ht
                            ht *= 2.0

                        self.axes.add_patch(ptc.Rectangle((pos, bp),dL,ht,
                                                           edgecolor='none',facecolor=info['color']))
                    else:
                        self.axes.add_line(lin.Line2D([pos,pos],[-0.1,0.1],color=info['color']))

            pos += dL

        self.total_length = pos

        if xlim is not None:
            self.axes.set_xlim(xlim)
            ancscl = xlim[1]-xlim[0]
        else :
            self.axes.set_xlim((0.0,pos))
            ancscl = pos

        self.axes.set_aspect(aspect)
        self.axes.set_ylim((-1.0,1.0))
        self.axes.set_yticks(())

        if len(foundelm) <= 4 :
            ncol = 4
        elif len(foundelm) <= 6:
            ncol = 3
        elif len(foundelm) <= 8 :
            ncol = 4
        elif len(foundelm) <= 9:
            ncol = 3
        else :
            ncol = 4

        self.axes.legend(ncol=ncol, loc=10, bbox_to_anchor=(0.5, -0.01*ancscl))

        self.axes.grid()
        self.axes.spines['right'].set_visible(False)
        self.axes.spines['left'].set_visible(False)
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['bottom'].set_visible(False)
        plt.tight_layout()

    def output(self,window=True,fname=None,**kws):
        """Output the lattice picture to window and/or file.

        Parameters
        ----------
        window : bool (True)
            Output flag to the window.

        fname : str, optional
            File path of the output picutre.

        **kwargs :
            The same kwargs as pyplot.savefig() are available.
        """

        if type(fname) == str :
            plt.savefig(fname, **kws)

        if window: plt.show()


if __name__ == "__main__":
    import sys

    try:
        lat = sys.argv[1]
    except:
        raise ValueError('First argument must be a lattice file.')

    try:
        out = sys.argv[2]
    except:
        raise ValueError('Second argument must be a output file name.')

    pl = PlotLat(lat, output=out)
