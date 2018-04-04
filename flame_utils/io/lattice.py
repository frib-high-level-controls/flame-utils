#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FLAME lattice operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import flame
import numpy as np

import logging

try:
    basestring
except NameError:
    basestring = str

from flame_utils.core import get_all_names
from flame_utils.core import generate_source


_LOGGER = logging.getLogger(__name__)

def generate_latfile(machine, latfile=None, state=None, original=None,
                     out=None, start=None, end=None):
    """Generate lattice file for the usage of FLAME code.

    Parameters
    ----------
    machine :
        FLAME machine object.
    latfile :
        File name for generated lattice file.
    original :
        File name for original lattice file to keep user comments and indents. (optional)
    state :
        BeamState object, accept FLAME internal State object also. (optional)
    out :
        New stream paramter, file stream. (optional)
    start :
        Start element (id or name) of generated lattice. (optional)
    end :
        End element (id or name) of generated lattice. (optional)

    Returns
    -------
    filename : str
        None if failed to generate lattice file, or the out file name.

    Note
    ----
    - If *latfile* and *out* are not defined, will print all output to screen;
    - If *latfile* and *out* are all defined, *out* stream is preferred;
    - For other cases, choose one that is defined.
    - If *start* is defined, user should define *state* also.
    - If user define *start* only, the initial beam state is the same as the *machine*.

    Examples
    --------
    >>> from flame import Machine
    >>> latfile = 'test.lat'
    >>> m = Machine(open(latfile))
    >>> outfile1 = generate_latfile(m, latfile='out1.lat')
    >>> m.reconfigure(80, {'theta_x': 0.1})
    >>> outfile2 = generate_latfile(m, latfile='out2.lat')
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
    except:
        _LOGGER.error("Failed to load FLAME machine object.")
        return None

    try:
        mconf_ks = list(mconf.keys())
        [mconf_ks.remove(i) for i in ['elements', 'name'] if i in mconf_ks]
        mc_src = m.conf(m.find(type='source')[0])

        # initial beam condition from input
        if state is not None:
            mc_src = generate_source(state, sconf={'index':0, 'properties':mc_src})['properties']

    except:
        _LOGGER.error("Failed to load initial beam state.")
        return None

    if not isinstance(original, basestring):

        try:
            lines = []

            for k in mconf_ks:
                v = mc_src[k]
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                if isinstance(v, str):
                    v = '"{0}"'.format(v)
                line = '{k} = {v};'.format(k=k, v=v)
                lines.append(line)

            mconfe = mconf['elements']
            elem_num = len(mconfe)

            if start is None:
                start = 1
            elif isinstance(start, basestring):
                start = m.find(name=start)[0]
            else :
                start = int(start)

            if start != 1 and state is None:
                _LOGGER.warning("Initial beam state is missing. Use original initial beam state.")

            if end is None:
                end = elem_num
            elif isinstance(end, basestring):
                end = m.find(name=end)[0] + 1
            else :
                end = int(end) + 1

            section = [0] + list(range(start,end))

            # element configuration
            elem_name_list = []
            for i in section:
                elem_i = m.conf(i)
                ename, etype = elem_i['name'], elem_i['type']
                if ename in elem_name_list:
                    continue
                elem_name_list.append(ename)
                ki = elem_i.keys()
                elem_k = set(ki).difference(mc_src.keys())
                if etype == 'source':
                    elem_k.add('vector_variable')
                    elem_k.add('matrix_variable')
                if etype == 'stripper':
                    elem_k.add('IonChargeStates')
                    elem_k.add('NCharge')
                p = []
                for k, v in sorted(elem_i.items()):
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

            dline = '(' + ', '.join(([m.conf(i)['name'] for i in section])) + ')'

            blname = mconf.get('name', 'default')
            lines.append('{0}: LINE = {1};'.format(blname, dline))
            lines.append('USE: {0};'.format(blname))

        except:
            _LOGGER.error("Failed to generate lattice file.")
            return None

    else:

        try:
            names = get_all_names(_machine=m)

            with open(original, 'r') as f:
                fline = f.readlines()

            def gps(l):
                ret = {}
                for k in ['#', ': ', ';', '=', ',']:
                    ret[k] = l.find(k)
                if ret['#'] == -1:
                    ret['#'] = len(l)
                return ret

            lines = []

            n = 0
            while n < len(fline):
                l = fline[n]
                rl = l.replace(' ', '')
                l = l[0:-1]
                if rl == '\n' or rl[0] == '#':
                    lines.append(l)
                    n += 1
                else:
                    p = gps(l)
                    nl = None
                    if (p['='] != -1 and p['='] < p['#']) and \
                        (p[': '] == -1 or (p['='] < p[': '] < p['#'])):
                        bp = l[0:p['=']].replace(' ', '')
                        if bp in mc_src:
                            if bp == 'Eng_Data_Dir':
                                nl = l[0:-1]
                            elif mc_src['vector_variable'] == bp[0:-1]:
                                nl = bp + ' = ' + str(mc_src[bp].tolist())
                            elif mc_src['matrix_variable'] == bp[0:-1]:
                                nl = bp + ' = [\n'
                                kk = 0
                                for i in range(7):
                                    nl += '    '
                                    for j in range(7):
                                        nl += str(mc_src[bp][kk]) + ', '
                                        kk += 1
                                    nl += '\n'
                                nl = nl[0:-3]
                                nl += ']'
                            else:
                                v = mc_src[bp]
                                if isinstance(v, np.ndarray):
                                    v = str(v.tolist())
                                elif isinstance(v, basestring):
                                    v = '"' + str(v) + '"'
                                else:
                                    v = str(v)
                                nl = bp + ' = ' + v

                    elif p[': '] != -1 and p[': '] < p['#']:
                        name = l[0:p[': ']].replace(' ', '')
                        if name in names:
                            c = m.conf(m.find(name=name)[0])
                            nl = c['name'] + ': ' + c['type']
                            keys = set(c.keys()).difference(mc_src.keys())
                            if c['type'] == 'source':
                                keys.add('vector_variable')
                                keys.add('matrix_variable')
                            if c['type'] == 'stripper':
                                keys.add('IonChargeStates')
                                keys.add('NCharge')

                            for k in sorted(keys):
                                v = c[k]
                                if isinstance(v, np.ndarray):
                                    v = str(v.tolist())
                                elif isinstance(v, basestring):
                                    v = '"' + str(v) + '"'
                                else:
                                    v = str(v)
                                nl += ', ' + k + ' = ' + v

                    if nl is None:
                        lines.append(l)
                    else:
                        lines.append(nl + '; ' + l[p['#']:])
                        if p[';'] == -1:
                            flg = 1
                            while flg:
                                n += 1
                                if n < len(fline):
                                    tl = fline[n]
                                    tp = gps(tl)
                                    if tp[';'] != -1 and  tp[';'] < tp['#'] : flg = 0
                                else:
                                    flg = 0
                    n += 1
        except:
            _LOGGER.error("Failed to generate lattice file with original file.")
            return None

    all_lines = '\n'.join(lines)
    try:
        if latfile is None and out is None:
            sout = sys.stdout
            print(all_lines, file=sout)
            retval = sout.name
        elif out is None:
            with open(latfile, 'wb') as sout:
                sout.write(all_lines.encode())
                retval = sout.name
        else:
            sout = out
            print(all_lines, file=sout)
            retval = 'string'
    except:
        _LOGGER.error("Failed to write to %s" % latfile)
        return None

    return retval
