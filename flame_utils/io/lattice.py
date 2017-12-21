#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FLAME lattice.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


def generate_latfile(machine, latfile=None, state=None, original=None, out=None):
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
        FLAME beam state object of initial condition. (optional)
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
        print("Failed to load FLAME machine object.")
        return None

    try:
        mconf_ks = mconf.keys()
        [mconf_ks.remove(i) for i in ['elements', 'name'] if i in mconf_ks]
        mc_src = m.conf(m.find(type = 'source')[0])

        # initial beam condition from input
        if type(state) == type(m.allocState({})):
            mc_src['IonEk'] = state.ref_IonEk
            mc_src['IonEs'] = state.ref_IonEs

            mc_src['IonChargeStates'] = state.IonZ
            mc_src['NCharge'] = state.IonQ

            cenkey = mc_src['vector_variable']
            envkey = mc_src['matrix_variable']
            for i in range(len(state.IonZ)):
                mc_src[cenkey+str(i)] = state.moment0[:,i]
                mc_src[envkey+str(i)] = state.moment1[:,:,i].flatten()

    except:
        print("Failed to load initial beam state.")
        return None

    if not isinstance(original, (str, unicode)):

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
                elem_k = set(ki).difference(mc_src.keys())
                if etype == 'source':
                    elem_k.add('vector_variable')
                    elem_k.add('matrix_variable')
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

    else:

        try:
            names = [m.conf(i)['name'] for i in range(len(m))]

            with open(original, 'rb') as f:
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
                        (p[': '] == -1 or (p[': '] > p['='] and p[': '] < p['#'])):
                        bp = l[0:p['=']].replace(' ', '')
                        if mc_src.has_key(bp):
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
                                elif isinstance(v, (str, unicode)):
                                    v = '"' + str(v) + '"'
                                else:
                                    v = str(v)
                                nl = bp + ' = ' + v

                    elif (p[': '] != -1 and p[': '] < p['#']):
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

                            for k in keys:
                                v = c[k]
                                if isinstance(v, np.ndarray):
                                    v = str(v.tolist())
                                elif isinstance(v, (str, unicode)):
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
            print("Failed to generate lattice file with original file.")
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
