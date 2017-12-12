#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unittest for flameutils module.
"""

curdir = os.path.dirname(__file__)

FLAME_DATA = os.path.join(curdir, "lattice/flame_data")


def t_set0(index=None, name=None, type=None):
    index_v = [] if index is None else index
    name_v = [] if name is None else name
    type_v = [] if type is None else type
    s = set()
    for li in [index_v, name_v, type_v]:
        if s == set() or li == []:
            s = s.union(li)
        else:
            s = s.intersection(li)
    return s


def t_set(**kws):
    s = set()
    for k in kws:
        v = kws.get(k, [])
        if s == set() or v == []:
            s = s.union(v)
        else:
            s = s.intersection(v)
    return s


def tt_set():
    index, name, type = [], [], []
    print(t_set(index=index,name=name,type=name))

    index, name, type = [1], [], []
    print(t_set(index=index,name=name,type=name))

    index, name, type = [], [1], []
    print(t_set(index=index,name=name,type=name))

    index, name, type = [], [], [1]
    print(t_set(index=index,name=name,type=name))

    index, name, type = [1,2], [2], [1,3]
    print(t_set(index=index,name=name,type=name))


def t_modelflame():
    #import logging
    #logging.getLogger().setLevel(logging.INFO)

    latfile0 = 'lattice/test_0.lat'
    latfile = make_latfile(latfile0)
    # latfile is None
    fm1 = flameutils.ModelFlame()
    #print(fm1.latfile)
    #print(fm1.machine)
    #print(fm1.mstates)
    fm1.inspect_lattice()

    m = Machine(open(latfile))
    fm1.latfile = latfile
    fm1.machine = m
    fm1.mstates = m.allocState({})
    #fm1.machine, fm1.mstates = fm1.init_machine(latfile)

    print(fm1.latfile)
    #print(fm1.machine)
    print(fm1.mstates)
    print('-'*70)
    
    # latfile is not None
    fm2 = flameutils.ModelFlame(lat_file=latfile)
    #print(fm2.latfile)
    #print(fm2.machine)
    #print(fm2.mstates)


    # inspection
    fm2.inspect_lattice()
    print(fm2.get_element(type='stripper'))
    #sio = StringIO()
    #flameutils.inspect_lattice(latfile, out=sio)
    #print(sio.getvalue())
    

    # get index by name
    names = [
             'LS1_CA01:SOL1_D1131_1', 
             'LS1_CA01:CAV4_D1150',
             'LS1_WB01:BPM_D1286',
             'LS1_CA01:GV_D1124',
             'LS1_CA01:BPM_D1144'
             ]
    print flameutils.get_index_by_name(name=names, latfile=latfile)
    print flameutils.get_index_by_name(name=names, _machine=m)
    print flameutils.get_index_by_name(name=names, _machine=m, rtype='list')
    print('-'*70)
    print(flameutils.get_element(name=names, latfile=latfile))
    print('-'*70)

    #names = ''
    #print flameutils.get_index_by_name(name=names, latfile=latfile)


    types = ['stripper', 'source']
    print flameutils.get_index_by_type(type=types, latfile=latfile)
    print flameutils.get_index_by_type(type=types, _machine=m)


    insres1 = flameutils.get_element(name=names, type=['source','bpm'], latfile=latfile)
    print(insres1)

    insres2 = flameutils.get_element(type='stripper', latfile=latfile)
    print(insres2)
    

    print(flameutils.get_all_types(latfile=latfile))
    

def t_machinestates():
    latfile0 = os.path.join(curdir, 'lattice/test_0.lat')
    latfile = make_latfile(latfile0)
    m = Machine(open(latfile, 'r'))
    s0 = m.allocState({})
    s1 = s0.clone()
    m.propagate(s1, 0, 1)
    
    ms0 = flameutils.MachineStates(s0)
    iter_all_attrs(ms0, s0)
    #a, b = ms0.moment1, s0.moment1
    #print ((a == b) | (np.isnan(a) & np.isnan(b))).all()
    
    #np.testing.assert_allclose(a,b)

    ms1 = flameutils.MachineStates(s0, machine=m)
    iter_all_attrs(ms1, s1)
    print ms1.moment0_env
    print s1.moment0_env

    ms2 = flameutils.MachineStates(s0, latfile=latfile)
    iter_all_attrs(ms2, s1)
    

def iter_all_attrs(ms, s):
    k = ['beta', 'bg', 'gamma', 'IonEk', 'IonEs', 'IonQ', 'IonW', 
         'IonZ', 'phis', 'SampleIonK']
    all_keys = [i for i in k]
    all_keys.extend(['ref_{}'.format(i) for i in k] + ['pos'])
    all_keys.extend(['moment0', 'moment0_env', 'moment0_rms', 'moment1'])
    for attr in all_keys:
        left_val, right_val = getattr(ms, attr), getattr(s, attr)
        if isinstance(left_val, np.ndarray):
            #print(attr, (np.allclose(left_val, right_val, equal_nan=True) | (np.isnan(left_val) & np.isnan(right_val))).all())
            print(attr, ((left_val == right_val) | (np.isnan(left_val) & np.isnan(right_val))).all())
        else:
            print(attr, left_val==right_val)


if __name__ == '__main__':
    #t_modelflame()
    t_machinestates()
    #tt_set()
