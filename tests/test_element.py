# -*- coding: utf-8 -*-

import unittest
import os
from cStringIO import StringIO
import random
from numpy import array
import numpy as np

from flame import Machine

from flame_utils import inspect_lattice
from flame_utils import get_element
from flame_utils import get_index_by_type
from flame_utils import get_index_by_name
from flame_utils import get_all_types
from flame_utils import get_all_names
from _utils import make_latfile

curdir = os.path.dirname(__file__)


class TestInspectLattice(unittest.TestCase):
    def setUp(self):
        latfile = os.path.join(curdir, 'lattice/test_0.lat')
        self.latfile = make_latfile(latfile)
        self.inslat = os.path.join(curdir, 'data/inslat.out')

    def test_wrong_file(self):
        retval = inspect_lattice('')
        self.assertEqual(retval, None)

    def test_right_file(self):
        sio = StringIO()
        inspect_lattice(self.latfile, out=sio)
        outstr = open(self.inslat).read()
        line1 = [line.strip() for line in sio.getvalue().split('\n')]
        line0 = [line.strip() for line in outstr.split('\n')]
        self.assertEqual(line1, line0)


class TestGetElement(unittest.TestCase):
    def setUp(self):
        latfile = os.path.join(curdir, 'lattice/test_0.lat')
        self.latfile = make_latfile(latfile)
        self.m = Machine(open(self.latfile, 'r'))

    def test_one_name(self):
        ename = 'LS1_CA01:CAV4_D1150'
        e0 = [{'index': 27, 'properties': {'L': 0.24, 'aper': 0.017,
              'cavtype': '0.041QWR', 'f': 80500000.0,
              'name': 'LS1_CA01:CAV4_D1150', 'phi': 325.2,
              'scl_fac': 0.819578, 'type': 'rfcavity'}}]
        e1 = get_element(name=ename, latfile=self.latfile)
        e2 = get_element(name=ename, _machine=self.m)
        e3 = get_element(name='', _machine=self.m)
        self.assertEqual(e1, e2)
        self.assertEqual(e1, e0)
        self.assertEqual(e3, [])

    def test_multi_names(self):
        enames = [
                    'LS1_CA01:SOL1_D1131_1',
                    'LS1_CA01:CAV4_D1150',
                    'LS1_WB01:BPM_D1286',
                    'LS1_CA01:GV_D1124',
                    'LS1_CA01:BPM_D1144',
                ]
        e0 = [
                {'index': 8,
                 'properties': {'aper': 0.02, 'B': 5.34529,
                                'type': 'solenoid', 'L': 0.1,
                                'name': 'LS1_CA01:SOL1_D1131_1'}},
                {'index': 1,
                 'properties': {'aper': 0.02, 'type': 'drift',
                                'L': 0.072, 'name': 'LS1_CA01:GV_D1124'}},
                {'index': 154,
                 'properties': {'type': 'bpm',
                                'name': 'LS1_WB01:BPM_D1286'}},
                {'index': 27,
                 'properties': {'aper': 0.017,
                                'name': 'LS1_CA01:CAV4_D1150',
                                'f': 80500000.0, 'cavtype': '0.041QWR',
                                'L': 0.24, 'phi': 325.2,
                                'scl_fac': 0.819578,
                                'type': 'rfcavity'}},
                {'index': 18,
                 'properties': {'type': 'bpm',
                                'name': 'LS1_CA01:BPM_D1144'}}
              ]

        e1 = get_element(name=enames, latfile=self.latfile)
        self.assertEqual(e1, e0)

    def test_one_index(self):
        idx = 10
        e0 = [{'index': 10,
               'properties': {'name': 'LS1_CA01:DCV_D1131',
               'theta_y': 0.0, 'type': 'orbtrim'}}]
        e1 = get_element(index=idx, latfile=self.latfile)
        self.assertEqual(e1, e0)

    def test_get_source(self):
        source_conf = {'index': 0,
                       'properties': {
                           'IonChargeStates': array([0.138655]),
                           'IonEk': 500000.0,
                           'IonEs': 931494000.0,
                           'NCharge': array([ 10111.]),
                           'P0': array([-7.88600000e-04, 1.08371000e-05, 1.33734000e-02,
                                        6.67853000e-06, -1.84773000e-04, 3.09995000e-04,
                                        1.00000000e+00]),
                           'S0': array([
                                 2.76309000e+00,  -4.28247000e-04,   1.58179000e-02,
                                 2.15594000e-05,   1.86381000e-04,  -2.99394000e-05,
                                 0.00000000e+00,  -4.28247000e-04,   3.84947000e-06,
                                -1.38385000e-06,  -1.85410000e-08,   1.06778000e-07,
                                 5.28564000e-09,   0.00000000e+00,   1.58179000e-02,
                                -1.38385000e-06,   2.36251000e+00,  -6.69320000e-04,
                                -5.80100000e-04,   6.71652000e-06,   0.00000000e+00,
                                 2.15594000e-05,  -1.85410000e-08,  -6.69320000e-04,
                                 4.89711000e-06,  -5.01615000e-07,   5.57484000e-08,
                                 0.00000000e+00,   1.86381000e-04,   1.06778000e-07,
                                -5.80100000e-04,  -5.01615000e-07,   6.71687000e-04,
                                -1.23223000e-05,   0.00000000e+00,  -2.99394000e-05,
                                 5.28564000e-09,   6.71652000e-06,   5.57484000e-08,
                                -1.23223000e-05,   1.99525000e-06,   0.00000000e+00,
                                 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                                 0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                                 0.00000000e+00]),
                           'matrix_variable': 'S',
                           'name': 'S',
                           'type': 'source',
                           'vector_variable': 'P'}}
        sconf = get_element(index=0, latfile=self.latfile)[0]
        for k,v in sconf['properties'].items():
            left_val, right_val = v, source_conf['properties'][k]
            if isinstance(v, np.ndarray):
                self.assertTrue(((left_val == right_val) | (np.isnan(left_val) & np.isnan(right_val))).all())
            else:
                self.assertAlmostEqual(left_val, right_val)

    def test_multi_indice(self):
        idx = range(1,3)
        e0 = [
                {'index': 1,
                 'properties': {'L': 0.072,
                 'aper': 0.02,
                 'name': 'LS1_CA01:GV_D1124',
                 'type': 'drift'}},
                {'index': 2,
                 'properties': {'L': 0.135064,
                 'aper': 0.02,
                 'name': 'DRIFT_1',
                 'type': 'drift'}}
            ]
        e1 = get_element(index=idx, latfile=self.latfile)
        self.assertEqual(e1, e0)

    def test_multi_filters(self):
        eidx, etyp = range(20), 'bpm'
        e0 = [
              {'index': 18,
               'properties': {'name': 'LS1_CA01:BPM_D1144', 'type': 'bpm'}},
              {'index': 5,
               'properties': {'name': 'LS1_CA01:BPM_D1129', 'type': 'bpm'}}
              ]
        e1 = get_element(index=eidx, type=etyp, latfile=self.latfile)
        self.assertEqual(e1, e0)
        
        e2 = get_element(index=eidx, type=etyp, latfile=self.latfile, name='LS1_CA01:BPM_D1144')
        self.assertEqual(e2, [e0[0]])

    def test_name_pattern(self):
        name_pattern = 'FS1_BBS:DH_D2394_.?$'
        e1 = get_element(_pattern=name_pattern, latfile=self.latfile)
        names = ['FS1_BBS:DH_D2394_{}'.format(i) for i in range(1,10)]
        e2 = get_element(name=names, latfile=self.latfile)
        self.assertEqual(e1, e2)


class TestGetIndexByType(unittest.TestCase):
    def setUp(self):
        latfile = os.path.join(curdir, 'lattice/test_0.lat')
        self.latfile = make_latfile(latfile)
        self.m = Machine(open(self.latfile, 'r'))

    def test_wrong_type(self):
        etyp = ''
        e = get_index_by_type(type=etyp, latfile=self.latfile)
        self.assertEqual(e, {etyp: []})

        etyp1 = 'no_exist_type'
        e1 = get_index_by_type(type=etyp1, latfile=self.latfile)
        self.assertEqual(e1, {etyp1: []})
    
    def test_one_type(self):
        for etyp in get_all_types(latfile=self.latfile):
            e = get_index_by_type(type=etyp, latfile=self.latfile)
            self.assertEqual(e, {etyp: self.m.find(type=etyp)})

    def test_multi_types(self):
        all_types = get_all_types(latfile=self.latfile)
        for n in range(2, len(all_types)):
            etyps = [random.choice(all_types) for _ in range(n)]
            e = get_index_by_type(type=etyps, 
                                             latfile=self.latfile)
            e0 = {t: self.m.find(type=t) for t in etyps}
            self.assertEqual(e, e0)


class TestGetIndexByName(unittest.TestCase):
    def setUp(self):
        latfile = os.path.join(curdir, 'lattice/test_0.lat')
        self.latfile = make_latfile(latfile)
        self.m = Machine(open(self.latfile, 'r'))

    def test_wrong_name(self):
        ename = ''
        e = get_index_by_name(name=ename, latfile=self.latfile)
        self.assertEqual(e, {ename: []})
        
        ename1 = 'no_exist_name'
        e = get_index_by_name(name=ename1, latfile=self.latfile)
        self.assertEqual(e, {ename1: []})

    def test_one_name(self):
        """ test_one_name: repeat for 10 times, each with random name
        """
        all_names = get_all_names(latfile=self.latfile)
        for n in range(10):
            ename = random.choice(all_names)
            e = get_index_by_name(name=ename, latfile=self.latfile)
            self.assertEqual(e, {ename: self.m.find(name=ename)})

    def test_multi_names(self):
        """ test_multi_names: test names list length of 2~10
        """
        all_names = get_all_names(latfile=self.latfile)
        for n in range(2, 10):
            enames = [random.choice(all_names) for _ in range(n)]
            e = get_index_by_name(name=enames, latfile=self.latfile)
            e0 = {n: self.m.find(name=n) for n in enames}
            self.assertEqual(e, e0)


