# -*- coding: utf-8 -*-

import unittest
import os
import numpy as np
import random

from flame import Machine
from flame_utils import MachineStates
from flame_utils import ModelFlame
from flame_utils import collect_data

from _utils import make_latfile

curdir = os.path.dirname(__file__)


class TestMachineStates(unittest.TestCase):
    def setUp(self):
        latfile = os.path.join(curdir, 'lattice/test_0.lat')
        self.latfile = make_latfile(latfile)

    def test_init_with_s1(self):
        """ test_init_with_s1: s it not None
        """
        m = Machine(open(self.latfile, 'r'))
        s0 = m.allocState({})
        s1 = s0.clone()
        m.propagate(s1, 0, 1)

        ms0 = MachineStates(s0)
        self.iter_all_attrs(ms0, s0)

        ms1 = MachineStates(s0, machine=m)
        self.iter_all_attrs(ms1, s1)

        ms1_1 = MachineStates(s0, latfile=self.latfile)
        self.iter_all_attrs(ms1_1, s1)

    def test_init_with_s2(self):
        """ test_init_with_s2: s it None
        """
        m = Machine(open(self.latfile, 'r'))
        s = m.allocState({})
        ms = MachineStates()
        ms.mstates = s
        m.propagate(s, 0, 1)
        self.iter_all_attrs(ms, s)
    
    def test_init_with_machine(self):
        m = Machine(open(self.latfile, 'r'))
        ms = MachineStates(machine=m)
        s = m.allocState({})
        m.propagate(s, 0, 1)
        self.iter_all_attrs(ms, s)
 
    def test_init_with_latfile(self):
        m = Machine(open(self.latfile, 'r'))
        ms = MachineStates(latfile=self.latfile)
        s = m.allocState({})
        m.propagate(s, 0, 1)
        self.iter_all_attrs(ms, s)

    def test_init_with_mix(self):
        m = Machine(open(self.latfile, 'r'))
        ms = MachineStates(machine=m, latfile=self.latfile)
        s = m.allocState({})
        m.propagate(s, 0, 1)
        self.iter_all_attrs(ms, s)

    def iter_all_attrs(self, ms, s):
        k = ['beta', 'bg', 'gamma', 'IonEk', 'IonEs', 'IonQ', 'IonW', 
             'IonZ', 'phis', 'SampleIonK']
        all_keys = [i for i in k]
        all_keys.extend(['ref_{}'.format(i) for i in k] + ['pos'])
        all_keys.extend(['moment0', 'moment0_env', 'moment0_rms', 'moment1'])
        for attr in all_keys:
            left_val, right_val = getattr(ms, attr), getattr(s, attr)
            if isinstance(left_val, np.ndarray):
                #self.assertTrue((np.allclose(left_val, right_val, equal_nan=True) | (np.isnan(left_val) & np.isnan(right_val))).all())
                self.assertTrue(((left_val == right_val) | (np.isnan(left_val) & np.isnan(right_val))).all())
            else:
                self.assertAlmostEqual(left_val, right_val)


class TestModelFlame(unittest.TestCase):
    def setUp(self):
        testfile = os.path.join(curdir, 'lattice/test_0.lat')
        self.testfile = make_latfile(testfile)
        self.fm = ModelFlame(self.testfile)

    def test_set_latfile(self):
        fm_none = ModelFlame()
        self.assertIsNone(fm_none.latfile)
        fm_none.latfile = self.testfile
        self.assertEqual(fm_none.latfile, self.testfile)

    def test_set_machine(self):
        fm_none = ModelFlame()
        self.assertIsNone(fm_none.machine)
        m = Machine(open(self.testfile, 'r'))
        fm_none.machine = m
        self.assertEqual(fm_none.machine, m)

    def test_set_mstates(self):
        fm_none = ModelFlame()
        self.assertIsNone(fm_none.mstates)
        m = Machine(open(self.testfile, 'r'))
        s = m.allocState({})
        m.propagate(s, 0, 1)
        fm_none.mstates = s
        self.assertEqual(fm_none.mstates, s)

    def test_init_machine(self):
        fm_none = ModelFlame()
        m, s = fm_none.init_machine(self.testfile)
        fm_none.machine, fm_none.mstates = m, s
        self.assertEqual(fm_none.machine, m)
        self.assertEqual(fm_none.mstates, s)

    def test_get_all_types(self):
        fm = ModelFlame(self.testfile)
        etypes = ['quadrupole', 'bpm', 'drift', 'source', 'rfcavity',
                  'sbend', 'orbtrim', 'solenoid', 'stripper']
        self.assertEqual(fm.get_all_types(), etypes)
    
    def test_get_index_by_name(self):
        fm = ModelFlame(self.testfile)
        m = fm.machine
        all_names = fm.get_all_names()
        for n in range(2, 20):
            enames = [random.choice(all_names) for _ in range(n)]
            e = fm.get_index_by_name(name=enames)
            e0 = {n: m.find(name=n) for n in enames}
            self.assertEqual(e, e0)

    def test_get_index_by_type(self):
        fm = ModelFlame(self.testfile)
        m = fm.machine
        all_types = fm.get_all_types()
        for n in range(2, len(all_types)):
            etyps = [random.choice(all_types) for _ in range(n)]
            e = fm.get_index_by_type(type=etyps)
            e0 = {t: m.find(type=t) for t in etyps}
            self.assertEqual(e, e0)

    def test_run_1(self):
        """ test_run_1: propagate from the first to last, monitor None
        """ 
        latfile = self.testfile
        m0 = Machine(open(latfile, 'r'))
        s0 = m0.allocState({})
        m0.propagate(s0, 0, -1)
        fm = ModelFlame(latfile)
        r,s = fm.run()
        self.assertEqual(r, [])
        self.iter_all_attrs(s, s0)

    def test_run_2(self):
        """ test_run_2: propagate from the first to last, monitor all BPMs
        """
        latfile = self.testfile
        m0 = Machine(open(latfile, 'r'))
        s0 = m0.allocState({})
        fm = ModelFlame(latfile)
        obs = fm.get_index_by_type(type='bpm')['bpm']
        r0 = m0.propagate(s0, 0, -1, observe=obs)
        r,s = fm.run(monitor=obs)
        rs0 = [ts for (ti,ts) in r0] 
        rs = [ts for (ti,ts) in r] 
        for (is1, is2) in zip(rs0, rs):
            self.iter_all_attrs(is1, is2)
        self.iter_all_attrs(s, s0)

    def test_run_3(self):
        """ test run_3: test initial states
        """
        latfile = self.testfile
        m0 = Machine(open(latfile, 'r'))
        s0 = m0.allocState({})
        m0.propagate(s0, 0, 1)
        fm = ModelFlame(latfile)
        r, s = fm.run(from_element=0, to_element=0)
        self.iter_all_attrs(s0, s)

    def test_run_4(self):
        """ test_run_4: run and monitor from element index of 10 to 20
        """
        latfile = self.testfile
        m0 = Machine(open(latfile, 'r'))
        s0 = m0.allocState({})
        m0.propagate(s0, 0, 1)
        r0 = m0.propagate(s0, 10, 11, observe=range(10, 21))

        fm = ModelFlame(latfile)
        r, s = fm.run(from_element=10, to_element=20, monitor=range(10,21))
        self.iter_all_attrs(s0, s)
        rs0 = [ts for (ti,ts) in r0] 
        rs = [ts for (ti,ts) in r] 
        for (is1, is2) in zip(rs0, rs):
            self.iter_all_attrs(is1, is2)

    def test_run_5(self):
        """ test_run_5: using MachineStates object
        """
        latfile = self.testfile
        m0 = Machine(open(latfile, 'r'))
        ms = MachineStates(machine=m0) 

        fm = ModelFlame()
        fm.mstates = ms
        fm.machine = m0
        obs = fm.get_index_by_type(type='bpm')['bpm']
        r,s = fm.run(monitor=obs)

        s0 = m0.allocState({})
        m0.propagate(s0, 0, 1)
        r0 = m0.propagate(s0, 1, -1, observe=obs)
        rs0 = [ts for (ti,ts) in r0] 
        rs = [ts for (ti,ts) in r] 
        for (is1, is2) in zip(rs0, rs):
            self.iter_all_attrs(is1, is2)
        self.iter_all_attrs(s, s0)

    def test_collect_data(self):
        """ test_collect_data: get pos, x0, IonEk
        """
        latfile = self.testfile
        m0 = Machine(open(latfile, 'r'))
        s0 = m0.allocState({})
        r0 = m0.propagate(s0, 0, 100, observe=range(100))

        data0 = collect_data(r0, pos=True, x0=True, IonEk=True)

        fm = ModelFlame(latfile)
        r, s = fm.run(from_element=1, to_element=99, monitor=range(100))
        data = fm.collect_data(r, pos=True, x0=True, IonEk=True)

        self.assertEqual(data0['pos'][1:].tolist(), data['pos'].tolist())
        self.assertEqual(data0['x0'][1:].tolist(), data['x0'].tolist())
        self.assertEqual(data0['IonEk'][1:].tolist(), data['IonEk'].tolist())
 
    def iter_all_attrs(self, ms, s):
        k = ['beta', 'bg', 'gamma', 'IonEk', 'IonEs', 'IonQ', 'IonW', 
             'IonZ', 'phis', 'SampleIonK']
        all_keys = [i for i in k]
        all_keys.extend(['ref_{}'.format(i) for i in k] + ['pos'])
        all_keys.extend(['moment0', 'moment0_env', 'moment0_rms', 'moment1'])
        for attr in all_keys:
            left_val, right_val = getattr(ms, attr), getattr(s, attr)
            if isinstance(left_val, np.ndarray):
                #self.assertTrue((np.allclose(left_val, right_val, equal_nan=True) | (np.isnan(left_val) & np.isnan(right_val))).all())
                self.assertTrue(((left_val == right_val) | (np.isnan(left_val) & np.isnan(right_val))).all())
            else:
                self.assertAlmostEqual(left_val, right_val)

    def test_configure(self):
        latfile = self.testfile
        m0 = Machine(open(latfile, 'r'))
        s0 = m0.allocState({})
        e_cor_idx = 10
        m0.reconfigure(10, {'theta_x': 0.005})
        m0.propagate(s0, 0, 1)
        r0 = m0.propagate(s0, 1, -1, range(len(m0)))

        fm = ModelFlame(latfile)
        e = fm.get_element(index=10)[0]
        e['properties']['theta_x'] = 0.005
        fm.configure(e)
        r, s = fm.run(monitor=range(len(m0)))

        rs0 = [ts for (ti,ts) in r0] 
        rs = [ts for (ti,ts) in r] 
        for (is1, is2) in zip(rs0, rs):
            self.iter_all_attrs(is1, is2)


