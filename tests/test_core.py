# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import os
import numpy as np
import random

from flame import Machine
from flame_utils import BeamState
from flame_utils import ModelFlame
from flame_utils import collect_data
from flame_utils import generate_source

from _utils import make_latfile
from _utils import compare_mstates
from _utils import compare_source_element

curdir = os.path.dirname(__file__)


class TestBeamState(unittest.TestCase):
    def setUp(self):
        latfile = os.path.join(curdir, 'lattice/test_0.lat')
        self.latfile = make_latfile(latfile)

    def test_init_with_s1(self):
        """ test_init_with_s1: s is not None
        """
        with open(self.latfile, 'rb') as f:
            m = Machine(f)
        s0 = m.allocState({})
        s1 = s0.clone()
        m.propagate(s1, 0, 1)

        ms0 = BeamState(s0)
        compare_mstates(self, ms0, s0)

        ms1 = BeamState(s0, machine=m)
        compare_mstates(self, ms1, s1)

        ms1_1 = BeamState(s0, latfile=self.latfile)
        compare_mstates(self, ms1_1, s1)

    def test_init_with_s2(self):
        """ test_init_with_s2: s is None
        """
        with open(self.latfile, 'rb') as f:
            m = Machine(f)
        s = m.allocState({})
        m.propagate(s, 0, 1)
        ms = BeamState()
        ms.state = s
        compare_mstates(self, ms, s)

    def test_init_with_machine(self):
        with open(self.latfile, 'rb') as f:
            m = Machine(f)
        ms = BeamState(machine=m)
        s = m.allocState({})
        m.propagate(s, 0, 1)
        compare_mstates(self, ms, s)

    def test_init_with_latfile(self):
        with open(self.latfile, 'rb') as f:
            m = Machine(f)
        ms = BeamState(latfile=self.latfile)
        s = m.allocState({})
        m.propagate(s, 0, 1)
        compare_mstates(self, ms, s)

    def test_init_with_mix(self):
        with open(self.latfile, 'rb') as f:
            m = Machine(f)
        ms = BeamState(machine=m, latfile=self.latfile)
        s = m.allocState({})
        m.propagate(s, 0, 1)
        compare_mstates(self, ms, s)

    def test_attr_alias(self):
        aliases = {
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
        ms = BeamState(latfile=self.latfile)
        for k,v in aliases.items():
            left_val, right_val = getattr(ms, k), getattr(ms, v)
            if isinstance(left_val, np.ndarray):
                self.assertTrue(((left_val == right_val) | (np.isnan(left_val) & np.isnan(right_val))).all())
            else:
                self.assertAlmostEqual(left_val, right_val)

    def test_rms_size(self):
        ms = BeamState(latfile=self.latfile)
        for k in ('xrms', 'yrms', 'xprms', 'yprms', 'phirms', 'dEkrms'):
            self.assertEqual(getattr(ms, k), getattr(ms, k + '_all')[0])

    def test_twiss_parameter(self):
        ms = BeamState(latfile=self.latfile)
        self.assertAlmostEqual(ms.xtwsb, ms.xrms*ms.xrms/ms.xeps)
        self.assertAlmostEqual(ms.ytwsb, ms.yrms*ms.yrms/ms.yeps)
        self.assertAlmostEqual(ms.ztwsb, ms.phirms*ms.phirms/ms.zeps)
        self.assertAlmostEqual(ms.xtwsa, -ms.moment1_env[0, 1]/ms.xeps*1e3)
        self.assertAlmostEqual(ms.ytwsa, -ms.moment1_env[2, 3]/ms.yeps*1e3)
        self.assertAlmostEqual(ms.ztwsa, -ms.moment1_env[4, 5]/ms.zeps)
        ms.set_twiss('x', alpha=0.2, beta=3.0, emittance=5.0, cs=0)
        ms.set_twiss('y', alpha=0.2, beta=3.0, emittance=5.0, cs=0)
        ms.set_twiss('z', alpha=0.2, beta=3.0, emittance=5.0, cs=0)
        self.assertAlmostEqual(ms.xtwsa_all[0], 0.2)
        self.assertAlmostEqual(ms.ytwsa_all[0], 0.2)
        self.assertAlmostEqual(ms.ztwsa_all[0], 0.2)
        self.assertAlmostEqual(ms.xtwsb_all[0], 3.0)
        self.assertAlmostEqual(ms.ytwsb_all[0], 3.0)
        self.assertAlmostEqual(ms.ztwsb_all[0], 3.0)
        self.assertAlmostEqual(ms.xeps_all[0], 5.0)
        self.assertAlmostEqual(ms.yeps_all[0], 5.0)
        self.assertAlmostEqual(ms.zeps_all[0], 5.0)

    def test_transmat(self):
        with open(self.latfile, 'rb') as f:
            m = Machine(f)
        s = m.allocState({})
        m.propagate(s, 0, 10)
        ms = BeamState(s)
        left_val = ms.transfer_matrix
        right_val = s.transmat
        self.assertTrue((left_val == right_val).all())

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
        with open(self.testfile, 'rb') as f:
            m = Machine(f)
        fm_none.machine = m
        self.assertEqual(fm_none.machine, m)

    def test_set_bmstate(self):
        fm_none = ModelFlame()
        self.assertIsNone(fm_none.bmstate)
        with open(self.testfile, 'rb') as f:
            m = Machine(f)
        s = m.allocState({})
        m.propagate(s, 0, 1)
        fm_none.bmstate = s
        compare_mstates(self, fm_none.bmstate, s)

    def test_init_machine(self):
        fm_none = ModelFlame()
        m, s = fm_none.init_machine(self.testfile)
        fm_none.machine, fm_none.bmstate = m, s
        self.assertEqual(fm_none.machine, m)
        compare_mstates(self, fm_none.bmstate, s)

    def test_get_all_types(self):
        fm = ModelFlame(self.testfile)
        etypes = {'quadrupole', 'bpm', 'drift', 'source', 'rfcavity',
                  'sbend', 'orbtrim', 'solenoid', 'stripper'}
        self.assertEqual(set(fm.get_all_types()), etypes)

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
        with open(latfile, 'rb') as f:
            m0 = Machine(f)
        s0 = m0.allocState({})
        m0.propagate(s0, 0, len(m0))
        fm = ModelFlame(latfile)
        r,s = fm.run()
        self.assertEqual(r, [])
        compare_mstates(self, s, s0)

    def test_run_2(self):
        """ test_run_2: propagate from the first to last, monitor all BPMs
        """
        latfile = self.testfile
        with open(latfile, 'rb') as f:
            m0 = Machine(f)
        s0 = m0.allocState({})
        fm = ModelFlame(latfile)
        obs = fm.get_index_by_type(type='bpm')['bpm']
        r0 = m0.propagate(s0, 0, len(m0), observe=obs)
        r,s = fm.run(monitor=obs)
        rs0 = [ts for (ti,ts) in r0]
        rs = [ts for (ti,ts) in r]
        for (is1, is2) in zip(rs0, rs):
            compare_mstates(self, is1, is2)
        compare_mstates(self, s, s0)

    def test_run_3(self):
        """ test run_3: test initial states
        """
        latfile = self.testfile
        with open(latfile, 'rb') as f:
            m0 = Machine(f)
        s0 = m0.allocState({})
        m0.propagate(s0, 0, 1)
        fm = ModelFlame(latfile)
        r, s = fm.run(from_element=0, to_element=0)
        compare_mstates(self, s0, s)

    def test_run_4(self):
        """ test_run_4: run and monitor from element index of 10 to 20
        """
        latfile = self.testfile
        with open(latfile, 'rb') as f:
            m0 = Machine(f)
        s0 = m0.allocState({})
        m0.propagate(s0, 0, 1)
        r0 = m0.propagate(s0, 10, 11, observe=range(10, 21))

        fm = ModelFlame(latfile)
        r, s = fm.run(from_element=10, to_element=20, monitor=range(10,21))
        compare_mstates(self, s0, s)

        rs0 = [ts for (ti,ts) in r0]
        rs = [ts for (ti,ts) in r]
        for (is1, is2) in zip(rs0, rs):
            compare_mstates(self, is1, is2)

    def test_run_5(self):
        """ test_run_5: using BeamState object
        """
        latfile = self.testfile
        with open(latfile, 'rb') as f:
            m0 = Machine(f)
        ms = BeamState(machine=m0)

        fm = ModelFlame()
        fm.bmstate = ms
        fm.machine = m0
        obs = fm.get_index_by_type(type='bpm')['bpm']
        r,s = fm.run(monitor=obs)

        s0 = m0.allocState({})
        m0.propagate(s0, 0, 1)
        r0 = m0.propagate(s0, 1, len(m0), observe=obs)
        rs0 = [ts for (ti,ts) in r0]
        rs = [ts for (ti,ts) in r]
        for (is1, is2) in zip(rs0, rs):
            compare_mstates(self, is1, is2)
        compare_mstates(self, s, s0)

    def test_run_6(self):
        """ test_run_6: optional monitor setting 'all'
        """
        latfile = self.testfile
        with open(latfile, 'rb') as f:
            m0 = Machine(f)
        s0 = m0.allocState({})
        fm = ModelFlame(latfile)
        r0 = m0.propagate(s0, 0, len(m0), observe=range(len(m0)))
        r,s = fm.run(monitor='all')
        rs0 = [ts for (ti,ts) in r0]
        rs = [ts for (ti,ts) in r]
        for (is1, is2) in zip(rs0, rs):
            compare_mstates(self, is1, is2)
        compare_mstates(self, s, s0)

    def test_run_7(self):
        """ test_run_7: optional monitor setting 'type'
        """
        latfile = self.testfile
        with open(latfile, 'rb') as f:
            m0 = Machine(f)
        s0 = m0.allocState({})
        fm = ModelFlame(latfile)
        obs = fm.get_index_by_type(type='bpm')['bpm']
        r0 = m0.propagate(s0, 0, len(m0), observe=obs)
        r,s = fm.run(monitor='bpm')
        rs0 = [ts for (ti,ts) in r0]
        rs = [ts for (ti,ts) in r]
        for (is1, is2) in zip(rs0, rs):
            compare_mstates(self, is1, is2)
        compare_mstates(self, s, s0)

    def test_collect_data(self):
        """ test_collect_data: get pos, x0, IonEk
        """
        latfile = self.testfile
        with open(latfile, 'rb') as f:
            m0 = Machine(f)
        s0 = m0.allocState({})
        r0 = m0.propagate(s0, 0, 100, observe=range(100))

        data0 = collect_data(r0, pos=True, x0=True, IonEk=True)
        data0_1 = collect_data(r0, 'pos', 'x0', 'IonEk')
        data0_2 = collect_data(r0, 'pos', 'x0', IonEk=True)

        for k in ('pos', 'x0', 'IonEk'):
            self.assertEqual(data0[k].tolist(), data0_1[k].tolist())
            self.assertEqual(data0[k].tolist(), data0_2[k].tolist())

        fm = ModelFlame(latfile)
        r, s = fm.run(from_element=1, to_element=99, monitor=range(100))
        data = fm.collect_data(r, pos=True, x0=True, IonEk=True)

        self.assertEqual(data0['pos'].tolist(), data['pos'].tolist())
        self.assertEqual(data0['x0'].tolist(), data['x0'].tolist())
        self.assertEqual(data0['IonEk'].tolist(), data['IonEk'].tolist())

    def test_configure(self):
        latfile = self.testfile
        with open(latfile, 'rb') as f:
            m0 = Machine(f)
        s0 = m0.allocState({})
        e_cor_idx = 10
        m0.reconfigure(10, {'theta_x': 0.005})
        r0 = m0.propagate(s0, 0, len(m0), range(len(m0)))

        fm = ModelFlame(latfile)
        e = fm.get_element(index=10)[0]
        e['properties']['theta_x'] = 0.005
        fm.configure(e)
        r, s = fm.run(monitor=range(len(m0)))

        rs0 = [ts for (ti,ts) in r0]
        rs = [ts for (ti,ts) in r]
        for (is1, is2) in zip(rs0, rs):
            compare_mstates(self, is1, is2)

    def test_reconfigure(self):
        latfile = self.testfile
        with open(latfile, 'rb') as f:
            m0 = Machine(f)
        s0 = m0.allocState({})
        e_cor_idx = 10
        e_name = m0.conf(e_cor_idx)['name']
        m0.reconfigure(10, {'theta_x': 0.005})
        r0 = m0.propagate(s0, 0, len(m0), range(len(m0)))

        fm = ModelFlame(latfile)
        fm.reconfigure(e_name, {'theta_x': 0.005})
        r, s = fm.run(monitor=range(len(m0)))

        rs0 = [ts for (ti,ts) in r0]
        rs = [ts for (ti,ts) in r]
        for (is1, is2) in zip(rs0, rs):
            compare_mstates(self, is1, is2)

    def test_configure_source1(self):
        """Update source, as well as Ion_Z (and others)
        """
        latfile = self.testfile
        fm = ModelFlame(lat_file=latfile)
        s = generate_source(fm.bmstate)
        s['properties']['IonChargeStates'] = np.asarray([0.1, ])
        fm.configure(econf=s)
        self.assertEqual(fm.bmstate.IonZ, np.asarray([0.1, ]))


class TestStateToSource(unittest.TestCase):
    def setUp(self):
        testfile = os.path.join(curdir, 'lattice/test_0.lat')
        self.testfile = make_latfile(testfile)

    def test_generate_source(self):
        latfile = self.testfile
        fm = ModelFlame(latfile)
        ms = fm.bmstate
        sconf = generate_source(ms)
        sconf0 = fm.get_element(type='source')[0]
        compare_source_element(self, sconf, sconf0)

        r0, s0 = fm.run(monitor=range(len(fm.machine)))
        fm.configure(sconf)
        r, s = fm.run(monitor=range(len(fm.machine)))
        compare_mstates(self, s, s0)

        rs0 = [ts for (ti,ts) in r0]
        rs = [ts for (ti,ts) in r]
        for (is1, is2) in zip(rs0, rs):
            compare_mstates(self, is1, is2)


class TestInsertElemInModelFlame(unittest.TestCase):
    def setUp(self):
        testfile = os.path.join(curdir, 'lattice/test_0.lat')
        self.testfile = make_latfile(testfile)

    def test_insert_in_modelflame(self):
        latfile = self.testfile
        fm = ModelFlame(latfile)
        r0,s0 = fm.run(to_element=6)
        econf_before_insertion = fm.get_element(index=5)[0]
        total_before_insertion = len(fm.machine)

        new_econf = {'index':5, 'properties':{'name':'test_drift', 'type':'drift', 'L':0.05588}}
        fm.insert_element(econf=new_econf)
        total_after_insertion = len(fm._mach_ins)
        test_econf = fm.get_element(index=5)[0]
        self.assertEqual(test_econf['index'], new_econf['index'])
        self.assertEqual(test_econf['properties']['name'], new_econf['properties']['name'])
        self.assertEqual(test_econf['properties']['type'], new_econf['properties']['type'])
        self.assertEqual(test_econf['properties']['L'], new_econf['properties']['L'])
        self.assertEqual(total_before_insertion+1, total_after_insertion)

        test_econf2 = fm.get_element(index=6)[0]
        self.assertEqual(test_econf2['index'], 6)
        self.assertEqual(test_econf2['properties']['name'], econf_before_insertion['properties']['name'])
        self.assertEqual(test_econf2['properties']['type'], econf_before_insertion['properties']['type'])

        r1,s1 = fm.run(to_element=6)

        compare_mstates(self, s0, s1)
