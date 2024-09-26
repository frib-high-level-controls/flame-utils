# -*- coding: utf-8 -*-

import unittest
import os
import numpy as np
import random

from flame import Machine
from flame_utils import BeamState
from flame_utils import ModelFlame
from flame_utils import collect_data
from flame_utils import generate_source
from flame_utils import twiss_to_matrix

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
        mat = twiss_to_matrix('x', 1, 2, 3)
        mat = twiss_to_matrix('y', 1, 2, 3, matrix=mat)
        mat = twiss_to_matrix('z', 1, 2, 3, matrix=mat)
        for i in [0, 2]:
            self.assertAlmostEqual(mat[i,   i],    6.0)
            self.assertAlmostEqual(mat[i,   i+1], -3e-3)
            self.assertAlmostEqual(mat[i+1, i],   -3e-3)
            self.assertAlmostEqual(mat[i+1, i+1],  3e-06)
        self.assertAlmostEqual(mat[4, 4],  6.0)
        self.assertAlmostEqual(mat[4, 5], -3e0)
        self.assertAlmostEqual(mat[5, 4], -3e0)
        self.assertAlmostEqual(mat[5, 5],  3e0)
        mat = twiss_to_matrix('x', np.nan, np.inf, 3)
        self.assertAlmostEqual(mat[0, 0], 0.0)
        self.assertAlmostEqual(mat[0, 1], 0.0)
        self.assertAlmostEqual(mat[1, 0], 0.0)
        self.assertAlmostEqual(mat[1, 1], 0.0)

    def test_transmat(self):
        with open(self.latfile, 'rb') as f:
            m = Machine(f)
        s = m.allocState({})
        m.propagate(s, 0, 10)
        ms = BeamState(s)
        left_val = ms.transfer_matrix
        right_val = s.transmat
        self.assertTrue((left_val == right_val).all())

    def test_set_ref(self):
        ms = BeamState(latfile=self.latfile)
        ms.ref_beta = 0.03
        self.assertAlmostEqual(ms.ref_IonEk, 419455.4536757469)
        ms.ref_bg = 0.03
        self.assertAlmostEqual(ms.ref_IonEk, 419078.028649807)
        ms.ref_gamma = 1.005
        self.assertAlmostEqual(ms.ref_IonEk, 4657469.999999881)
        ms.ref_IonEk = 500000.0
        self.assertAlmostEqual(ms.ref_IonEk, 500000.0)

        ms.ref_IonEs = 931494000.0 + 1.0
        self.assertAlmostEqual(ms.ref_IonEs, 931494001.0)
        ms.ref_IonEs = 931494000.0

        ms.ref_IonQ = 10000.0
        self.assertAlmostEqual(ms.ref_IonQ, 10000.0)

        ms.ref_IonW = 931994000.0 + 1.0
        self.assertAlmostEqual(ms.ref_IonEk, 500001.0)

        ms.ref_phis = 1e-4
        self.assertAlmostEqual(ms.moment0[4, 0], -0.000284773)

        ms.ref_SampleFreq = 40.25e6
        self.assertAlmostEqual(ms.ref_SampleFreq, 40250000.0)

        ms.ref_Brho = 0.7
        self.assertAlmostEqual(ms.ref_IonEk, 454352.15719020367)

    def test_set_real(self):
        ms = BeamState(latfile=self.latfile)
        ms.beta = np.array([0.03])
        self.assertAlmostEqual(ms.IonEk[0], 419455.4536757469)
        ms.bg = np.array([0.03])
        self.assertAlmostEqual(ms.IonEk[0], 419078.028649807)
        ms.gamma = np.array([1.005])
        self.assertAlmostEqual(ms.IonEk[0], 4657469.999999881)
        ms.IonEk = np.array([500000.0])
        self.assertAlmostEqual(ms.IonEk[0], 500000.0)

        ms.IonEs = np.array([931494000.0 + 1.0])
        self.assertAlmostEqual(ms.IonEs[0], 931494001.0)
        ms.IonEs = np.array([931494000.0])

        ms.IonQ = np.array([10000.0])
        self.assertAlmostEqual(ms.IonQ[0], 10000.0)

        ms.IonW = np.array([931994000.0 + 1.0])
        self.assertAlmostEqual(ms.IonEk[0], 500001.0)

        ms.phis = np.array([1e-4])
        self.assertAlmostEqual(ms.moment0[4, 0], 1e-4)

        ms.SampleFreq = np.array([40.25e6])
        self.assertAlmostEqual(ms.SampleFreq[0], 40250000.0)

        ms.Brho = np.array(0.7)
        self.assertAlmostEqual(ms.IonEk[0], 454352.15719020367)

    def test_set_centroid(self):
        ms = BeamState(latfile=self.latfile)
        ms.xcen = 1.0
        self.assertAlmostEqual(ms.moment0[0, 0], 1.0)
        ms.xpcen = 1.0
        self.assertAlmostEqual(ms.moment0[1, 0], 1.0)
        ms.ycen = 1.0
        self.assertAlmostEqual(ms.moment0[2, 0], 1.0)
        ms.ypcen = 1.0
        self.assertAlmostEqual(ms.moment0[3, 0], 1.0)
        ms.zcen = 1.0
        self.assertAlmostEqual(ms.moment0[4, 0], 1.0)
        ms.zpcen = 1.0
        self.assertAlmostEqual(ms.moment0[5, 0], 1.0)

        ms.xcen_all = np.array([2.0])
        self.assertAlmostEqual(ms.moment0[0, 0], 2.0)
        ms.xpcen_all = np.array([2.0])
        self.assertAlmostEqual(ms.moment0[1, 0], 2.0)
        ms.ycen_all = np.array([2.0])
        self.assertAlmostEqual(ms.moment0[2, 0], 2.0)
        ms.ypcen_all = np.array([2.0])
        self.assertAlmostEqual(ms.moment0[3, 0], 2.0)
        ms.zcen_all = np.array([2.0])
        self.assertAlmostEqual(ms.moment0[4, 0], 2.0)
        ms.zpcen_all = np.array([2.0])
        self.assertAlmostEqual(ms.moment0[5, 0], 2.0)

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

    def test_find(self):
        fm = ModelFlame(self.testfile)
        m = fm.machine
        all_types = fm.get_all_types()
        for i in range(2, 20):
            e = fm.find(m.conf(i)['name'])[0]
            self.assertEqual(i, e)
        for ntype in all_types:
            e0 = m.find(type=ntype)
            e  = fm.find(type=ntype)
            self.assertEqual(e, e0)

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
        m0.propagate(s0, 0, 10)
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

    def test_run_8(self):
        """ test_run_8: include_initial_state
        """
        latfile = self.testfile
        with open(latfile, 'rb') as f:
            m0 = Machine(f)
        s0 = m0.allocState({})
        fm = ModelFlame(latfile)
        m0.propagate(s0, 0, 1)
        r0 = m0.propagate(s0, 1, len(m0), observe=range(len(m0)))
        r,s = fm.run(monitor='all', include_initial_state=False)
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

    def test_transfer_matrix(self):
        """Calculate transfer matrix from A to B.
        """
        cs = 0
        latfile = self.testfile
        fm = ModelFlame(latfile)
        r, s = fm.run(from_element=1, to_element=3, monitor=[1,2,3])
        s1 = r[0][-1]
        s2 = r[1][-1]
        s3 = r[2][-1]
        m21 = fm.get_transfer_matrix(from_element=1, to_element=2,
                                     charge_state_index=cs)
        m31 = fm.get_transfer_matrix(from_element=1, to_element=3,
                                     charge_state_index=cs)
        self.assertEqual(m21.tolist(),
                         s2.transfer_matrix[:, :, cs].tolist())
        self.assertEqual(m31.tolist(),
                         np.dot(
                            s3.transfer_matrix[:, :, cs],
                            s2.transfer_matrix[:, :, cs]).tolist())
        for i in range(7):
            self.assertAlmostEqual(np.dot(m31, s1.moment0[:,0])[i],
                                   s3.moment0[:,0][i])


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

    def test_pop_in_modelflame(self):
        latfile = self.testfile
        fm = ModelFlame(latfile)
        r0,s0 = fm.run(to_element=6)
        total_before_pop = len(fm.machine)
        fm.pop_element(4)
        total_after_pop = len(fm.machine)
        self.assertEqual(total_before_pop-1, total_after_pop)