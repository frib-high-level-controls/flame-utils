# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import os

from flame import Machine
from _utils import make_latfile
from flame_utils import generate_latfile
from numpy.testing import assert_array_almost_equal as assertAEqual

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

curdir = os.path.dirname(__file__)


class TestGenerateLatfile(unittest.TestCase):
    def setUp(self):
        testfile = os.path.join(curdir, 'lattice/test_0.lat')
        out1file = os.path.join(curdir, 'lattice/out1_0.lat')
        out2file = os.path.join(curdir, 'lattice/out2_0.lat')
        self.testfile = make_latfile(testfile)
        self.out1file = make_latfile(out1file)
        self.out2file = make_latfile(out2file)

        with open(self.testfile, 'rb') as f:
            self.m = Machine(f)
        s0 = self.m.allocState({})
        self.r = self.m.propagate(s0, 0, len(self.m), range(len(self.m)))
        self.s = s0
        k = ['beta', 'bg', 'gamma', 'IonEk', 'IonEs', 'IonQ', 'IonW',
             'IonZ', 'phis', 'SampleIonK']
        self.keys = [i for i in k]
        self.ref_keys = ['ref_{}'.format(i) for i in k] + ['pos']
        self.keys_ps = ['moment0', 'moment0_env', 'moment0_rms', 'moment1']

        with open(self.out1file, 'rb') as f:
            self.fout1_str = f.read().strip().decode()

        with open(self.out2file, 'rb') as f:
            self.fout2_str = f.read().strip().decode()

        self.latfile0 = os.path.join(curdir, 'lattice/test_1.lat')
        self.latfile1 = os.path.join(curdir, 'lattice/out1_1.lat')
        self.latfile2 = os.path.join(curdir, 'lattice/out2_1.lat')

        self.testcfile = os.path.join(curdir, 'lattice/test_c.lat')
        self.out3file = os.path.join(curdir, 'lattice/out3.lat')

        with open(self.out3file, 'rb') as f:
            self.fout3_str = f.read().strip().decode()

        with open(self.testcfile, 'rb') as f:
            self.mc = Machine(f)
            self.foutc_str = f.read().strip().decode()

        out4file = os.path.join(curdir, 'lattice/out4_0.lat')
        self.out4file = make_latfile(out4file)

        with open(self.out4file, 'rb') as f:
            self.fout4_str = f.read().strip().decode()

        self.latfile4 = os.path.join(curdir, 'lattice/out4_1.lat')

    def tearDown(self):
        for f in [self.latfile0, self.latfile1, self.latfile2, self.latfile4]:
            if os.path.isfile(f):
                os.remove(f)

    def test_generate_latfile_original1(self):
        sio = StringIO()
        sioname = generate_latfile(self.m, out=sio)
        self.assertEqual(sioname, 'string')
        self.assertEqual(sio.getvalue().strip(), self.fout1_str)

    def test_generate_latfile_original2(self):
        # TEST LATFILE
        fout1_file = generate_latfile(self.m, latfile=self.latfile1)
        with open(fout1_file, 'rb') as f:
            f_str = f.read().strip().decode()
        self.assertEqual(f_str, self.fout1_str)

        with open(fout1_file, 'rb') as f:
            m = Machine(f)
        s = m.allocState({})
        r = m.propagate(s, 0, len(m), range(len(m)))

        # TEST RESULTS
        for i in range(len(m)):
            s1, s0 = r[i][1], self.r[i][1]
            for k in self.ref_keys:
                self.assertEqual(getattr(s1, k), getattr(s0, k))

            for k in self.keys:
                self.assertEqual(getattr(s1, k).tolist(),
                                 getattr(s0, k).tolist())

            for k in self.keys_ps:
                self.assertEqual(getattr(s1, k).tolist(),
                                 getattr(s0, k).tolist())

    @unittest.skip("output precision depends on the python version")
    def test_generate_latfile_original3(self):
        sio = StringIO()
        sioname = generate_latfile(self.mc, original=self.testcfile, out=sio)
        self.assertEqual(sioname, 'string')
        self.assertEqual(sio.getvalue().strip(), self.fout3_str)

    def test_generate_latfile_original4(self):
        sio = StringIO()
        sioname = generate_latfile(self.m, start=30, end=60, out=sio)
        self.assertEqual(sioname, 'string')
        self.assertEqual(sio.getvalue().strip(), self.fout4_str)

    def test_generate_latfile_update1(self):
        idx = 80
        self.m.reconfigure(idx, {'theta_x': 0.1})
        fout2_file = generate_latfile(self.m, latfile=self.latfile2)
        with open(fout2_file) as f:
            self.assertEqual(f.read().strip(), self.fout2_str)

        with open(fout2_file, 'rb') as f:
            m = Machine(f)
        self.assertEqual(m.conf(idx)['theta_x'], 0.1)

    def test_generate_latfile_update2(self):
        idx = 0
        s = self.m.allocState({})
        self.m.propagate(s,0,1)
        s.moment0 = [[0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [1.0]]
        fout2_file = generate_latfile(self.m, latfile=self.latfile2, state=s)

        with open(fout2_file, 'rb') as f:
            m = Machine(f)
        assertAEqual(m.conf()['P0'], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0])
