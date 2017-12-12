# -*- coding: utf-8 -*-

import unittest
import os
from cStringIO import StringIO

from flame import Machine
from _utils import make_latfile
from flame_utils import generate_latfile

curdir = os.path.dirname(__file__)


class TestGenerateLatfile(unittest.TestCase):
    def setUp(self):
        testfile = os.path.join(curdir, 'lattice/test_0.lat')
        out1file = os.path.join(curdir, 'lattice/out1_0.lat')
        out2file = os.path.join(curdir, 'lattice/out2_0.lat')
        self.testfile = make_latfile(testfile)
        self.out1file = make_latfile(out1file)
        self.out2file = make_latfile(out2file)

        ftest = open(self.testfile)
        self.m = Machine(ftest)
        s0 = self.m.allocState({})
        self.r = self.m.propagate(s0, 0, len(self.m), range(len(self.m)))
        self.s = s0
        k = ['beta', 'bg', 'gamma', 'IonEk', 'IonEs', 'IonQ', 'IonW', 
             'IonZ', 'phis', 'SampleIonK']
        self.keys = [i for i in k]
        self.ref_keys = ['ref_{}'.format(i) for i in k] + ['pos']
        self.keys_ps = ['moment0', 'moment0_env', 'moment0_rms', 'moment1']

        self.fout1_str = open(self.out1file).read().strip()
        self.fout2_str = open(self.out2file).read().strip()

        self.latfile1 = os.path.join(curdir, 'lattice/out1_org.lat')
        self.latfile2 = os.path.join(curdir, 'lattice/out1_mod.lat')

    def tearDown(self):
        for f in [self.latfile1, self.latfile2]:
            if os.path.isfile(f):
                os.remove(f)

    def test_generate_latfile_original1(self):
        sio = StringIO()
        sioname = generate_latfile(self.m, out=sio)
        self.assertEqual(sioname, 'string')
        self.assertEqual(sio.getvalue().strip(), self.fout1_str)
        
    def test_generate_latfile_original2(self):
        # TEST LATFILE
        fout1_file = generate_latfile(self.m, self.latfile1)
        lines1 = [line.strip() for line in open(fout1_file).read().strip().split('\n')]
        lines0 = [line.strip() for line in self.fout1_str.split('\n')]
        self.assertEqual(lines1, lines0)
        
        m = Machine(open(fout1_file))
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
    
    def test_generate_latfile_update(self):
        idx = 80
        self.m.reconfigure(idx, {'theta_x': 0.1})
        fout2_file = generate_latfile(self.m, self.latfile2)
        self.assertEqual(open(fout2_file).read().strip(), self.fout2_str)
        
        m = Machine(open(fout2_file))
        self.assertEqual(m.conf(idx)['theta_x'], 0.1)
