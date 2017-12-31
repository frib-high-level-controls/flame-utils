# -*- coding: utf-8 -*-

import os
import numpy as np

curdir = os.path.dirname(__file__)
FLAME_DATA = os.path.join(curdir, "lattice/flame_data")


def make_latfile(latfile1):
    f1 = latfile1
    f1_pathname = os.path.dirname(f1)
    f1_filename = os.path.basename(f1)
    f2_filename = f1_filename.replace('0', '1')
    f2_pathname = f1_pathname
    f2 = os.path.join(f2_pathname, f2_filename)
    fp2 = open(f2, 'w')
    for line in open(f1, 'r'):
        if line.startswith('Eng'):
            name, _ = line.split('=')
            line = '{0} = "{1}";\n'.format(name.strip(),
                                           os.path.abspath(FLAME_DATA))
        fp2.write(line)
    fp2.close()
    return f2

def compare_source_element(self, s0, s1):
    for k,v in s0['properties'].items():
        left_val, right_val = v, s1['properties'][k]
        if isinstance(v, np.ndarray):
            self.assertTrue(((left_val == right_val) | (np.isnan(left_val) & np.isnan(right_val))).all())
        else:
            self.assertAlmostEqual(left_val, right_val)

def compare_mstates(self, ms, s):
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
