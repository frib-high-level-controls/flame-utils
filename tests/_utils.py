# -*- coding: utf-8 -*-

import os


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

