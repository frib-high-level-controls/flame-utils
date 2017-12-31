#!/usr/bin/python


from flame_utils import ModelFlame


latfile = '../lattice/test_0.lat'

mf = ModelFlame(lat_file=latfile)


src0 = mf.get_element(type='source')[0]

r_1,s_1 = mf.run(from_element=0, monitor=-1)

src0['properties']['IonEk'] = 400000.0

mf.configure(econf=src0)
r_2,s_2 = mf.run(from_element=0, monitor=-1)

from flame_utils import MachineStates
ms = MachineStates(mf.mstates.clone())
print(ms.moment0)
ms.IonEk = [400000.0] + ms.moment0[5]*1e6
ms.ref_IonEk = 400000.0
r_3,s_3 = mf.run(mstates=ms,monitor=-1)


print(s_1)
print(s_2)
print(s_3)



