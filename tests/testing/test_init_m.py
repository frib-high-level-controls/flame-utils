from flame import Machine

m = Machine(open('test_0.lat'))

m1 = Machine(m.conf())

s = m.allocState({})
r = m.propagate(s, 0, -1, range(len(m)))
print(s)

s1 = m1.allocState({})
r1 = m1.propagate(s1, 0, -1, range(len(m1)))
print(s1)

