from time import clock, time
import numpy as np
from CorrCoef import Pearson

n = 1000
r = np.random.random((n, n))

u, s = clock(), time()
a = np.corrcoef(r)
v, t = clock(), time()

print 'numpy.corrcoef():'
print 'process time: %.1f sec.' % (v - u)
print 'wall time: %.1f sec.' % (t - s)

b = a[np.triu_indices(n, 1)]
del a

u, s = clock(), time()
a = Pearson(r)
v, t = clock(), time()

print '------------------------------------'
print 'CorrCoef.Pearson():'
print 'process time: %.1f sec.' % (v - u)
print 'wall time: %.1f sec.' % (t - s)
print '------------------------------------'
print 'Maximum deviation:', np.max(b - a)
