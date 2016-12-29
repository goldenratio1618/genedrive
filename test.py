import numpy as np
from timeit import default_timer as timer
from gameoflife import *

size = 1000
start = timer()
coord = np.array((0,0,0))
for i in range(size * size):
    junk = torusAdjFunc(coord, np.array([512,512]))
dt = timer() - start
print("Time to run adjFunc 1000 times: %f" % dt)
# torusAdjFunc: 10.59 seconds