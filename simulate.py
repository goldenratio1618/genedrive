from copy import deepcopy
from math import floor
import cmath
from numba import *
import numpy as np
from timeit import default_timer as timer
from random import randrange
from operator import mul
from queue import Queue

def cNorm(a):
    return a.real + a.imag

def dual(dim, adjGrid):
    # create an array with the same shape as grid
    iter_arr = np.zeros(adjGrid.shape[0:len(adjGrid.shape) - 1])
    dualAdjGrid = np.zeros(adjGrid.shape, dtype=np.int32)
    currentPos = np.zeros(dim, dtype=np.int8)
    it = np.nditer(iter_arr, flags=['multi_index'])

    # initialize dual adjGrid
    while not it.finished:
        dualAdjGrid[it.multi_index] = dim
        it.iternext()
    
    it = np.nditer(iter_arr, flags=['multi_index'])    
    while not it.finished:
        pos = it.multi_index[0:len(dim)]
        if (adjGrid[it.multi_index] != dim).any():
            dualAdjGrid[tuple(adjGrid[it.multi_index])][currentPos[pos]] = pos
            currentPos[pos] += 1
        it.iternext()

    return dualAdjGrid

def unFlatten(ind, dim):
    """ Turns an index of a raveled array back into the unraveled verison. Assumes 2D array """
    if len(dim) == 1:
        return (ind,)
    elif len(dim) == 2:
        return ((int)(ind / dim[1]), ind % dim[1])
    else:
        raise ValueError("Only one and two dimensional graphs are supported.")

""" Code for initializing fitness values for each location in grid."""
class FDSCP:
    def __init__(self, dim, payoffMatrix, adjGrid, grid, dualAdjGrid=None, randomPlacement=False):
        self.dim = dim
        self.payoffMatrix = payoffMatrix
        self.adjGrid = adjGrid
        self.grid = grid
        self.totElements = 1
        self.randomPlacement = randomPlacement
        for i in range(len(self.dim)):
            self.totElements *= self.dim[i]
        if dualAdjGrid is None:
            self.dualAdjGrid = dual(self.dim, self.adjGrid)
        else:
            self.dualAdjGrid = dualAdjGrid
        self.init()


    def initFitnesses(self):
        fitnesses = np.zeros(self.dim, dtype=np.complex128)
        it = np.nditer(fitnesses, flags=['multi_index'])
        while not it.finished:
            fitnesses[it.multi_index] = self.computeFitness(it.multi_index)
            it.iternext()
        return fitnesses

    def init(self):
        self.fitnesses = self.initFitnesses()
        self.numMutants = 0
        it = np.nditer(self.grid, flags=['multi_index'])
        while not it.finished:
            if (it.multi_index != self.dim).any():
                self.numMutants += it[0]
            it.iternext()
        self.gridRavel = np.ravel(self.grid)
        self.fitnessReal = np.array(list(map(cNorm, self.fitnesses)))
        self.fitnessRavel = np.ravel(self.fitnessReal)
        self.totFitness = np.sum(self.fitnessReal)
        self.indList = np.arange(len(self.fitnessRavel))
        

    def computeFitness(self, loc):
        """ Computes the fitness of the individual at location loc. """
        payoff = 0
        for adjLoc in self.adjGrid[loc]:
            if (adjLoc != self.dim).any():
                payoff += self.payoffMatrix[self.grid[loc]][self.grid[tuple(adjLoc)]]
        return payoff

    def update(self, loc, old, new):
        """ Updates fitnesses and edge probabilities when the individual at loc changes
            type from old to new """
        loc = tuple(loc)
        # nothing changed
        if old == new:
            return
        # a new mutant (1) was born
        if old < new:
            self.numMutants += 1
        # a mutant was replaced
        if old > new:
            self.numMutants -= 1
        # update fitnesses of all of loc's neighbors
        for connLoc in self.dualAdjGrid[loc]:
            if (connLoc == self.dim).all():
                continue
            cl = tuple(connLoc)
            diff = self.payoffMatrix[self.grid[cl]][new] - self.payoffMatrix[self.grid[cl]][old]
            self.fitnesses[cl] += diff
            oldReal = self.fitnessReal[cl]
            self.fitnessReal[cl] = cNorm(self.fitnesses[cl])
            self.totFitness += self.fitnessReal[cl] - oldReal
        # update fitness of loc itself
        self.totFitness -= self.fitnessReal[loc]
        self.fitnesses[loc] = 0
        self.fitnessReal[loc] = 0
        for connLoc in self.adjGrid[loc]:
            if (connLoc == self.dim).all():
                continue
            cl = tuple(connLoc)
            self.fitnesses[loc] += self.payoffMatrix[new][self.grid[cl]]
        
        self.fitnessReal[loc] += cNorm(self.fitnesses[loc])
        self.totFitness += self.fitnessReal[loc]
            

    def getRandInd(self):
        """ Gets a random individual in the adjacency grid, with probability
            proportional to that edge's fitness """
        return np.random.choice(self.indList, p=self.fitnessRavel/self.totFitness)

        
    def evolve(self):
        """ The original evolve function.
            Works for grids of any dimension, but may be slower."""
        # simulation finished - one of the species has gone to fixation
        if (self.numMutants == 0) or (self.numMutants == self.totElements):
            return 1
        # individual that gets to reproduce
        ind = unFlatten(self.getRandInd(), self.dim)
        # individual we are going to replace
        if self.randomPlacement:
            replace = unFlatten(np.random.randint(0, len(self.indList)), self.dim)
        else:
            replace = self.adjGrid[ind + (np.random.randint(0, len(self.adjGrid[ind])),)]
            while (replace == self.dim).all():
                replace = self.adjGrid[ind + (np.random.randint(0, len(self.adjGrid[ind])),)]
        oldType = self.grid[tuple(replace)]
        r = np.random.random()
        if r <= self.fitnesses[ind].real / cNorm(self.fitnesses[ind]):
            self.grid[tuple(replace)] = self.grid[ind]
        else:
            self.grid[tuple(replace)] = 1 - self.grid[ind]
        self.update(replace, oldType, self.grid[tuple(replace)])
        return 0


""" Below are a variety of adjacency functions, which can be used
    to generate grids of various topologies for the grid. """
    
def stdAdjFunc(coord, dim):
    """ Returns all adjacent locations to a given position.
        Uses standard layout (with special edge/corner cases)"""
    ldim = len(dim)
    pos = coord[0:ldim]
    val = coord[ldim]
    # this adjacency function does not use this many adjacent locations
    if val >= 3 ** ldim - 1:
        return dim
    arr = dirFromNum(val, ldim)
    adj = np.add(arr, coord)

    for pos in adj:
        # position is not in grid; return "blank"
        if pos < 0 or pos >= ldim:
            return dim

    return adj
   
def torusAdjFunc(coord, dim):
    """ Returns all adjacent locations to a given position.
        Wraps around at edges and corners. """
    ldim = len(dim)
    pos = coord[0:ldim]
    # this adjacency function does not use this many adjacent locations
    val = coord[ldim]
    if val >= 3 ** ldim - 1:
        return dim
    arr = dirFromNum(val, ldim)
    adj = np.add(arr, pos)

    for i in range(ldim):
        if adj[i] < 0:
            adj[i] += dim[i]
        elif adj[i] >= dim[i]:
            adj[i] -= dim[i]

    return adj

def torusAdjFunc4(coord, dim):
    """ Same as torusAdjFunc but produces 4-regular instead of 8-regular lattice"""
    ldim = len(dim)
    pos = coord[0:ldim]
    # this adjacency function does not use this many adjacent locations
    val = coord[ldim]
    if val >= 2 * ldim:
        return dim
    arr = dirFromNum4(val, ldim)
    adj = np.add(arr, pos)

    for i in range(ldim):
        if adj[i] < 0:
            adj[i] += dim[i]
        elif adj[i] >= dim[i]:
            adj[i] -= dim[i]

    return adj
    
def randomizedAdjFunc(prevAdjFunc, dim, pos, currTuple, dist, jumpProb):
    """ Implements a randomized adjacency function.

        Implements the previous adjacency function, with a probability of
        the random jumps characteristic of small-world networks.
        Unlike smallWorldIfy, this does NOT generate bidirectional
        small-world networks, but instead introduces one-way "wormholes".
        Also, note that with this adjacency function, all outputs are of
        equal length - this means that every space has the same number of
        outgoing edges, but not necessarily the same number of incoming edges.
        The way the RNG works here is, for each edge there is a probability
        of it being replaced with a completely random edge.

        Note that "overrandom" networks, with more than 8 connections, can be
        created by using this function with a high jumpProb, and with extra 
        space (see initAdjGrid) """
    if np.random.random() < jumpProb:
        return np.array(getRandLoc(dim))
    else:
        return prevAdjFunc(dim, pos, currTuple, dist)
  
  
def dirFromNum(val, ldim):
    """ Returns the direction corresponding to an integer.

        Used to generate adjacency tables (i.e. the "space"). Operates using
        base 3, but excludes same point (0,0,...,0). Assumes the grid
        is Cartesian. Output will be a difference vector in the form of an
        array, which must be added to the current vector. Assumes val does
        not exceed maximum value (3^ldim-1). """
    maxVal = 3 ** ldim - 1
    if val >= maxVal / 2:
        val += 1
    arr = np.zeros(ldim, dtype=np.int32)
    # convert to base 3, for the conversion
    for i in range(ldim):
        arr[ldim - i - 1] = val % 3 - 1
        val = val // 3
    return arr  

def dirFromNum4(val, ldim):
    """ Returns the direction corresponding to an integer in a 4-lattice.

        Used to generate adjacency tables (i.e. the "space"). Operates using
        base 3, but excludes same point (0,0,...,0). Assumes the grid
        is Cartesian. Output will be a difference vector in the form of an
        array, which must be added to the current vector. Assumes val does
        not exceed maximum value (3^ldim-1). """
    # only going to implement this function in 2D case
    assert(ldim == 2)
    assert(val <= 3)
    if val == 0:
        return [1, 0]
    elif val == 1:
        return [0, 1]
    elif val == 2:
        return [-1, 0]
    elif val == 3:
        return [0, -1]
  

""" Below are a variety of useful operations on the grid. """





def initAdjGrid(adjFunc, dim, extraSpace):
    """ Initializes a grid from an adjacency function.
    
        Elements of the grid are arrays of coordinates (in array form) 
        of adjacent points, according to the adjacency function. The amount
        of extra connections that can be added is specified by the extraSpace
        parameter. """
    
    ldim = len(dim)
    buffer = (3 ** ldim - 1) * extraSpace
    adjGrid = np.zeros(tuple(dim) + (buffer, ldim), dtype=np.int32)
    # the array we iterate over, to get the multi_index of the iterator
    iter_arr = np.empty(tuple(dim) + (buffer,), dtype=np.int8)
    it = np.nditer(iter_arr, flags=['multi_index'])
    while not it.finished:
        adjGrid[it.multi_index] = adjFunc(np.array(it.multi_index), dim)
        it.iternext()
    return adjGrid

@autojit
def getHubs(numHubs, ldim, adjGridShape):
    hubs = np.empty((len(numHubs), ldim))
    
    filled = 0
    count = 0
    allVertices = np.empty(adjGridShape[0:ldim])
    it = np.nditer(allVertices, flags=['multi_index'])
    while not it.finished:
        if count in numHubs:
            hubs[filled] = np.array(it.multi_index)
            filled += 1
        count += 1
        it.iternext()

    if filled != len(hubs):
        print("WARNING: Incorrect number of hubs.")
    
    return hubs

def smallWorldIfyHeterogeneous(adjGrid, jumpProb, heterogeneity=0, replace=True):
    dim = adjGrid.shape[0:len(adjGrid.shape)-2]
    cpAdjGrid = np.copy(adjGrid)
    swh_notconnected(cpAdjGrid, jumpProb, heterogeneity, replace)
    while not isConnected(dim, cpAdjGrid):
        print("Grid not connected, trying again...")
        cpAdjGrid = np.copy(adjGrid)
        swh_notconnected(cpAdjGrid, jumpProb, heterogeneity, replace)
    return cpAdjGrid



#@autojit
def swh_notconnected(adjGrid, jumpProb, heterogeneity=0, replace=True):
    """ Turns the adjacency grid into a small-world network.
        This works as follows: for each edge, we rewire it into
        a random edge (with the same starting vertex) with a given
        probability, if replace. Otherwise, we simply add extra edges.
        The SWN will have tunable heterogeneity. Unlike other method,
        new edges are COMPLETELY random - they do not have same starting vertex.
        Assumes initial adjGrid is torus-like."""
    # we need to iterate over this to keep tuples intact
    ldim = len(adjGrid.shape) - 2
    dim = adjGrid.shape[0:ldim]
    iter_arr = np.empty(adjGrid.shape[0:ldim+1], dtype=np.int8)
    dim = np.array(dim)
    it = np.nditer(iter_arr, flags=['multi_index'])
    maxIndex = -1
    index = 0
    # calculate k, where the graph is assumed to be k-regular
    for nb in adjGrid[tuple([0 for i in range(ldim)])]:
        if (nb == dim).all():
            maxIndex = index
            break
        index += 1
    
    numVertices = np.prod(adjGrid.shape[0:ldim])
    hubs = getHubs(np.random.choice(numVertices, int((1 - heterogeneity) * numVertices), replace=False), ldim, adjGrid.shape)

    while not it.finished:
        # only consider left-facing edges (plus down) - that way we
        # count each edge exactly once (the other edges will be counted
        # when we iterate to the corresponding neighbor vertices).
        if it.multi_index[ldim] >= maxIndex / 2:
            it.iternext()
            continue

        # do not change this edge
        if np.random.random() > jumpProb:
            it.iternext()
            continue

        # the edge we are about to remove
        loc = it.multi_index[0:ldim]
        adjLoc = tuple(adjGrid[it.multi_index])

        
        # an edge already exists bewtween the two new locations (i.e. we need to resample)
        edgeExists = True
        # new, random locations for the new edge
        while edgeExists:
            # one of the locations must be a hub
            newLocPos = randrange(0, len(hubs))
            newLoc = tuple(hubs[newLocPos])
            newAdjLoc = getRandLoc(dim, newLoc)
            edgeExists = False
            # check if an edge already exists between these two vertices
            for i in range(len(adjGrid[newLoc])):
                if (adjGrid[newLoc + (i,)] == newAdjLoc).all():
                    edgeExists = True
        
        
        
        # add edge from newLoc to newAdjLoc
        # to do this we replace first available blank space
        added = False
        for i in range(len(adjGrid[newLoc])):
            if (adjGrid[newLoc + (i,)] == dim).all():
                adjGrid[newLoc + (i,)] = np.array(newAdjLoc)
                added = True
                break
        
        # add reverse edge from newAdjLoc to newLoc
        addedRev = False
        for i in range(len(adjGrid[newAdjLoc])):
            if (adjGrid[newAdjLoc + (i,)] == dim).all():
                adjGrid[newAdjLoc + (i,)] = np.array(newLoc)
                addedRev = True
                break
        
        # we never added edge: print warning message and continue
        # to next location
        if not added:
            print("WARNING: Failed to write edge from " + str(newLoc) + " to " +
                  str(tuple(newAdjLoc)) + ". Try adding more extra space.")
            it.iternext()
            continue

        if not addedRev:
            print("WARNING: Failed to write edge from " + str(newAdjLoc) + " to " +
                  str(tuple(newLoc)) + ". Try adding more extra space.")
            it.iternext()
            continue
           
        # remove original edges
        if replace:
            # delete original forwards edge from loc to adjLoc
            adjGrid[it.multi_index] = dim

            # remove backwards edge from adjLoc to loc
            for i in range(len(adjGrid[adjLoc])):
                if (adjGrid[adjLoc + (i,)] == np.array(loc)).all():
                    adjGrid[adjLoc + (i,)] = dim
                    break

        it.iternext()

def isConnected(dim, adjGrid):
    """ Checks if the given graph is connected. Assumes graph is undirected. """
    nodesVisited = []
    queue = Queue()
    queue.put((0,0))
    while not queue.empty():
        node = queue.get()
        if node not in nodesVisited:
            nodesVisited.append(node)
            for nb in adjGrid[node]:
                if (nb != dim).any():
                    queue.put(tuple(nb))
    # get size of graph
    size = 1
    for d in dim:
        size *= d
    return len(nodesVisited) == size

@autojit
def smallWorldIfy(adjGrid, jumpProb):
    """ Turns the adjacency grid into a small-world network.
        This works as follows: for each edge, we rewire it into
        a random edge (with the same starting vertex) with a given
        probability. Assumes initial adjGrid is torus-like."""
    # we need to iterate over this to keep tuples intact
    ldim = len(adjGrid.shape) - 2
    dim = adjGrid.shape[0:ldim]
    iter_arr = np.empty(adjGrid.shape[0:ldim+1], dtype=np.int8)
    dim = np.array(dim)
    it = np.nditer(iter_arr, flags=['multi_index'])
    maxIndex = 3 ** ldim - 1

    while not it.finished:
        # only consider left-facing edges (plus down) - that way we
        # count each edge exactly once (the other edges will be counted
        # when we iterate to the corresponding neighbor vertices).
        if it.multi_index[ldim] >= maxIndex / 2:
            it.iternext()
            continue

        # do not change this edge
        if np.random.random() > jumpProb:
            it.iternext()
            continue

        # the location in question, and the adjacent location
        loc = it.multi_index[0:ldim]
        adjLoc = tuple(adjGrid[it.multi_index])

        # new, random location that the edge will connect to
        newLoc = getRandLoc(dim, loc)
        
        # add backwards edge from newLoc to loc
        # to do this we replace first available blank space
        added = False
        for i in range(len(adjGrid[newLoc])):
            if (adjGrid[newLoc + (i,)] == dim).all():
                adjGrid[newLoc + (i,)] = np.array(loc)
                added = True
                break
        
        # we never added edge: print warning message and continue
        # to next location
        if not added:
            print("WARNING: Failed to write edge from " + str(loc) + " to " +
                  str(tuple(newLoc)) + ". Try adding more extra space.")
            it.iternext()
            continue


        # replace forwards edge with random edge (not to same vertex)
        adjGrid[it.multi_index] = np.array(newLoc)

        # remove backwards edge from adjLoc to loc
        for i in range(len(adjGrid[adjLoc])):
            if (adjGrid[adjLoc + (i,)] == np.array(loc)).all():
                adjGrid[adjLoc + (i,)] = dim
                break

        it.iternext()


    
def getRandLoc(dim, loc=None):
    """ Generates a random location in the grid, that isn't loc. """
    newLoc = tuple(np.random.randint(0, dim[i]) for i in range(len(dim)))
    while newLoc == loc:
        newLoc = tuple(np.random.randint(0, dim[i]) for i in range(len(dim)))
    return newLoc

def genRandGrid(dim, prob=0.5):
    """ Generates a random grid with a given cell density. """
    grid = np.random.random(tuple(dim))
    alive = grid < prob
    intGrid = np.zeros(tuple(dim + 1), dtype=np.int8) # make an integer grid
    intGrid[alive] = 1
    return intGrid

def genRandGridNum(dim, num=0):
    numLeft = num
    grid = np.zeros(tuple(dim+1), dtype=np.int8)
    while numLeft > 0:
        loc = getRandLoc(dim)
        while grid[loc] == 1:
            loc = getRandLoc(dim)
        grid[loc] = 1
        numLeft -= 1
    return grid

def gridToStr2D(grid):
    """ Returns a string representation of grid, ignoring first row and column. OUT OF DATE """
    dim = grid.shape
    s = ""
    for i in range(1,dim[0]):
        s += "["
        for j in range(1,dim[1]):
            s += str(grid[i,j])
            if j is not dim[1] - 1:
                s += ", "
        s += "]\n"
    return s

    
def addToTuple(tp, num):
    l = len(tp)
    newTp = np.array(tp)
    for i in range(l):
        newTp[i] += num
    return tuple(newTp)


    
    
""" Evolution methods. These are placed outside the class for clarity, and to
    enable easier compiling or parallelization."""


   
@autojit
def evolve2D(rows, cols, grid, adjGrid, newGrid):
    """ Like evolve, but only compatible with 2D arrays. Uses loops rather than
        iterators, so hopefully easier to parallelize. Assumes grid and adjGrid
        are what they should be for dim = [rows, cols] (AND ARE CONFIGURED.)"""
    maxLen = len(adjGrid[0,0])
    for i in range(rows):
        for j in range(cols):
            numAlive = 0
            for k in range(maxLen):
                numAlive += grid[adjGrid[i,j,k,0], adjGrid[i,j,k,1]]

            if numAlive == 3 or (numAlive == 2 and grid[i,j] == 1):
                newGrid[i,j] = 1


# @cuda.jit(argtypes=[uint8[:,:], uint32[:,:,:,:], uint8[:,:]])
# def evolve2D_kernel(grid, adjGrid, newGrid):
#     """ Like evolve, but only compatible with 2D arrays. Uses loops rather than
#         iterators, so hopefully easier to parallelize. Assumes grid and adjGrid
#         are what they should be for dim = dimArr[0:1] (AND ARE CONFIGURED.)
#         dimArr is [rows, cols, maxLen] """
#     rows = grid.shape[0] - 1
#     maxLen = adjGrid.shape[2]
#     cols = grid.shape[1] - 1
#     startX, startY = cuda.grid(2)
#     gridX = cuda.gridDim.x * cuda.blockDim.x
#     gridY = cuda.gridDim.y * cuda.blockDim.y
#     for i in range(startX, rows, gridX):
#         for j in range(startY, cols, gridY):
#             numAlive = 0
#             for k in range(maxLen):
#                 # if adjGrid is configured, a placeholder value of dim
#                 # will result in a 0 being looked up (as desired)
#                 numAlive += grid[adjGrid[i,j,k,0], adjGrid[i,j,k,1]]
#             if numAlive == 3 or (numAlive == 2 and grid[i,j] == 1):
#                 newGrid[i,j] = 1
    
class Game:
    """ Initializes the game of life.
        The grid will be a numpy array of int8s, i.e. the alive/dead state,
        such that the last row and column are all 0s, and will always remain 0
        (this is so direct array indexing is possible, without having
        conditional statements). The specified dimension must be the dimension
        of the "real" grid, i.e. not including that last row and column.
        The adjacency function can be used to specify the geometry of the
        grid. """
    def __init__(self, grid=None, dim=np.array([10,10]),
                 adjFunc=stdAdjFunc, extraSpace=1):
        if grid is None:
            self.grid = genRandGrid(dim)
        else:
            self.grid = grid
        self.dim = dim
        start = timer()
        self.adjGrid = initAdjGrid(adjFunc, self.dim, extraSpace)
        dt = timer() - start
        print("Time to generate adjGrid: %f" % dt)

    def evolve2D_self(self):
        newGrid = np.zeros_like(self.grid)
        evolve2D(self.dim[0], self.dim[1], self.grid, self.adjGrid, newGrid)
        self.grid = newGrid
    
    def smallWorldIfy(self, jumpFrac):
        """ Turns the adjacency grid into a small-world network.
            The number of random jumps inserted is a proportion of the total
            number of distinct grid values. Connections are removed."""
        prod = 1
        for i in range(len(self.dim)):
            prod *= self.dim[i]
        
        for _ in range(floor(prod * jumpFrac)):
            # get the location we're about to switch
            loc = getRandLoc(self.dim)
            # get all adjacent locations
            adj = self.adjGrid[loc]
            # if we don't have any neighbors, abort since we can't switch
            if len(adj) == 0:
                continue
            # get new location that we're going to make adjacent to loc
            newLoc = getRandLoc(self.dim, loc)
            # if they're already neighbors, or equal, abort operation
            if (loc in self.adjGrid[newLoc]) or (newLoc in self.adjGrid[loc])\
                or loc == newLoc:
                continue
            # this is the location we're going to swap
            changePos = np.random.randint(0, len(adj))
            # remove the other edge to loc
            adjToChangeLoc = self.adjGrid[adj[changePos]]
            if loc in adjToChangeLoc:
                adjToChangeLoc.remove(loc)
            # switch edge in loc
            adj[changePos] = newLoc
            # now add the reverse edge
            self.adjGrid[adj[changePos]].append(loc)
            

    def smallWorldIfy_noremove(self, jumpFrac):
        """ Turns the adjacency grid into a small-world network.
            The number of random jumps inserted is a proportion of the total
            number of distinct grid values. Note that no connections are
            removed, so using this method increases overall connectivity of the
            grid (in slight deviation with Strogatz & Watts's model)."""
        prod = 1
        for i in range(len(self.dim)):
            prod *= self.dim[i]
        
        for _ in range(floor(prod * jumpFrac)):
            # get the location we're about to switch
            loc = getRandLoc(self.dim)
            # append a random location to our adjacent locations, and vice
            # versa
            randLoc = getRandLoc(self.dim, loc)
            self.adjGrid[loc].append(randLoc)
            self.adjGrid[randLoc].append(loc)
            
    def __str__(self):
        return str(self.grid)