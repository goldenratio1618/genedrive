import numpy as np
import os, sys
from numba import *
from queue import Queue

def countLiveCells(grid):
    """ Returns the number of mutant cells in the grid """
    count = 0
    for val in np.nditer(grid):
        count += val
    return count

@autojit
def cluster(grid, adjGrid):
    """ Returns the probability that neighbors of mutant cells are mutants
   
        This can be used as a measure of "randomness" of the grid -- i.e. if the
        grid is completely random, then this should roughly equal 1 - 2f + 2f^2,
        where f is the fraction of live cells. """
    if countLiveCells(grid) == 0:
        return -1 # nothing is alive; return as such
    it = np.nditer(grid, flags=['multi_index'])
    matches = 0 # number of neighbors of live cells that are live
    total = 0 # total number of neighbors of live cells
    while not it.finished:
        flag = False

        for i in range(len(it.multi_index)):
            # do not record situations where we're in dead zone of grid
            if it.multi_index[i] == grid.shape[i] - 1:
                flag = True

        if flag:
            it.iternext()
            continue
        
        for adj in adjGrid[it.multi_index]:
            # do not record situations where adjacent cell is in dead zone
            flag = False

            for i in range(len(it.multi_index)):
                # do not record situations where we're in dead zone of grid
                if adj[i] == grid.shape[i] - 1:
                    flag = True

            if flag:
                continue

            loc = tuple(adj)
            # we only care about live cell matches
            if grid[it.multi_index] == 1:
                total += 1
                if grid[loc] == 1:
                    matches += 1
        it.iternext()
    return matches/total
    
# Below are a variety of adjacency functions, which can be used
# to generate grids of various topologies for the grid.
    
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
  

# Below are a variety of useful operations on the grid.





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
    """ Gets hubs for heterogeneous Newman-Watts small-world network """
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
    """ Creates Newman-Watts small-world network """
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


def horizontalLine(dim):
    """Draws a horizontal line, with two vertical bars at either end."""
    line = "|"
    for _ in range(dim):
        line += "_"
    line += "|\n"
    return line

def printGrid(grid, step, dim, file=None):
    """ Prints the grid """
    grid_str = ""
    if step is not -1:
        grid_str += "STEP: " + str(step) + "\n"
    grid_str += horizontalLine(dim[1])
    for i in range(dim[0]):
        grid_str += "|"
        for j in range(dim[1]):
            if grid[i][j]:
                grid_str += "X"
            else:
                grid_str += " "
        grid_str += "|\n"
    # cross-platform way to clear terminal, for the next round of
    # printing the grid
    grid_str += horizontalLine(dim[1])
    if file is None:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(grid_str)
        sys.stdout.flush()
    else:
        file.writelines(grid_str)

def parseGraph(graph):
    adjGridDict = {}
    dualAdjGridDict = {}
    for line in graph:
        start,end=line.split(",")
        if int(start) in adjGridDict.keys():
            adjGridDict[int(start)].append(int(end))
        else:
            adjGridDict[int(start)] = [int(end)]
        
        if int(start) in dualAdjGridDict.keys():
            dualAdjGridDict[int(start)].append(int(end))
        else:
            dualAdjGridDict[int(start)] = [int(end)]
    
    # add 1 to be consistent with other grid layout, which has dimension dim + 1
    dim = max(max(adjGridDict.keys()), max([max(i) for i in adjGridDict.values()])) + 1
    adjGrid = np.full((dim, max([len(a) for a in adjGridDict.values()]), 1), dim, dtype=np.int32)
    dualAdjGrid = np.full((dim, max([len(a) for a in dualAdjGridDict.values()]), 1), dim, dtype=np.int32)
    it = np.nditer(adjGrid, flags=['multi_index'])
    while not it.finished:
        if it.multi_index[0] in adjGridDict.keys() and it.multi_index[1] < len(adjGridDict[it.multi_index[0]]):
            adjGrid[it.multi_index] = adjGridDict[it.multi_index[0]][it.multi_index[1]]
        it.iternext()
    
    it = np.nditer(dualAdjGrid, flags=['multi_index'])
    while not it.finished:
        if it.multi_index[0] in dualAdjGridDict.keys() and it.multi_index[1] < len(dualAdjGridDict[it.multi_index[0]]):
            dualAdjGrid[it.multi_index] = dualAdjGridDict[it.multi_index[0]][it.multi_index[1]]
        it.iternext()
    return dim,adjGrid,dualAdjGrid
