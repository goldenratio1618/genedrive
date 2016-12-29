import numpy as np
from gameoflife import addToTuple
from numba import *

def countLiveCells(grid):
    """ Returns the number of live cells in the grid.
        
        If the cells have multiple lives, returns the total number of lives. """
    count = 0
    for val in np.nditer(grid):
        count += val
    return count

@autojit
def cluster(grid, adjGrid):
    """ Returns the probability that neighbors of live cells are live
   
        This can be used as a measure of "randomness" of the grid -- i.e. if the
        grid is completely random, then this should roughly equal 1 - 2f + 2f^2,
        where f is the fraction of live cells. """
    if countLiveCells(grid) == 0:
        return 0 # nothing is alive; cluster is 0
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
    