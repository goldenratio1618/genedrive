import os
import sys
import numpy as np
from time import sleep
from numba import *
from simulate import cNorm

def run(fdscp, steps, delay, initDelay, printInd, indSteps, debug=False):
    """ Runs the Command-Line interface for a specified number of steps,
        or forever if the number of steps is specified to be -1."""
    step = 0
    while step < steps or steps == -1:
        # print grid
        if printInd is not -1 and step % printInd is 0:
            printGrid(fdscp.grid, step, fdscp.dim)
        # print index
        if indSteps is not -1 and step % indSteps is 0:
            print("Step = " + str(step) + ", mutants = " + str(fdscp.numMutants))
            if debug:
                assert((fdscp.initFitnesses() == fdscp.fitnesses).all())
                assert((fdscp.fitnessReal == np.array(list(map(cNorm,fdscp.fitnesses)))).all())
        # we are at fixation
        if fdscp.evolve():
            break
        if delay > 0:
            sleep(delay)
        if step == 0:
            # allow initial position to be more easily visible
            sleep(initDelay)
        step += 1
            
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

