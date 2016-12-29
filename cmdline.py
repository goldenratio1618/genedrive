import os
import sys
import numpy as np
from gameoflife import evolve2D, evolve2D_kernel
from time import sleep
from numba import *
from numbapro import cuda

def run_GPU(grid, adjGrid, steps, delay, initDelay, printInd, indSteps):
    """ Runs the Command-Line interface for a specified number of steps,
        or forever if the number of steps is specified to be -1.
        Note that here, grid and adjGrid must be explicitly specified as
        opposed to passed in as a Game, to enable everything to be run on the
        GPU. Returns the final grid state. """
    step = 0
    dim = grid.shape
    # move arrays to GPU
    d_grid = cuda.to_device(grid)
    d_adjGrid = cuda.to_device(adjGrid)
    blockDim = (32,16)
    gridDim = (32,8)
    while step < steps or steps == -1:
        # print grid
        if printInd is not -1 and step % printInd is 0:
            # in order to print grid, first need memory back in CPU
            d_grid.to_host()
            printGrid(grid, step, dim)
        # print index
        if indSteps is not -1 and step % indSteps is 0:
            print("Step = " + str(step))
        newGrid = np.zeros_like(grid)
        d_newGrid = cuda.to_device(newGrid)
        evolve2D_kernel[gridDim, blockDim](d_grid, d_adjGrid, d_newGrid)
        d_grid = d_newGrid
        grid = newGrid
        sleep(delay)
        if step == 0:
            # allow initial position to be more easily visible
            sleep(initDelay)
        step += 1
    d_grid.to_host()
    return grid

    
def run(game, steps, delay, initDelay, printInd, indSteps):
    """ Runs the Command-Line interface for a specified number of steps,
        or forever if the number of steps is specified to be -1."""
    step = 0
    while step < steps or steps == -1:
        # print grid
        if printInd is not -1 and step % printInd is 0:
            printGrid(game.grid, step, game.dim)
        # print index
        if indSteps is not -1 and step % indSteps is 0:
            print("Step = " + str(step))
        game.evolve2D_self()
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
    grid_str += horizontalLine(dim[1]-1)
    for i in range(dim[0]-1):
        grid_str += "|"
        for j in range(dim[1]-1):
            if grid[i][j]:
                grid_str += "X"
            else:
                grid_str += " "
        grid_str += "|\n"
    # cross-platform way to clear command prompt, for the next round of
    # printing the grid
    grid_str += horizontalLine(dim[1]-1)
    if file is None:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(grid_str)
        sys.stdout.flush()
    else:
        file.writelines(grid_str)
