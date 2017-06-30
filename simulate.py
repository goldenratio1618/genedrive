from copy import deepcopy
from math import floor
import cmath
from numba import *
import numpy as np
from timeit import default_timer as timer
from random import randrange
from gridtools import *
from time import sleep

def cNorm(a):
    return a.real + a.imag

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
        """ Code for initializing fitness values for each location in grid."""
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
            # make sure rounding errors don't create negative probabilities
            if abs(self.fitnesses[cl]) < 1E-15:
                self.fitnesses[cl] = 0
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
        """ Runs one step of the FDSCP process and updates the state of the system """
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

def run(fdscp, steps, delay, initDelay, printInd, indSteps, unitTest=False):
    """ Runs the FDSCP process for a specified amount of time, or until one species successfully fixates """
    step = 0
    while step < steps or steps == -1:
        # print grid
        if printInd is not -1 and step % printInd is 0:
            printGrid(fdscp.grid, step, fdscp.dim)
        # print index
        if indSteps is not -1 and step % indSteps is 0:
            print("Step = " + str(step) + ", mutants = " + str(fdscp.numMutants))
            if unitTest:
                assert((fdscp.initFitnesses() == fdscp.fitnesses).all())
                assert((fdscp.fitnessReal == np.array(list(map(cNorm,fdscp.fitnesses)))).all())
                assert((fdscp.totFitness == np.sum(fdscp.fitnessReal)))
        # we are at fixation
        if fdscp.evolve():
            break
        if delay > 0:
            sleep(delay)
        if step == 0:
            # allow initial position to be more easily visible
            sleep(initDelay)
        step += 1
    return step

def calcFixProb(args, fdscp, dim, start, folder, datestr, ind=0):
    """ Calculates the fixation probability for the given arguments """
    # these will be arrays of the values for every simulation
    mutants = np.zeros(args.niters)
    stepsToFix = np.zeros(args.niters)
    # run the simulation on this many different, random grids
    # (for symmetric grids, randomness will not matter)
    for sim in range(args.niters):
        if args.debug:
            print("Running simulation: " + str(sim))
            print("Sim = " + str(sim) + ". Time elapsed: " + str(timer() - start))
        # make file to output live cell count and cluster every step
        if args.output >= 3:
            outfile_steps = open(args.outfile + folder + "data3/" + "sim=" + \
            str(sim) + datestr + ".txt", "w")
            outfile_steps.writelines("Step  mutants Cluster\n")
        # reset grid to fresh state
        if args.initMutant == -1:
            grid = genRandGrid(dim, prob=args.frac)
        else:
            grid = genRandGridNum(dim, num=args.initMutant)
        fdscp.grid = grid
        fdscp.init()
        if args.debug:
            print("Grid reset. Time elapsed: " + str(timer() - start))
        steps = args.simlength
        if args.output < 3:
            if args.debug:
                stepsToFix[sim] = run(fdscp, steps, args.delay, 0, args.visible, 1000, args.test)
            else:
                stepsToFix[sim] = run(fdscp, steps, args.delay, 0, args.visible, -1, args.test)
        else:
            # TODO: get this stuff to work...
            step = 0
            while steps == -1 or step < steps/args.sample:
                if args.debug:
                    print("Step = " + str(step) + " Time elapsed: " + str(timer() - start))
                # output data to file
                if args.debug:
                    stepsToFix[sim] = run(fdscp, args.sample, args.delay, 0, args.visible, 1000, args.test)
                else:
                    stepsToFix[sim] = run(fdscp, args.sample, args.delay, 0, args.visible, -1, args.test)

                outfile_steps.writelines(str(step * args.sample) + "    " + str(countLiveCells(grid)) + "\n")
                # step once
                finished = run(fdscp, args.sample, args.delay, 0, args.visible, -1, args.test)
                # make file, and output grid to that file
                if args.output >= 4:
                    outfile_grids = open(args.outfile + folder + "data4/" + "ind=" + str(ind) + "sim=" + str(sim) + \
                    "_step=" + str(step * args.sample) + datestr + ".txt", "w")
                    printGrid(grid, -1, grid.shape, outfile_grids)
                    outfile_grids.close()
                step += 1

            outfile_steps.close()

        if args.debug:
            print("Simulation finished. Time elapsed: " + str(timer() - start))

        mutants[sim] = countLiveCells(grid)
        
        if args.debug:
            print("Finished running simulation. Time elapsed: " + str(timer() - start))
    
    zero = np.zeros(len(mutants))
    fixationEvents = mutants > 0
    fixTime = 0
    if len(stepsToFix[fixationEvents]) > 0:
        fixTime = np.sum(stepsToFix[fixationEvents]) / len(stepsToFix[fixationEvents])

    # NOTE: if simulation never ends until one of the species fixates, the second entry will equal the fixation probability
    return (mutants, np.sum(fixationEvents) / len(fixationEvents), stepsToFix, fixTime)

def binarySearch(args, fdscp, dim, start, folder, datestr):
    """ Starts a binary search for 'critical' value F_{B,cr} with P_F(F_{B,cr}) = 1/n.
        Assumes parameter value F_B is given in args, and uses input parameter a. """
    ind = 0
    currMin = 0
    currMax = args.wtfitness
    guess = -1
    data = {}
    for i in range(args.depth):
        guess = (currMax + currMin)/2
        # replace wild type fitness with our guessed fitness
        fdscp.payoffMatrix = np.array([[guess ** 2, guess*1j],[guess, fdscp.payoffMatrix[1,1]]], dtype = np.complex128)
        data[guess] = calcFixProb(args, fdscp, dim, start, folder, datestr, ind)
        if data[guess][1] == 1/fdscp.totElements:
            return (guess, data)
        elif data[guess][1] < 1/fdscp.totElements:
            # FP too small, so wild-type fitness is too high
            currMax = guess
        else:
            # too high of a FP, so wtfitness is too low
            currMin = guess
    guess = (currMax + currMin)/2
    return (guess, data)
