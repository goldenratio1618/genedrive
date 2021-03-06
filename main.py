﻿from simulate import *
from cmdline import *
from gridtools import cluster, countLiveCells
from sys import stdout
from copy import deepcopy
from timeit import default_timer as timer
import numpy as np
import argparse
import datetime
import os
from matplotlib import pyplot as plt

def genPayoffMatrix(a, fb, p):
    """ Creates a payoff matrix corresponding to the given parameters """
    return np.array([[fb ** 2, fb*(p * 1j + (1-p))],[fb*(p + (1-p)*1j), a]], dtype = np.complex128)

def main(args):
    """ Initializes and runs a simulation with the given command-line arguments """
    if args.output > 2:
        raise ValueError("Illegal value of output argument.")
    if args.plot == 'fb' and args.min == 0:
        raise ValueError("Minimum value must be nonzero when plotting with respect to fb")
    if args.hzfitness < 0 or args.wtfitness < 0:
        raise ValueError("Fitnesses must be nonnegative")
    start = datetime.datetime.now()
    # all new datafiles will be stored in this folder
    folder = start.strftime("%m-%d-%Y_%H-%M-%S") + "/"
    # this folder should be unique, so no need to check if dir exists
    os.mkdir(args.outfile + folder)
    for i in range(1,args.output+1):
        os.mkdir(args.outfile + folder + "/data" + str(i) + "/")
    # add extra parameters
    if args.graphFile == "":
        if args.fourRegular:
            datestr = "4regular_" + "rows=" + str(args.rows) + "_cols=" + \
                str(args.cols) + "_a=" + str(args.hzfitness) + "_fb=" + str(args.wtfitness) + \
                "_niters=" + str(args.niters)
        else:
            datestr = "8regular_" + "rows=" + str(args.rows) + "_cols=" + \
                str(args.cols) + "_a=" + str(args.hzfitness) + "_fb=" + str(args.wtfitness) + \
                "_niters=" + str(args.niters)
    else:
        datestr = "graphFile=" + args.graphFile.split('/')[-1].split('.')[0] + \
                  "_a=" + str(args.hzfitness) + "_fb=" + str(args.wtfitness) + \
                  "_niters=" + str(args.niters)
    if args.randomPlacement:
        datestr += "_randomplacement"
    if args.plot != "":
        datestr += "_plotting_" + args.plot + "_min=" + str(args.min)
    if args.binsearch != "":
        datestr += "_binsearch_fb"
        if args.depth != 10:
            datestr += "_depth=" + str(args.depth)
    if args.simlength != -1:
        datestr += "_simlength=" + str(args.simlength)
    if args.swc != 0:
        datestr += "_swc=" + str(args.swc)
        if args.heterogeneity != 0:
            datestr += "_heterogeneity=" + str(args.heterogeneity)
        if args.keep:
            datestr += "_keep=True"

    adjGrid = None
    dim = None
    dualAdjGrid = None
    # load the graph from a file
    if args.graphFile != '':
        graph = open(args.graphFile, 'r').readlines()
        dim,adjGrid,dualAdjGrid = parseGraph(graph)
        # dim must be a numpy array for other functions to work properly
        dim = np.array([dim])

    np.set_printoptions(threshold=np.inf)

    if args.output >= 1:
        # this file stores averages of final values across all simulations per swc
        outfile_avg = open(args.outfile + folder + "data1/" + datestr + ".txt", "w")

    start = timer()
    if dim is None:
        dim = np.array([args.rows,args.cols])

    payoffMatrix = genPayoffMatrix(args.hzfitness, args.wtfitness, args.genedriveprob)
    if adjGrid is None:
        if args.fourRegular:
            adjGrid = initAdjGrid(torusAdjFunc4, dim, args.extraspace)
        else:
            adjGrid = initAdjGrid(torusAdjFunc, dim, args.extraspace)
    # original torus adjacency grid, to be used as fresh template for small-world
    origAdjGrid = np.copy(adjGrid)

    if args.debug and args.graphFile != '':
        print("adjGrid = " + str(adjGrid))
        print("dualAdjGrid = " + str(dualAdjGrid))
        
    max_val = -1
    if args.plot == 'a': max_val = args.hzfitness
    elif args.plot == 'fb': max_val = args.wtfitness
    elif args.plot == 'p': max_val = args.genedriveprob
    elif args.plot == 's': max_val = args.swc
    elif args.plot == 'h': max_val = args.heterogeneity

    # if we aren't plotting, we can simply compute fixation probability at the given point
    var_values = [0]
    # if we are plotting, get the range we will be plotting over
    if max_val != -1:
        var_values = np.linspace(args.min, max_val, args.numpoints)

    if args.plot == 'fb' and args.binsearch:
        raise ValueError("Cannot run binary search unless fb is fixed.")

    fixProbs = []
    fixTimes = []

    # critical w values found by binary search
    wcVals = []

    for var in var_values:
        if args.initMutant == -1:
            grid = genRandGrid(dim, prob=args.frac)
        else:
            grid = genRandGridNum(dim, num=args.initMutant)

        # we need to change the adjacency grid if we are changing small world properties
        if args.plot in ['s', 'h']:
            adjGrid = np.copy(origAdjGrid)
        strParam = str(round(var, 6))
        if args.debug:
            print(args.plot + " = " + strParam + ". Time elapsed: " + str(timer() - start))

        # make sure we incorporate the relevant parameter value into the calculation
        if args.plot == 's':
            adjGrid = smallWorldIfyHeterogeneous(adjGrid, var, args.heterogeneity, not args.keep)
        elif args.plot == 'h':
            adjGrid = smallWorldIfyHeterogeneous(adjGrid, args.swc, var, not args.keep)
        elif args.swc > 0:
            adjGrid = smallWorldIfyHeterogeneous(adjGrid, args.swc, args.heterogeneity, not args.keep)

        # make dual adj grid for SWN case
        if args.plot == 'a':
            payoffMatrix = genPayoffMatrix(var, args.wtfitness, args.genedriveprob)
        elif args.plot == 'fb':
            payoffMatrix = genPayoffMatrix(args.hzfitness, var, args.genedriveprob)
        elif args.plot == 'p':
            payoffMatrix = genPayoffMatrix(args.hzfitness, args.wtfitness, var)
            

        if args.debug:
            print("payoff matrix (mutant is second row & column):")
            print(payoffMatrix)
            print("Initialized simulation. Time elapsed: " + str(timer() - start))

        if args.debug and (args.plot in ['s', 'h'] or args.swc > 0):
            print("Grid smallworldified. Time elapsed: " + str(timer() - start))
            print(adjGrid)

        # now that the adjGrid is set up, we can proceed to compute the fixation probability
        fdscp = FDSCP(dim, payoffMatrix, adjGrid, grid, dualAdjGrid, args.randomPlacement)

        if args.binsearch:
            wc, data = binarySearch(args, fdscp, dim, start, folder, datestr)
            wcVals.append(wc)
        else:
            mutants, fixProb, stepsToFix, timeToFix = calcFixProb(args, fdscp, dim, start, folder, datestr)
            fixProbs.append(fixProb)
            fixTimes.append(timeToFix)
        
        plotStr = ""
        if args.plot != '':
            plotStr = args.plot + " = " + strParam
        if args.binsearch:
            if args.plot != '':
                plotStr += ', '
            for guess in data:
                if args.output >= 1:
                    outfile_avg.writelines(plotStr + " fb = " + str(guess) + ": fixation prob = " + str(data[guess][1]) + ", fixation time = " + str(data[guess][3]) + "\n")
                print(plotStr + "fb = " + str(guess) + ": fixation prob = " + str(data[guess][1]) + ", fixation time = " + str(data[guess][3]))
            if args.output >= 1:
                outfile_avg.writelines("Critical fb value = " + str(wc) + "\n")
            print("Critical fb value = " + str(wc))
        else:
            if args.plot != '':
                plotStr += ': '
            if args.output >= 1:
                outfile_avg.writelines(plotStr + "fixation prob = " + str(fixProb) + ", fixation time = " + str(timeToFix) + "\n")
            print(plotStr + "fixation prob = " + str(fixProb) + ", fixation time = " + str(timeToFix))
        if args.plot != '':
            plotStr = args.plot + " = " + strParam + '_'
        # make an output file of the range of different final mutant values in simulations (as opposed to simply the fixation probability)
        if args.output >= 2:
            if args.binsearch:
                for guess in data:
                    outfile_final = open(args.outfile + folder + "data2/" + plotStr + "fb=" + str(guess) + datestr + ".txt", "w")
                    outfile_final.writelines("Run  mutants  time\n")
                    for i in range(len(data[guess][0])):
                        outfile_final.writelines(str(i) + "    " + str(data[guess][0][i]) + "     " + str(data[guess][2][i]) + "\n")
                    outfile_final.close()
            else:
                outfile_final = open(args.outfile + folder + "data2/" + plotStr + datestr + ".txt", "w")
                outfile_final.writelines("Run  mutants  time\n")
                for i in range(len(mutants)):
                    outfile_final.writelines(str(i) + "    " + str(mutants[i]) + "     " + str(stepsToFix[i]) + "\n")
                outfile_final.close()

        if args.debug:
            print("Finished outputting everything to files. Time elapsed: " + str(timer() - start))

    if args.output >= 1:
        outfile_avg.close()

    if args.plot != '':
        if args.binsearch:
            plt.plot(var_values, wcVals)
            plt.xlabel(args.plot)
            plt.ylabel('Critical fb value')
            plt.show()
        else:
            plt.plot(var_values, fixProbs)
            plt.xlabel(args.plot)
            plt.ylabel('Fixation probability')
            plt.show()

if __name__ == '__main__':
    args = getArgs()
    main(args)
