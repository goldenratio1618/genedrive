from simulate import *
from cmdline import *
from gui import GUI
from gridtools import cluster, countLiveCells
from sys import stdout
from copy import deepcopy
from timeit import default_timer as timer
import numpy as np
import argparse
import datetime
import os
from matplotlib import pyplot as plt

def calcFixProb(args, fdscp, dim, start, folder, datestr):
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
            outfile_steps = open(args.outfile + folder + "data3/" + param + "=" + strParam +\
                "_sim=" + str(sim) + datestr + ".txt", "w")
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
            for step in range(steps/args.sample + 1):
                if args.debug:
                    print("Step = " + str(step) + " Time elapsed: " + str(timer() - start))
                # output data to file
                outfile_steps.writelines(str(step * args.sample) + "    " + str(countLiveCells(grid)) + "    " + str(cluster(grid, fdscp.adjGrid)) + "\n")
                # step once
                run(fdscp, args.sample, args.delay, 0, args.visible, -1)
                # make file, and output grid to that file
                if args.output >= 4:
                    outfile_grids = open(args.outfile + folder + "data4/" + "swc=" + strswc + "_sim=" + str(sim) + "_step=" + str(step * args.sample) + datestr + ".txt", "w")
                    printGrid(grid, -1, grid.shape, outfile_grids)
                    outfile_grids.close()

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
    """ Starts a binary search for 'critical' value w_c with FP(w_c) = 1/n.
        Assumes parameter value w is given in args, and uses input parameter p. """
    currMin = args.min
    currMax = args.wtfitness
    guess = -1
    data = {}
    for i in range(args.depth):
        guess = (currMax + currMin)/2
        # replace wild type fitness with our guessed fitness
        fdscp.payoffMatrix = np.array([[guess ** 2, guess*1j],[guess, fdscp.payoffMatrix[1,1]]], dtype = np.complex128)
        data[guess] = calcFixProb(args, fdscp, dim, start, folder, datestr)
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

def main(args):
    start = datetime.datetime.now()
    # all new datafiles will be stored in this folder
    folder = start.strftime("%m-%d-%Y_%H-%M-%S") + "/"
    # this folder is "guaranteed" to be unique, so no need to check if dir exists
    os.mkdir(args.outfile + folder)
    for i in range(1,args.output+1):
        os.mkdir(args.outfile + folder + "/data" + str(i) + "/")
    # add extra parameters
    datestr = "frac=" + str(args.frac) + "_rows=" + str(args.rows) + "_cols=" + \
        str(args.cols) + "_p=" + str(args.probsurvival) + "_w=" + str(args.wtfitness) + \
         "_niters=" + str(args.niters) + "_simlength=" + str(args.simlength) + \
        "_heterogeneity=" + str(args.heterogeneity)

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
    
    payoffMatrix = np.array([[args.wtfitness ** 2, args.wtfitness*1j],[args.wtfitness, args.probsurvival]], dtype = np.complex128)
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
    if args.plot == 'p': max_val = args.probsurvival
    elif args.plot == 'w': max_val = args.wtfitness
    elif args.plot == 's': max_val = args.swc
    elif args.plot == 'h': max_val = args.heterogeneity

    # if we aren't plotting, we can simply compute fixation probability at the given point
    var_values = [args.probsurvival]
    # if we are plotting, get the range we will be plotting over
    if max_val != -1:
        var_values = np.linspace(args.min, max_val, args.numpoints)

    if args.plot == 'w' and args.binsearch:
        raise ValueError("Cannot run binary search unless w is fixed.")

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
            adjGrid = smallWorldIfyHeterogeneous(adjGrid, var, args.heterogeneity, args.replace)
        elif args.plot == 'h':
            adjGrid = smallWorldIfyHeterogeneous(adjGrid, args.swc, var, args.replace)
        elif args.swc > 0:
            adjGrid = smallWorldIfyHeterogeneous(adjGrid, args.swc, args.heterogeneity, args.replace)

        # make dual adj grid for SWN case
        if args.plot == 'p':
            payoffMatrix = np.array([[args.wtfitness ** 2, args.wtfitness*1j],[args.wtfitness, var]], dtype = np.complex128)
        elif args.plot == 'w':
            payoffMatrix = np.array([[var ** 2, var*1j],[var, args.probsurvival]], dtype = np.complex128)

        if args.debug:
            print("payoff matrix (mutant is second row & column):")
            print(payoffMatrix)
            print("Initialized simulation. Time elapsed: " + str(timer() - start))

        if args.debug and (args.plot in ['s', 'h'] or args.swc > 0):
            print("Grid smallworldified. Time elapsed: " + str(timer() - start))
            print(adjGrid)

        # now that the adjGrid is set up, we can proceed to compute the fixation probability
        fdscp = FDSCP(dim, payoffMatrix, adjGrid, grid, dualAdjGrid)

        if args.binsearch:
            wc, data = binarySearch(args, fdscp, dim, start, folder, datestr)
            wcVals.append(wc)
        else:
            mutants, fixProb, stepsToFix, timeToFix = calcFixProb(args, fdscp, dim, start, folder, datestr)
            fixProbs.append(fixProb)
            fixTimes.append(timeToFix)
        
        if args.output >= 1:
            if args.binsearch:
                for guess in data:
                    outfile_avg.writelines(args.plot + " = " + strParam + ", w = " + str(guess) + ": fixation prob = " + str(data[guess][1]) + ", fixation time = " + str(data[guess][3]) + "\n")
                    print(args.plot + " = " + strParam + ", w = " + str(guess) + ": fixation prob = " + str(data[guess][1]) + ", fixation time = " + str(data[guess][3]))
                outfile_avg.writelines("Critical w value = " + str(wc) + "\n")
                print("Critical w value = " + str(wc))
            else:
                outfile_avg.writelines(args.plot + " = " + strParam + ": fixation prob = " + str(fixProb) + ", fixation time = " + str(timeToFix) + "\n")
                print(args.plot + " = " + strParam + ": fixation prob = " + str(fixProb) + ", fixation time = " + str(timeToFix))
        # make an output file of the range of different final mutant values in simulations (as opposed to simply the fixation probability)
        if args.output >= 2:
            if args.binsearch:
                for guess in data:
                    for i in range(len(data[guess][0])):
                        outfile_final = open(args.outfile + folder + "data2/" + args.plot + "=" + str(var) + "_w=" + str(guess) + datestr + ".txt", "a")
                        outfile_final.writelines("Run  mutants  time\n")
                        outfile_final.writelines(str(i) + "    " + str(data[guess][0][i]) + "     " + str(data[guess][2][i]) + "\n")
                        outfile_final.close()
            else:
                for i in range(len(mutants)):
                    outfile_final = open(args.outfile + folder + "data2/" + args.plot + "=" + str(var) + datestr + ".txt", "w")                
                    outfile_final.writelines("Run  mutants  time\n")                
                    outfile_final.writelines(str(i) + "    " + str(mutants[i]) + "     " + str(stepsToFix[i]) + "\n")
                    outfile_final.close()

        if args.debug:
            print("Finished outputting everything to files. Time elapsed: " + str(timer() - start))
        
    if args.output >= 1:
        outfile_avg.close()
    if args.binsearch:
        plt.plot(var_values, wcVals)
        plt.xlabel(args.plot)
        plt.ylabel('Critical w value')
        plt.show()
    else:
        plt.plot(var_values, fixProbs)
        plt.xlabel(args.plot)
        plt.ylabel('Fixation probability')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gene Drive Analysis Frontend",
                                 epilog="")

    parser.add_argument('-f', '--frac', help="Fraction of mutant cells at beginning",
                        type=float, default=0.35) #
    parser.add_argument('-m', '--initMutant', help="Number of mutant cells at beginning. Overrides -f if not -1 (defaults to 1).",
                        type=int, default=1)
    parser.add_argument('-fr', '--fourRegular', help="Use 4-regular lattice instead of 8-regular lattice.",
                        action='store_true', default=False)
    parser.add_argument('-r', '--rows', help="Number of rows of the grid",
                        type=int, default=32) #
    parser.add_argument('-c', "--cols", help="Number of columns of the grid",
                        type=int, default=64) #
    parser.add_argument('-e', "--extraspace",
                        help="Amount of extra space to add to adjacency grid (this should not be changed in most cases).",
                        type=int, default=5) #
    parser.add_argument('-v', "--visible", help="Number of steps to show grid",
                        type=int, default=-1) #
    parser.add_argument('-n', "--niters", help=("Number of times to run simulation "
                                                "per each value of small world "
                                                "coefficient"),
                        type=int, default=1) #
    parser.add_argument('-l', "--simlength", help="Length of each simulation",
                        type=int, default=-1) #
    parser.add_argument('-s', "--swc", help="Small world coefficient (for lattice only)",
                        type=float, default=0) #
    parser.add_argument('-o', "--output",
                        help=("Specify output format. Options:\n"
                            "0: do not output to file\n"
                            "1: output averages of final values across all "
                            "simulations per given small world coefficient\n"
                            "2: output final values for every simulation\n"
                            "3: output values for every step in every "
                            "simulation\n"
                            "4: output grid state at every step in every "
                            "simulation\n"
                            "Higher numbers also output everything for all lower"
                            " numbers, e.g. 3 will also output 2\n"),
                        type=int, default=0)
                        
    parser.add_argument('-rp', '--replace', help="Remove edges when constructing small world",
                        action="store_false", default=True)
                            
    parser.add_argument('-g', '--heterogeneity', help="Heterogeneity of SWN",
                        type=float, default=0)

    parser.add_argument('-d', "--delay", help="Time delay between steps",
                        type=float, default=0)
                        

    parser.add_argument('-ps', "--probsurvival", help="Probability of homozygous offspring surviving the embryo", type=float, default=1)

    parser.add_argument('-w', "--wtfitness", help="Fitness of wild-type (assuming fitness of gene drive is 1)", type=float, default=2)


    parser.add_argument('-sa', "--sample", help="When using output modes 3 or 4, how often should the grid be sampled?",
                        type=int, default=10)

    parser.add_argument('-of', "--outfile", help="Output folder to store data in", default="D:/OneDrive/Documents/genedrive_data/")

    parser.add_argument('-gr', "--graphFile",
                        help=("Name of file containing graph structure. If blank, loads Cartesian graph.\n"
                        "If enabled, this setting will override the row and column arguments.\n"
                        "This graph will still be small-worldified - if this is not desired, set small world coefficient to 0.\n"
                        "File should be structured as a sequence of lines A,B. Here A and B are vertices, and this line represents an edge from A to B.\n"
                        "There is currently no way to represent weighted graphs with this simulation.\n"
                        "If this option is used, the -v flag must not be used.\n"
                        "The graph-displaying function can currently display only grid-like graphs, not custom graphs.\n"),
                        default="")

    parser.add_argument('-p', '--plot', help=("Plot fixation probabilities of the given parameter"
                        "from 0 to its stated value.\n Fixation probabilities are computed by"
                        "running the simulation niters times for each value of the parameter.\n"
                        "Valid inputs are p, w, s, or h, for the parameters a, w,"
                        "the small-world coefficient of the graph, and the small-world heterogeneity."),
                        default="")

    parser.add_argument('-b', '--binsearch', help=("Binary search for the value of w such that FP = 1/n.\n"
                        "Searches w values from 0 to input w value (recommended value: 4).\n"
                        "Cannot be used in conjunction with '-p w' argument.\n"
                        "Can be used in conjunction with other plot parameters, in which case a separate binary search will be run"
                        "for each value of the parameter, and the plot will use the determined w values on the y-axis."),
                        default=False, action="store_true")

    parser.add_argument('--depth', help="Depth of binary search - accuracy will be w/2^depth", type=int, default=10)

    parser.add_argument('-min', '--min', help="Minimum value of parameter in the plot. Strongly advised to be positive if 'w' plot argument used.", type=float, default=0)

    parser.add_argument('-np', '--numpoints', help="Number of points to plot (only used when --plot argument is used)", type=int, default=51)

    parser.add_argument('-db', "--debug", help="Enter debug mode (prints more stuff to output)",
                        action='store_true', default=False)

    parser.add_argument('-t', "--test", help="Activate unit tests (may increase computation time)", action='store_true', default=False)
    


    args = parser.parse_args()
    main(args)
